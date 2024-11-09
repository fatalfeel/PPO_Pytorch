import argparse
import os

import torch
import torch.nn as nn
import torch.distributions
import gym

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str,       default='./checkpoint', help='path to checkpoint')
parser.add_argument('--resume',         type=str2bool,  default=False)
parser.add_argument('--cuda',           type=str2bool,  default=False)
args = parser.parse_args()

class GameContent:
    def __init__(self):
        self.actions        = []
        self.states         = []
        self.actorlogprobs  = []
        self.rewards        = []
        self.is_terminals   = []

    def ReleaseData(self):
        del self.actions[:]
        del self.states[:]
        del self.actorlogprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor_Critic(nn.Module):
    def __init__(self, dim_states, dim_acts, action_std, h_neurons, device):
        super(Actor_Critic, self).__init__()
        # action mean range -1 to 1
        self.network_act    = nn.Sequential(nn.Linear(dim_states, h_neurons),
                                            nn.Tanh(),
                                            nn.Linear(h_neurons, h_neurons),
                                            nn.Tanh(),
                                            nn.Linear(h_neurons, dim_acts),
                                            nn.Tanh()).double().to(device) #must add nn.Tanh in tail for normal mu
        # network_value
        self.network_critic = nn.Sequential(nn.Linear(dim_states, h_neurons),
                                            nn.Tanh(),
                                            nn.Linear(h_neurons, h_neurons),
                                            nn.Tanh(),
                                            nn.Linear(h_neurons, 1)).double().to(device)

        #self.action_std = torch.full((dim_acts,), action_std*action_std).double().to(device)
        self.action_std     = torch.full((dim_acts,), action_std).double().to(device) #standard deviations
        self.device         = device

    def forward(self):
        raise NotImplementedError

    # https://pytorch.org/docs/stable/distributions.html
    # backpropagation conditions are continue and differential. Sampling probs need in one distribution
    def interact(self, envstate, gamedata):
        torchstate      = torch.DoubleTensor(envstate.reshape(1, -1)).to(self.device) #reshape(1,-1) 1d to 2d
        act_mu          = self.network_act(torchstate)
        std_mat         = torch.diag(self.action_std).double().to(self.device) #transfer to matrix
        #distribute     = torch.distributions.MultivariateNormal(action_mu, cov_mat)
        distribute      = torch.distributions.MultivariateNormal(act_mu, scale_tril=std_mat) #act_mu=center, scale_tril=width
        action          = distribute.sample()
        actlogprob      = distribute.log_prob(action) #logeX

        gamedata.states.append(torchstate)
        gamedata.actions.append(action)
        gamedata.actorlogprobs.append(actlogprob)

        #flatten do 2d to 1d
        return action.detach().cpu().data.numpy().flatten()

    # policy_ac.calculation will call
    # sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
    # usually mini_batch_size sample < states'size do forward
    # in our example the mini_batch_size = states'size
    def calculation(self, states, actions):
        acts_mu             = self.network_act(states)
        acts_std            = self.action_std.expand_as(acts_mu)
        mats_std            = torch.diag_embed(acts_std).double().to(self.device)
        #distribute         = torch.distributions.MultivariateNormal(action_mu, cov_mat)
        distribute          = torch.distributions.MultivariateNormal(acts_mu, scale_tril=mats_std) #act_mu=center, scale_tril=width
        epoch_actlogprobs   = distribute.log_prob(actions) #natural log prob
        entropy             = distribute.entropy() #entropy is uncertain percentage, value higher mean uncertain more
        critic_values       = self.network_critic(states) #c_values is V(s) in A3C theroy

        #if dimension can squeeze then tensor 3d to 2d.
        #EX: squeeze tensor[2,1,3] become to tensor[2,3]
        return epoch_actlogprobs, torch.squeeze(critic_values), entropy

    # if is_terminals is false use Markov formula to replace last reward
    def getNextValue(self, next_state, is_terminals):
        ''' self.returns[-1] = next_value (next_value from critic network)
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1] * self.masks[step + 1] '''
        if is_terminals is False:
            torchstate = torch.DoubleTensor(next_state.reshape(1, -1)).to(self.device)  # reshape(1,-1) 1d to 2d
            next_value = self.network_critic(torchstate)
            next_value = next_value.detach().cpu().numpy()[0,0]
        else:
            next_value = 0.0

        return next_value

class CPPO:
    def __init__(self, dim_states, dim_acts, action_std, h_neurons, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef, device):
        self.lr             = lr
        self.betas          = betas
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.vloss_coef     = vloss_coef
        self.entropy_coef   = entropy_coef
        self.train_epochs   = train_epochs

        self.policy_ac      = Actor_Critic(dim_states, dim_acts, action_std, h_neurons, device)
        self.optimizer      = torch.optim.Adam(self.policy_ac.parameters(), lr=lr, betas=betas)
        self.device         = device

    def train_update(self, gamedata, next_value):
        returns         = []
        discount_reward = next_value
        # Monte Carlo estimate of state rewards:
        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discount_reward = 0
            # R(τ) = gamma^n * τ(a|s)R(a,s) , n=1~k
            discount_reward = reward + (self.gamma * discount_reward)
            returns.insert(0, discount_reward) #always insert in the first

        returns = torch.tensor(returns).double().to(self.device)

        # convert list to tensor
        old_states         = torch.squeeze(torch.stack(gamedata.states).double().to(self.device), 1).detach()
        old_actions        = torch.squeeze(torch.stack(gamedata.actions).double().to(self.device), 1).detach()
        old_actlogprobs    = torch.squeeze(torch.stack(gamedata.actorlogprobs).double().to(self.device), 1).detach()

        # critic_state_reward   = network_critic(curraccu_states)
        '''refer to a2c-ppo should modify like this
           advantages   = rollouts.returns[:-1] - rollouts.value_preds[:-1]
           advantages   = (advantages - advantages.mean()) / (advantages.std() + 1e-5)'''
        old_values  = self.policy_ac.network_critic(old_states) #faster than do every times in interact
        old_values  = torch.squeeze(old_values).detach()
        rv_diff     = returns - old_values # A(s,a) => Q(s,a) - V(s), V(s) is critic
        advantages  = (rv_diff - rv_diff.mean()) / (rv_diff.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.train_epochs):
            #cstate_value is V(s) in A3C theroy. critic network weights as an actor feed state out
            epoch_actlogprobs, critic_values, entropy = self.policy_ac.calculation(old_states, old_actions)

            # https://socratic.org/questions/what-is-the-derivative-of-e-lnx
            # log(critic) - log(curraccu) = log(critic/curraccu)
            # ratios  = e^(State2_natureLogActProbs-Stat1_natureLogActProbs)
            # ratios  = State2_natureLogActProbs/State1_natureLogActProbs
            # ratios  = next_critic_actprobs/curr_actions_prob = Pw(A1|S2)/Pw(A1|S1), where w is weights(theta)
            ratios  = torch.exp(epoch_actlogprobs - old_actlogprobs)

            #advantages is stdscore mode
            surr1   = ratios * advantages
            surr2   = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            #we get all samples of critic_values, so value_preds_batch equal critic_values
            ''' value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_losses = (values - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,value_losses_clipped).mean() '''
            # value_predict_clip is critical predict value + (critical predict value - originla critical value)|range between(-param~+param)
            # value_predict_loss is (value_predict_clip - MDP-reward)^2
            # value_critic_loss is (critical predict value - MDP-reward)^2
            # value_loss is 0.5 x select max items in (predict_loss or value_critic_loss) ex: A=[2,6] b=[4,5] torch max=>[4,6]
            value_predict_clip  = old_values + (critic_values - old_values).clamp(-self.eps_clip, self.eps_clip)
            value_predict_loss  = (value_predict_clip - returns) ** 2
            value_critic_loss   = (critic_values - returns) ** 2
            value_loss          = 0.5 * torch.max(value_predict_loss, value_critic_loss)

            # MseLoss is Mean Square Error = (target - output)^2, critic_values in first param follow libtorch rules
            # loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(critic_values, returns) - 0.01*entropy
            loss = -torch.min(surr1, surr2).mean() + self.vloss_coef * value_loss.mean() - self.entropy_coef * entropy.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()         #get grade.data
            self.optimizer.step()   #update grade.data by adam method which is smooth grade

        # Copy new weights into old policy:
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "BipedalWalker-v3"
    render          = False
    solved_reward   = 280           # don't change the topest avg score if more can not reach
    h_neurons       = 1024          # number of variables in hidden layer
    max_episodes    = 200000        # max training episodes
    max_timesteps   = 1300          # max timesteps in one episode
    train_epochs    = 10            # train_update policy for K epochs
    update_timestep = 8000          # train_update samples need 8*best_length(1000)
    epoch_save_freq = 100
    log_interval    = 10            # print avg reward in the interval
    lr              = 0.0001        # parameters for learning rate
    betas           = (0.9, 0.999)  # Adam β
    gamma           = 0.99          # discount factor
    eps_clip        = 0.2           # clip parameter for PPO2
    vloss_coef      = 0.5           # clip parameter for PPO2
    entropy_coef    = 0.01
    action_std      = 0.5           # constant std for action distribution (Multivariate Normal)
    s_episode       = 0
    #############################################
    device = torch.device("cuda" if args.cuda else "cpu")
    #if not os.path.exists(args.checkpoint_dir):
    #    os.makedirs(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]
    dim_acts    = env.action_space.shape[0]

    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, action_std, h_neurons, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef, device)
    ppo.policy_ac.train()

    if args.resume:
        lastname    = args.checkpoint_dir + '/PPO_{}_last.pth'.format(env_name)
        checkpoint  = torch.load(lastname)
        ppo.policy_ac.load_state_dict(checkpoint['state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_dict'])
        s_episode   = checkpoint['episode']

    # logging variables
    ts              = 0
    timestep        = 0
    running_reward  = 0
    total_length    = 0

    # training loop
    for i_episode in range(s_episode+1, max_episodes+1):
        envstate = env.reset() #Done-0 State-0
        for ts in range(max_timesteps):
            timestep += 1

            # Running policy_current: #Done-0 State-0 Act-0
            action = ppo.policy_ac.interact(envstate, gamedata)

            # Done-1 State-1 Act-0 R-0
            envstate, reward, done, _ = env.step(action)

            running_reward += reward

            # one reward R(τ) = τ(a|s)R(a,s) in a certain state select an action and return the reward
            gamedata.rewards.append(reward)

            # is_terminal in next state:
            gamedata.is_terminals.append(done)

            # train_update if its time
            if timestep >= update_timestep:
                next_value = ppo.policy_ac.getNextValue(envstate, gamedata.is_terminals[-1])
                ppo.train_update(gamedata, next_value)
                gamedata.ReleaseData()
                timestep = 0

            if render:
                env.render()

            if done:
                break

        total_length += (ts+1)

        if i_episode % log_interval == 0:
            avg_length      = int(total_length/log_interval)
            avg_reward      = int((running_reward/log_interval))
            total_length    = 0
            running_reward  = 0
            print('Episode {} \t avg length: {} \t avg reward: {}'.format(i_episode, avg_length, avg_reward))
            if avg_reward > solved_reward:
                print("########## Solved! ##########")
                checkpoint  = {'state_dict': ppo.policy_ac.state_dict(),
                               'optimizer_dict': ppo.optimizer.state_dict(),
                               'episode': i_episode}
                pname       = args.checkpoint_dir + '/PPO_{}_episode_{}.pth'.format(env_name, i_episode)
                lastname    = args.checkpoint_dir + '/PPO_{}_last.pth'.format(env_name)
                torch.save(checkpoint, pname)
                torch.save(checkpoint, lastname)
                print('save ' + pname)
                break

        # save every epoch_save_freq episodes
        if i_episode % epoch_save_freq == 0:
            checkpoint  = {'state_dict':     ppo.policy_ac.state_dict(),
                           'optimizer_dict': ppo.optimizer.state_dict(),
                           'episode':        i_episode}
            pname       = args.checkpoint_dir + '/PPO_{}_episode_{}.pth'.format(env_name, i_episode)
            lastname    = args.checkpoint_dir + '/PPO_{}_last.pth'.format(env_name)
            torch.save(checkpoint, pname)
            torch.save(checkpoint, lastname)
            print('save ' + pname)
