# Refer to:
# http://fatalfeel.blogspot.com/2013/12/ppo-and-awr-guiding.html #PPO lessons
# https://towardsdatascience.com/solving-lunar-lander-openaigym-reinforcement-learning-785675066197
# https://pytorch.org/docs/stable/distributions.html
# check 8 states and 4 actions at ./gym/envs/box2d/lunar_lander.py
# python3 ./keyboard_agent.py LunarLander-v2 #play use key 1,2,3
'''The Scenario
s[0] is the horizontal coordinate
s[1] is the vertical coordinate
s[2] is the horizontal speed
s[3] is the vertical speed
s[4] is the angle
s[5] is the angular speed
s[6] 1 if first leg has contact
s[7] 1 if second leg has contact
The 8 elements present to one state
When those elements happened we need do one action and get the highest reward
key 1 - right engine
key 2 - main engine
key 3 - left engine
key 4 - nope'''
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

device = torch.device("cuda" if args.cuda else "cpu")

class GameContent:
    def __init__(self):
        self.actions        = []
        self.states         = []
        self.rewards        = []
        self.actorlogprobs  = []
        self.is_terminals   = []
    
    def ReleaseData(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.actorlogprobs[:]
        del self.is_terminals[:]

class Actor_Critic(nn.Module):
    def __init__(self, dim_states, dim_acts, h_neurons):
        super(Actor_Critic, self).__init__()

        ###game is env = gym.make(env_name)
        ###game-state combine with 8 elements that describe in lunar_lander.py
        '''In policy_curr network_act is first-person perspective actor playing in the game. 
           when game state input then action prob out.
        (1) game-state -> first-persion network_act => prob(r, main, left, nope)
        (2) According to prob(r, main, left, nope), pick one of direction(r, main, left, nope) into game
            the code is env.step(action)
        (3) env.step(action) return the rewards'''

        '''In policy_next self.network_act is second-person perspective actor observer in the game
        (1) game-state -> second-persion network_act => prob(r, main, left, nope)
        (2) use samples of first-person actions put in distribute.log_prob(actions) get critic actorlogprobs
        (3) ratios = e^log(critic_log_prob/currenr_log_prob) = e^(logcritic_log_prob-currenr_log_prob)'''
        self.network_act = nn.Sequential(nn.Linear(dim_states, h_neurons),
                                         nn.Tanh(),
                                         nn.Linear(h_neurons, h_neurons),
                                         nn.Tanh(),
                                         nn.Linear(h_neurons, dim_acts))

        '''critic network is second-person perspective actor observer in the game, when game state in then reward out
        (1)game-state -> second-persion network -> Get Reward that call Value = V(s)'''
        self.network_critic = nn.Sequential(nn.Linear(dim_states, h_neurons),
                                            nn.Tanh(),
                                            nn.Linear(h_neurons, h_neurons),
                                            nn.Tanh(),
                                            nn.Linear(h_neurons, 1))
        
    def forward(self):
        raise NotImplementedError

    #https://pytorch.org/docs/stable/distributions.html
    #Categorical distribution follow actor_prob which sum is 1.0 then sample out your action also do entropy
    def interact(self, envstate, gamedata):
        torchstate  = torch.from_numpy(envstate).double().to(device)
        actor_prob  = self.network_act(torchstate) #tau(a|s) = P(a,s) 8 elements corresponds to one action
        actor_prob  = torch.softmax(actor_prob, dim=-1).double()
        distribute  = torch.distributions.Categorical(actor_prob)
        action      = distribute.sample()
        actlogprob  = distribute.log_prob(action) #logeX

        gamedata.states.append(torchstate)
        gamedata.actions.append(action)
        gamedata.actorlogprobs.append(actlogprob) #the action number corresponds to the action_probs into log

        return action.detach().item() #return action_probs index corresponds to key 1,2,3,4

    #policy_ac.calculation will call
    #sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
    #usually mini_batch_size sample < states'size do forward
    #in our example the mini_batch_size = states'size
    def calculation(self, states, actions):
        actor_probs         = self.network_act(states) #each current with one action probility
        actor_probs         = torch.softmax(actor_probs, dim=-1).double()
        distribute          = torch.distributions.Categorical(actor_probs)
        epoch_actlogprobs   = distribute.log_prob(actions) #natural log prob
        entropy             = distribute.entropy() # entropy is uncertain percentage, value higher mean uncertain more
        critic_values       = self.network_critic(states) #c_values is V(s) in A3C theroy

        #if dimension can squeeze then tensor 3d to 2d.
        #EX: squeeze tensor[2,1,3] become to tensor[2,3]
        return epoch_actlogprobs, torch.squeeze(critic_values), entropy

    # if is_terminals is false use Markov formula to replace last reward
    def GetNextValue(self, next_state, is_terminals):
        ''' self.returns[-1] = next_value (next_value from critic network)
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1] * self.masks[step + 1] '''
        if is_terminals is False:
            torchstate = torch.from_numpy(next_state).double().to(device)
            next_value = self.network_critic(torchstate)
            next_value = next_value.detach().cpu().numpy()[0]
        else:
            next_value = 0.0

        return next_value

class CPPO:
    def __init__(self, dim_states, dim_acts, h_neurons, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef):
        self.lr             = lr
        self.betas          = betas
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.vloss_coef     = vloss_coef
        self.entropy_coef   = entropy_coef
        self.train_epochs   = train_epochs

        self.policy_ac      = Actor_Critic(dim_states, dim_acts, h_neurons).double().to(device)
        self.optimizer      = torch.optim.Adam(self.policy_ac.parameters(), lr=lr, betas=betas)

    def train_update(self, gamedata, next_value):
        returns             = []
        discounted_reward   = next_value
        # Monte Carlo estimate of state rewards:
        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # R(τ) = gamma^n * τ(a|s)R(a,s) , n=1~k
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward) #always insert in the first

        returns = torch.tensor(returns).double().to(device)

        '''returns.mean() is E[R(τ)]
        returns.std on torch is {1/(n-1) * Σ(x - x_average)} ** 0.5  (x ** 0.5 = x^0.5)
        1e-5 = 0.00001 which avoid returns.std() is zero
        (returns - average_R) / (standard_R + 0.00001) is standard score
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)'''
        # should modify
        #curraccu_stdscore   = (returns - returns.mean()) / (returns.std() + 1e-5) #Q(s,a)

        # convert list to tensor
        # torch.stack is combine many tensor 1D to 2D
        old_states      = torch.stack(gamedata.states).double().to(device).detach()
        old_actions     = torch.stack(gamedata.actions).double().to(device).detach()
        old_actlogprobs = torch.stack(gamedata.actorlogprobs).double().to(device).detach()

        # critic_state_reward   = network_critic(curraccu_states)
        '''refer to a2c-ppo should modify like this
           advantages   = rollouts.returns[:-1] - rollouts.value_preds[:-1]
           advantages   = (advantages - advantages.mean()) / (advantages.std() + 1e-5)'''
        old_values  = self.policy_ac.network_critic(old_states) #faster than do every times in interact
        old_values  = torch.squeeze(old_values)
        rv_diff     = returns - old_values.detach()  # A(s,a) => Q(s,a) - V(s), V(s) is critic
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
            ratios  = torch.exp(epoch_actlogprobs - old_actlogprobs.detach())

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
            value_predict_clip  = old_values.detach() + (critic_values - old_values.detach()).clamp(-self.eps_clip, self.eps_clip)
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
    env_name        = "LunarLander-v2"
    render          = False
    solved_reward   = 285           # don't change the topest avg score if more can not reach
    h_neurons       = 1024          # number of variables in hidden layer
    max_episodes    = 200000        # max training episodes
    max_timesteps   = 400           # max timesteps in one episode
    train_epochs    = 4             # train_update policy for K epochs
    update_timestep = 1600          # train_update samples need 8*avg_length
    log_interval    = 20            # print avg reward in the interval
    lr              = 0.0001        # parameters for learning rate
    betas           = (0.9, 0.999)  # Adam β
    gamma           = 0.99          # discount factor
    eps_clip        = 0.2           # clip parameter for PPO2
    vloss_coef      = 0.5           # clip parameter for PPO2
    entropy_coef    = 0.01
    s_episode       = 1
    #############################################
    #if not os.path.exists(args.checkpoint_dir):
    #    os.makedirs(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]  # LunarLander give 8 states
    dim_acts    = 4  # 4 action directions

    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, h_neurons, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef)
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
    for i_episode in range(s_episode, max_episodes+1):
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
                next_value = ppo.policy_ac.GetNextValue(envstate, gamedata.is_terminals[-1])
                ppo.train_update(gamedata, next_value)
                gamedata.ReleaseData()
                timestep = 0

            if render:
                env.render()

            if done:
                break

        total_length += (ts+1)

        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            avg_length  = int(total_length/log_interval)
            avg_reward  = int((running_reward/log_interval))
            print('Episode {} \t avg length: {} \t avg reward: {}'.format(i_episode, avg_length, avg_reward))
            checkpoint = {'state_dict':     ppo.policy_ac.state_dict(),
                          'optimizer_dict': ppo.optimizer.state_dict(),
                          'episode':        i_episode}
            lastname = args.checkpoint_dir + '/PPO_{}_last.pth'.format(env_name)
            torch.save(checkpoint, lastname)
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            checkpoint = {'state_dict':     ppo.policy_ac.state_dict(),
                          'optimizer_dict': ppo.optimizer.state_dict(),
                          'episode':        i_episode}
            pname       = args.checkpoint_dir + '/PPO_{}_episode_{}.pth'.format(env_name, i_episode)
            torch.save(checkpoint, pname)
            lastname    = args.checkpoint_dir + '/PPO_{}_last.pth'.format(env_name)
            torch.save(checkpoint, lastname)

        if i_episode % log_interval == 0:
            avg_length  = int(total_length/log_interval)
            avg_reward  = int((running_reward/log_interval))
            print('Episode {} \t avg length: {} \t avg reward: {}'.format(i_episode, avg_length, avg_reward))
            running_reward  = 0
            total_length    = 0
