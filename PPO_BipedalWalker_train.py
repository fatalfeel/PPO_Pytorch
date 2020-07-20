import torch
import torch.nn as nn
import torch.distributions
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GameContent:
    def __init__(self):
        self.actions        = []
        self.states         = []
        self.actorlogprobs  = []
        self.rewards        = []
        self.is_terminals   = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.actorlogprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor_Critic(nn.Module):
    def __init__(self, dim_states, dim_acts, action_std):
        super(Actor_Critic, self).__init__()
        # action mean range -1 to 1
        self.network_act =  nn.Sequential(  nn.Linear(dim_states, 64),
                                            nn.Tanh(),
                                            nn.Linear(64, 32),
                                            nn.Tanh(),
                                            nn.Linear(32, dim_acts),
                                            nn.Tanh()  )
        # network_value
        self.network_critic = nn.Sequential(nn.Linear(dim_states, 64),
                                            nn.Tanh(),
                                            nn.Linear(64, 32),
                                            nn.Tanh(),
                                            nn.Linear(32, 1) )

        #self.action_std = torch.full((dim_acts,), action_std*action_std).double().to(device)
        self.action_std  = torch.full((dim_acts,), action_std).double().to(device) #standard deviations

    def forward(self):
        raise NotImplementedError

    # https://pytorch.org/docs/stable/distributions.html
    # backpropagation conditions are continue and differential. Sampling probs need in one distribution
    def interact(self, envstate, gamedata):
        torchstate      = torch.FloatTensor(envstate.reshape(1, -1)).double().to(device) #reshape(1,-1) 1d to 2d
        act_mu          = self.network_act(torchstate)
        std_mat         = torch.diag(self.action_std).double().to(device) #transfer to matrix
        #distribute     = torch.distributions.MultivariateNormal(action_mu, cov_mat)
        distribute      = torch.distributions.MultivariateNormal(act_mu, scale_tril=std_mat) #act_mu=center, scale_tril=width
        action          = distribute.sample()
        actlogprob      = distribute.log_prob(action) #logeX

        gamedata.states.append(torchstate)
        gamedata.actions.append(action)
        gamedata.actorlogprobs.append(actlogprob)

        #flatten do 2d to 1d
        return action.detach().cpu().data.numpy().flatten()

    def calculation(self, states, actions):
        acts_mu             = self.network_act(states)
        acts_std            = self.action_std.expand_as(acts_mu)
        std_mat             = torch.diag_embed(acts_std).double().to(device)
        #distribute         = torch.distributions.MultivariateNormal(action_mu, cov_mat)
        distribute          = torch.distributions.MultivariateNormal(acts_mu, scale_tril=std_mat) #act_mu=center, scale_tril=width
        critic_actlogprobs  = distribute.log_prob(actions) #logeX
        entropy             = distribute.entropy() #entropy is uncertain percentage, value higher mean uncertain more
        next_critic_values  = self.network_critic(states) #c_values is V(s) in A3C theroy

        '''future using'''
        #states_sampling = None
        #sampler = BatchSampler(SubsetRandomSampler(range(states.size()[0])), states.size()[0], drop_last=False)
        #for indices in sampler:
        #    states_sampling = states[indices]
        #next_critic_values = self.network_critic(states_sampling)  # c_values is V(s) in A3C theroy
        #next_critic_actprobs= critic_actprobs.gather(1, actions.unsqueeze(1).type(torch.int64))
        #next_critic_actprobs= torch.squeeze(next_critic_actprobs)

        #if dimension can squeeze then tensor 3d to 2d.
        #EX: squeeze tensor[2,1,3] become to tensor[2,3]
        return critic_actlogprobs, torch.squeeze(next_critic_values), entropy

class CPPO:
    def __init__(self, dim_states, dim_acts, action_std, lr, gamma, train_epochs, eps_clip, betas):
        self.lr             = lr
        self.betas          = betas
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.train_epochs   = train_epochs

        self.policy_ac      = Actor_Critic(dim_states, dim_acts, action_std).double().to(device)
        self.optimizer      = torch.optim.Adam(self.policy_ac.parameters(), lr=lr, betas=betas)

        #self.policy_curr = Actor_Critic(dim_states, dim_acts, action_std).double().to(device)
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

        self.MseLoss = nn.MSELoss(reduction='mean')

    #def select_action(self, estates, gamedata):
    #    tstates = torch.FloatTensor(estates.reshape(1, -1)).double().to(device)
    #    return self.policy_curr.interact(tstates, gamedata).cpu().data.numpy().flatten()

    def train_update(self, gamedata):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).double().to(device)

        # convert list to tensor
        curr_states         = torch.squeeze(torch.stack(gamedata.states).double().to(device), 1).detach()
        curr_actions        = torch.squeeze(torch.stack(gamedata.actions).double().to(device), 1).detach()
        curr_actlogprobs    = torch.squeeze(torch.stack(gamedata.actorlogprobs).double().to(device), 1).detach()

        # critic_state_reward   = network_critic(curraccu_states)
        '''refer to a2c-ppo should modify like this
           advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
           advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)'''
        critic_vpi  = self.policy_ac.network_critic(curr_states)
        critic_vpi  = torch.squeeze(critic_vpi)
        qsa_sub_vs  = rewards - critic_vpi.detach()  # A(s,a) => Q(s,a) - V(s), V(s) is critic
        advantages  = (qsa_sub_vs - qsa_sub_vs.mean()) / (qsa_sub_vs.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.train_epochs):
            #cstate_value is V(s) in A3C theroy. critic network is another actor input state
            critic_actlogprobs, next_critic_values, entropy = self.policy_ac.calculation(curr_states, curr_actions)

            # https://socratic.org/questions/what-is-the-derivative-of-e-lnx
            # log(critic) - log(curraccu) = log(critic/curraccu)
            # ratios  = e^(ln(State2_actProbs)-ln(State1_actProbs)) =  e^ln(State2_actProbs/State1_actProbs)
            # ratios  = (State2_critic_actProbs/State1_actor_actProbs)
            # ratios  = next_critic_actprobs/curr_actions_prob = Pw(A1|S2)/Pw(A1|S1), where w is weights(theta)
            ratios  = torch.exp(critic_actlogprobs - curr_actlogprobs.detach())

            #advantages is stdscore mode
            surr1   = ratios * advantages
            surr2   = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # mseLoss is Mean Square Error = (target - output)^2
            loss    = -torch.min(surr1, surr2) + 0.5*self.MseLoss(rewards, next_critic_values) - 0.01*entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "BipedalWalker-v3"
    render          = False
    solved_reward   = 300           # stop training if avg_reward > solved_reward
    log_interval    = 20            # print avg reward in the interval
    max_episodes    = 50000         # max training episodes
    max_timesteps   = 1500          # max timesteps in one episode
    update_timestep = 4000          # train_update policy every n timesteps
    train_epochs    = 40            # train_update policy for K epochs
    action_std      = 0.5           # constant std for action distribution (Multivariate Normal)
    gamma           = 0.99          # discount factor
    lr              = 0.0001        # parameters for Adam optimizer
    eps_clip        = 0.2           # clip parameter for CPPO
    betas           = (0.9, 0.999)
    random_seed     = None
    #############################################

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]
    dim_acts    = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, action_std, lr, gamma, train_epochs, eps_clip, betas)

    # logging variables
    running_reward  = 0
    avg_length      = 0
    timestep        = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        envstate = env.reset()
        for t in range(max_timesteps):
            timestep +=1
            # Running policy_old:
            #action = ppo.select_action(estates, gamedata)
            action = ppo.policy_ac.interact(envstate, gamedata)
            envstate, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            gamedata.rewards.append(reward)
            gamedata.is_terminals.append(done)

            # train_update if its time
            if timestep % update_timestep == 0:
                ppo.train_update(gamedata)
                gamedata.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy_ac.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy_ac.state_dict(), './PPO_{}_episode_{}.pth'.format(env_name, i_episode))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
