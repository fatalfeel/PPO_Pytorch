import torch
import torch.nn as nn
import torch.distributions
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GameContent:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
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
        self.network_value = nn.Sequential( nn.Linear(dim_states, 64),
                                            nn.Tanh(),
                                            nn.Linear(64, 32),
                                            nn.Tanh(),
                                            nn.Linear(32, 1) )
        
        self.action_var = torch.full((dim_acts,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def interact(self, envstate, gamedata):
        tstates         = torch.FloatTensor(envstate.reshape(1, -1)).to(device)
        action_mean     = self.network_act(tstates)
        cov_mat         = torch.diag(self.action_var).to(device)
        dist            = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        action          = dist.sample()
        action_logprob  = dist.log_prob(action)
        
        gamedata.states.append(tstates)
        gamedata.actions.append(action)
        gamedata.logprobs.append(action_logprob)

        return action.detach().cpu().data.numpy().flatten()

    def calculation(self, states, actions):
        action_mean     = self.network_act(states)
        action_var      = self.action_var.expand_as(action_mean)
        cov_mat         = torch.diag_embed(action_var).to(device)
        dist            = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy    = dist.entropy()
        cstate_reward   = self.network_value(states)
        
        return action_logprobs, torch.squeeze(cstate_reward), dist_entropy

class CPPO:
    def __init__(self, dim_states, dim_acts, action_std, lr, gamma, train_epochs, eps_clip, betas):
        self.lr             = lr
        self.betas          = betas
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.train_epochs   = train_epochs
        
        self.policy         = Actor_Critic(dim_states, dim_acts, action_std).to(device)
        self.optimizer      = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = Actor_Critic(dim_states, dim_acts, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    #def select_action(self, estates, gamedata):
    #    tstates = torch.FloatTensor(estates.reshape(1, -1)).to(device)
    #    return self.policy_old.interact(tstates, gamedata).cpu().data.numpy().flatten()
    
    def train_update(self, gamedata):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards             = torch.tensor(rewards, dtype=torch.float32).to(device)
        curraccu_stdscore   = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        curr_states      = torch.squeeze(torch.stack(gamedata.states).to(device), 1).detach()
        curr_actions     = torch.squeeze(torch.stack(gamedata.actions).to(device), 1).detach()
        curr_logprobs    = torch.squeeze(torch.stack(gamedata.logprobs), 1).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.train_epochs):
            # Evaluating old actions and values :
            critic_actlogprobs, cstate_reward, dist_entropy = self.policy.calculation(curr_states, curr_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(critic_actlogprobs - curr_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = curraccu_stdscore - cstate_reward.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(cstate_reward, curraccu_stdscore) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "BipedalWalker-v3"
    render          = False
    solved_reward   = 300         # stop training if avg_reward > solved_reward
    log_interval    = 20           # print avg reward in the interval
    max_episodes    = 10000        # max training episodes
    max_timesteps   = 1500        # max timesteps in one episode
    update_timestep = 4000      # train_update policy every n timesteps
    action_std      = 0.5            # constant std for action distribution (Multivariate Normal)
    train_epochs    = 80               # train_update policy for K epochs
    eps_clip        = 0.2              # clip parameter for CPPO
    gamma           = 0.99                # discount factor
    lr              = 0.0003                 # parameters for Adam optimizer
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
    time_step       = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        envstate = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            #action = ppo.select_action(estates, gamedata)
            action = ppo.policy_old.interact(envstate, gamedata)
            envstate, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            gamedata.rewards.append(reward)
            gamedata.is_terminals.append(done)
            
            # train_update if its time
            if time_step % update_timestep == 0:
                ppo.train_update(gamedata)
                gamedata.clear_memory()
                time_step = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_{}_episode_{}.pth'.format(env_name, i_episode))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
