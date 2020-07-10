import torch
import torch.nn as nn
import torch.distributions
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GameContent:
    def __init__(self):
        self.actions        = []
        self.states         = []
        self.rewards        = []
        self.logprobs       = []
        self.is_terminals   = []
    
    def ReleaseData(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.is_terminals[:]

class Actor_Critic(nn.Module):
    def __init__(self, dim_states, dim_acts, h_neurons):
        super(Actor_Critic, self).__init__()

        # actor
        self.network_act = nn.Sequential(nn.Linear(dim_states, h_neurons),
                                        nn.Tanh(),
                                        nn.Linear(h_neurons, h_neurons),
                                        nn.Tanh(),
                                        nn.Linear(h_neurons, dim_acts),
                                        nn.Softmax(dim=-1))
        
        # critic
        self.network_value = nn.Sequential(nn.Linear(dim_states, h_neurons),
                                        nn.Tanh(),
                                        nn.Linear(h_neurons, h_neurons),
                                        nn.Tanh(),
                                        nn.Linear(h_neurons, 1))
        
    def forward(self):
        raise NotImplementedError
        
    def interact(self, estates, gamedata):
        tstates        = torch.from_numpy(estates).float().to(device)
        action_probs    = self.network_act(tstates)
        dist            = torch.distributions.Categorical(action_probs)
        action          = dist.sample()
        
        gamedata.states.append(tstates)
        gamedata.actions.append(action)
        gamedata.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def calculation(self, states, actions):
        action_probs    = self.network_act(states)
        dist            = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy    = dist.entropy()
        state_value     = self.network_value(states)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class CPPO:
    def __init__(self, dim_states, dim_acts, h_neurons, lr, gamma, train_epochs, eps_clip, betas):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.train_epochs = train_epochs
        
        self.policy     = Actor_Critic(dim_states, dim_acts, h_neurons).to(device)
        self.optimizer  = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = Actor_Critic(dim_states, dim_acts, h_neurons).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mseLoss    = nn.MSELoss()
    
    def train_update(self, gamedata):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states      = torch.stack(gamedata.states).to(device).detach()
        old_actions     = torch.stack(gamedata.actions).to(device).detach()
        old_logprobs    = torch.stack(gamedata.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.train_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.calculation(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.mseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()  #get grade.data
            self.optimizer.step()   #update grade.data by adam method which is smooth grade

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "LunarLander-v2"
    # creating environment
    render          = False
    solved_reward   = 230           # stop training if reach avg_reward > solved_reward
    log_interval    = 20            # print avg reward in the interval
    max_episodes    = 50000         # max training episodes
    max_timesteps   = 300           # max timesteps in one episode
    h_neurons       = 64            # number of variables in hidden layer
    update_timestep = 2000          # train_update policy every n timesteps
    gamma           = 0.99          # discount factor
    train_epochs    = 4             # train_update policy for epochs
    eps_clip        = 0.2           # clip parameter for PPO2
    lr              = 0.002         # learning rate
    betas           = (0.9, 0.999)  # Adam β
    random_seed     = None
    #############################################

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]  # LunarLander give 8 states
    dim_acts    = 4  # 4 action directions

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, h_neurons, lr, gamma, train_epochs, eps_clip, betas)
    
    # logging variables
    running_reward  = 0
    avg_length      = 0
    timestep        = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        estates = env.reset() #init state value to matrix
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.interact(estates, gamedata)
            estates, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            gamedata.rewards.append(reward)
            gamedata.is_terminals.append(done)

            # train_update if its time
            if timestep % update_timestep == 0:
                ppo.train_update(gamedata)
                gamedata.ReleaseData()
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
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
