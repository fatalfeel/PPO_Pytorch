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
        self.actoflogprobs  = []
        self.is_terminals   = []
    
    def ReleaseData(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.actoflogprobs[:]
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
        (2) use samples of first-person actions put in distribute.log_prob(actions) get critic actlogprobs
        (3) ratios = e^log(critic_log_prob/currenr_log_prob) = e^(logcritic_log_prob-currenr_log_prob)'''
        self.network_act = nn.Sequential(nn.Linear(dim_states, h_neurons),
                                        nn.Tanh(),
                                        nn.Linear(h_neurons, h_neurons),
                                        nn.Tanh(),
                                        nn.Linear(h_neurons, dim_acts),
                                        nn.Softmax(dim=-1))

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
    #backpropagation conditions are continue and differential. Sampling probs need in one distribution
    def interact(self, envstate, gamedata):
        torchstate      = torch.from_numpy(envstate).double().to(device)
        actor_actprob   = self.network_act(torchstate) #tau(a|s) = P(a,s) 8 elements corresponds to one action
        distribute      = torch.distributions.Categorical(actor_actprob) #category distribution
        action          = distribute.sample()
        actlogprob      = distribute.log_prob(action) #logeX

        gamedata.states.append(torchstate)
        gamedata.actions.append(action)
        gamedata.actoflogprobs.append(actlogprob) #the action number corresponds to the action_probs into log

        return action.detach().item() #return action_probs index corresponds to key 1,2,3,4

    #policy_ac.calculation will call
    def calculation(self, states, actions):
        critic_actprobs     = self.network_act(states) #each current with one action probility
        distribute          = torch.distributions.Categorical(critic_actprobs)
        critic_actlogprobs  = distribute.log_prob(actions) #logeX
        entropy             = distribute.entropy() # entropy is uncertain percentage, value higher mean uncertain more
        next_critic_values  = self.network_critic(states) #c_values is V(s) in A3C theroy

        #future using
        '''states_sampling = None
        sampler = BatchSampler(SubsetRandomSampler(range(states.size()[0])), states.size()[0], drop_last=False)
        for indices in sampler:
            states_sampling = states[indices]
        next_critic_values = self.network_critic(states_sampling)  # c_values is V(s) in A3C theroy
        next_critic_actprobs= critic_actprobs.gather(1, actions.unsqueeze(1).type(torch.int64))
        next_critic_actprobs= torch.squeeze(next_critic_actprobs)'''

        #if dimension can squeeze then tensor 3d to 2d.
        #EX: squeeze tensor[2,1,3] become to tensor[2,3]
        return critic_actlogprobs, torch.squeeze(next_critic_values), entropy
        #return critic_actlogprobs, entropy

class CPPO:
    def __init__(self, dim_states, dim_acts, h_neurons, lr, gamma, train_epochs, eps_clip, betas):
        self.lr             = lr
        self.betas          = betas
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.train_epochs   = train_epochs

        self.policy_ac      = Actor_Critic(dim_states, dim_acts, h_neurons).double().to(device)
        self.optimizer      = torch.optim.Adam(self.policy_ac.parameters(), lr=lr, betas=betas)

        #self.policy_curr    = Actor_Critic(dim_states, dim_acts, h_neurons).double().to(device)
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

        self.mseLoss        = nn.MSELoss(reduction='mean')

    def train_update(self, gamedata):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # R(τ) = gamma^n * τ(a|s)R(a,s) , n=1~k
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).double().to(device)

        '''rewards.mean() is E[R(τ)]
        rewards.std on torch is {1/(n-1) * Σ(x - x_average)} ** 0.5  (x ** 0.5 = x^0.5)
        1e-5 = 0.00001 which avoid rewards.std() is zero
        (Rewards - average_R) / (standard_R + 0.00001) is standard score
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)'''
        # should modify
        #curraccu_stdscore   = (rewards - rewards.mean()) / (rewards.std() + 1e-5) #Q(s,a)

        # convert list to tensor
        # torch.stack is combine many tensor 1D to 2D
        curr_states     = torch.stack(gamedata.states).double().to(device).detach()
        curr_actions    = torch.stack(gamedata.actions).double().to(device).detach()
        curr_logprobs   = torch.stack(gamedata.actoflogprobs).double().to(device).detach()

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
            #cstate_value is V(s) in A3C theroy. critic network weights as an actor feed state out reward value
            critic_actlogprobs, next_critic_values, entropy = self.policy_ac.calculation(curr_states, curr_actions)

            # https://socratic.org/questions/what-is-the-derivative-of-e-lnx
            # log(critic) - log(curraccu) = log(critic/curraccu)
            # ratios  = e^(ln(State2_actProbs)-ln(State1_actProbs)) =  e^ln(State2_actProbs/State1_actProbs)
            # ratios  = (State2_critic_actProbs/State1_actor_actProbs) = Pw(A1|S2)/Pw(A1|S1), where w is weights(theta)
            # ratios' = critic_actProb/actor_actProb' in derivative
            ratios  = torch.exp(critic_actlogprobs - curr_logprobs.detach())

            #advantages is stdscore mode
            surr1   = ratios * advantages
            surr2   = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # mseLoss is Mean Square Error = (target - output)^2
            loss    = -torch.min(surr1, surr2) + 0.5*self.mseLoss(rewards, next_critic_values) - 0.01*entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()  #get grade.data
            self.optimizer.step()   #update grade.data by adam method which is smooth grade

        # Copy new weights into old policy:
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "LunarLander-v2"
    # creating environment
    render          = False
    solved_reward   = 230           # stop training if reach avg_reward > solved_reward
    log_interval    = 20            # print avg reward in the interval
    h_neurons       = 64            # number of variables in hidden layer
    max_episodes    = 50000         # max training episodes
    max_timesteps   = 1500          # max timesteps in one episode
    update_timestep = 2000          # train_update policy every n timesteps
    train_epochs    = 4             # train_update policy for epochs
    lr              = 0.0005        # learning rate
    gamma           = 0.99          # discount factor
    eps_clip        = 0.2           # clip parameter for PPO2
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
        envstate = env.reset() #init state value to matrix
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_current:
            action = ppo.policy_ac.interact(envstate, gamedata)
            envstate, reward, done, _ = env.step(action)

            # one reward R(τ) = τ(a|s)R(a,s) in a certain state select an action and return the reward
            gamedata.rewards.append(reward)
            # Saving reward and is_terminal:
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
            torch.save(ppo.policy_ac.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        if i_episode % 500 == 0:
            torch.save(ppo.policy_ac.state_dict(), './PPO_{}_episode_{}.pth'.format(env_name, i_episode))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
