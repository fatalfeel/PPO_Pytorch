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
    #Categorical distribution follow actor_actprob which sum is 1.0 then sample out your action also do entropy
    def interact(self, envstate, gamedata):
        torchstate      = torch.from_numpy(envstate).double().to(device)
        actor_actprob   = self.network_act(torchstate) #tau(a|s) = P(a,s) 8 elements corresponds to one action
        distribute      = torch.distributions.Categorical(actor_actprob)
        action          = distribute.sample()
        actlogprob      = distribute.log_prob(action) #logeX

        gamedata.states.append(torchstate)
        gamedata.actions.append(action)
        gamedata.actorlogprobs.append(actlogprob) #the action number corresponds to the action_probs into log

        return action.detach().item() #return action_probs index corresponds to key 1,2,3,4

    #policy_ac.calculation will call
    #sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
    #usually mini_batch_size sample < states'size do forward
    #in our example the mini_batch_size = states'size
    def calculation(self, states, actions):
        critic_actprobs     = self.network_act(states) #each current with one action probility
        distribute          = torch.distributions.Categorical(critic_actprobs)
        critic_actlogprobs  = distribute.log_prob(actions) #logeX
        entropy             = distribute.entropy() # entropy is uncertain percentage, value higher mean uncertain more
        next_critic_values  = self.network_critic(states) #c_values is V(s) in A3C theroy

        #if dimension can squeeze then tensor 3d to 2d.
        #EX: squeeze tensor[2,1,3] become to tensor[2,3]
        return critic_actlogprobs, torch.squeeze(next_critic_values), entropy

    # if is_terminals is false use Markov formula to replace last reward
    def predict_reward(self, next_state, gamedata, gamma):
        '''torchstate      = torch.from_numpy(next_state).double().to(device)
        actor_actprob   = self.network_act(torchstate)  # tau(a|s) = P(a,s) 8 elements corresponds to one action
        distribute      = torch.distributions.Categorical(actor_actprob)
        action          = distribute.sample()
        actlogprob      = distribute.log_prob(action)  # =lnX
        next_value      = self.network_critic(torchstate)
        data_value      = next_value.detach().cpu().data.numpy()[0]

        gamedata.states.append(torchstate)
        gamedata.actions.append(action)  # next action
        gamedata.actorlogprobs.append(actlogprob)  # the action number corresponds to the action_probs into log
        gamedata.rewards.append(data_value)
        #gamedata.is_terminals[-1] = True   # make self.gamma * discounted_reward = 0 to keep last reward
        gamedata.is_terminals.append(True)'''

        if gamedata.is_terminals[-1] is False:
            torchstate = torch.from_numpy(next_state).double().to(device)
            next_value = self.network_critic(torchstate)
            data_value = next_value.detach().cpu().data.numpy()[0]
            gamedata.rewards[-1] = gamedata.rewards[-1] + gamma * data_value

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

        #self.policy_curr   = Actor_Critic(dim_states, dim_acts, h_neurons).double().to(device)
        #self.policy_curr.load_state_dict(self.policy_ac.state_dict())

        self.MseLoss        = nn.MSELoss(reduction='none')

    def train_update(self, gamedata):
        returns             = []
        discounted_reward   = 0
        # Monte Carlo estimate of state rewards:
        for reward, is_terminal in zip(reversed(gamedata.rewards), reversed(gamedata.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # R(τ) = gamma^n * τ(a|s)R(a,s) , n=1~k
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

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
        curr_states         = torch.stack(gamedata.states).double().to(device).detach()
        curr_actions        = torch.stack(gamedata.actions).double().to(device).detach()
        curr_actlogprobs    = torch.stack(gamedata.actorlogprobs).double().to(device).detach()

        # critic_state_reward   = network_critic(curraccu_states)
        '''refer to a2c-ppo should modify like this
           advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
           advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)'''
        critic_vpi  = self.policy_ac.network_critic(curr_states)
        critic_vpi  = torch.squeeze(critic_vpi)
        qsa_sub_vs  = returns - critic_vpi.detach()  # A(s,a) => Q(s,a) - V(s), V(s) is critic
        advantages  = (qsa_sub_vs - qsa_sub_vs.mean()) / (qsa_sub_vs.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.train_epochs):
            #cstate_value is V(s) in A3C theroy. critic network weights as an actor feed state out reward value
            critic_actlogprobs, next_critic_values, entropy = self.policy_ac.calculation(curr_states, curr_actions)

            # https://socratic.org/questions/what-is-the-derivative-of-e-lnx
            # log(critic) - log(curraccu) = log(critic/curraccu)
            # ratios  = e^(ln(State2_actProbs)-ln(State1_actProbs)) =  e^ln(State2_actProbs/State1_actProbs)
            # ratios  = (State2_critic_actProbs/State1_actor_actProbs)
            # ratios  = next_critic_actprobs/curr_actions_prob = Pw(A1|S2)/Pw(A1|S1), where w is weights(theta)
            ratios  = torch.exp(critic_actlogprobs - curr_actlogprobs.detach())

            #advantages is stdscore mode
            surr1       = ratios * advantages
            surr2       = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            #we get all samples of critic_values, so value_preds_batch equal critic_values
            ''' value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_losses = (values - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,value_losses_clipped).mean() '''
            # value_predict_clip is critical predict value + (critical predict value - originla critical value)|range between(-param~+param)
            # value_predict_loss is (value_predict_clip - MDP-reward)^2
            # value_critic_loss is (critical predict value - MDP-reward)^2
            # value_loss is 0.5 x select max items in (predict_loss or value_critic_loss) ex: A=[2,6] b=[4,5] torch max=>[4,6]
            value_predict_clip  = critic_vpi.detach() + (next_critic_values - critic_vpi.detach()).clamp(-self.eps_clip, self.eps_clip)
            value_predict_loss  = self.MseLoss(value_predict_clip, returns)
            value_critic_loss   = self.MseLoss(next_critic_values, returns)
            value_loss          = 0.5 * torch.max(value_predict_loss, value_critic_loss)

            # MseLoss is Mean Square Error = (target - output)^2, next_critic_values in first param follow libtorch rules
            #loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(next_critic_values, returns) - 0.01*entropy
            loss = -torch.min(surr1, surr2) + self.vloss_coef * value_loss - self.entropy_coef * entropy

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
    betas           = (0.9, 0.999)  # Adam β
    gamma           = 0.99          # discount factor
    eps_clip        = 0.2           # clip parameter for PPO2
    vloss_coef      = 0.5           # clip parameter for PPO2
    entropy_coef    = 0.01
    predict_trick   = True          # trick shot make PPO get better action & reward
    #############################################

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]  # LunarLander give 8 states
    dim_acts    = 4  # 4 action directions

    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, h_neurons, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef)

    # logging variables
    running_reward  = 0
    avg_length      = 0
    timestep        = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        envstate = env.reset() #Done-0 State-0
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_current: #Done-0 State-0 Act-0
            action = ppo.policy_ac.interact(envstate, gamedata)

            # Done-1 State-1 Act-0 R-0
            envstate, reward, done, _ = env.step(action)

            # one reward R(τ) = τ(a|s)R(a,s) in a certain state select an action and return the reward
            gamedata.rewards.append(reward)

            # is_terminal in next state:
            gamedata.is_terminals.append(done)

            # train_update if its time
            if timestep % update_timestep == 0:
                if predict_trick is True:
                    ppo.policy_ac.predict_reward(envstate, gamedata, gamma)

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
