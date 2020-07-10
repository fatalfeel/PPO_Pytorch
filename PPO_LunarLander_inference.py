import gym
from PPO_LunarLander_train import GameContent, CPPO
from PIL import Image
import torch

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env             = gym.make(env_name)
    dim_states      = env.observation_space.shape[0]
    dim_acts        = 4
    render          = False
    max_timesteps   = 500
    h_neurons       = 64           # number of variables in hidden layer
    lr              = 0.0007
    betas           = (0.9, 0.999)
    gamma           = 0.99                # discount factor
    train_epochs    = 4                # update policy for K epochs
    eps_clip        = 0.2              # clip parameter for PPO
    #############################################

    n_episodes      = 6
    max_timesteps   = 300
    render          = True
    save_gif        = False

    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, h_neurons, lr, gamma, train_epochs, eps_clip, betas)

    directory   = "./preTrained/"
    filename    = "PPO_{}.pth".format(env_name)
    ppo.policy_old.load_state_dict(torch.load(directory+filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        estates = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.interact(estates, gamedata)
            estates, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()

    
