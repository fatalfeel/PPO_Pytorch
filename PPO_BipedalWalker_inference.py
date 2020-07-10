from PPO_BipedalWalker_train import GameContent, CPPO
from PIL import Image
import torch
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "BipedalWalker-v3"
    env             = gym.make(env_name)
    dim_states      = env.observation_space.shape[0]
    dim_acts        = env.action_space.shape[0]
    
    n_episodes      = 3          # num of episodes to run
    max_timesteps   = 1500    # max timesteps in one episode
    render          = True           # render the environment
    save_gif        = False        # png images are saved in gif folder

    action_std      = 0.5        # constant std for action distribution (Multivariate Normal)
    train_epochs    = 80           # update policy for K epochs
    eps_clip        = 0.2          # clip parameter for CPPO
    gamma           = 0.99            # discount factor
    lr              = 0.0003             # parameters for Adam optimizer
    betas           = (0.9, 0.999)
    #############################################
    
    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, action_std, lr, gamma, train_epochs, eps_clip, betas)

    directory = "./preTrained/"
    filename = "PPO_{}.pth".format(env_name)
    ppo.policy_old.load_state_dict(torch.load(directory+filename))
    # filename and directory to load model from

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        estates = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(estates, gamedata)
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
