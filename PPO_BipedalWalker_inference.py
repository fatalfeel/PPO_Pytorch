import argparse
import torch
import gym

from PPO_BipedalWalker_train import GameContent, CPPO
from PIL import Image

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str,       default='./checkpoint', help='path to checkpoint')
parser.add_argument('--cuda',           type=str2bool,  default=False)
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name        = "BipedalWalker-v3"
    render          = True          # render the environment
    save_gif        = False         # png images are saved in gif folder
    h_neurons       = 1024          # number of variables in hidden layer
    n_episodes      = 200000        # num of episodes to run
    max_timesteps   = 1300          # max timesteps in one episode
    train_epochs    = 4             # train_update policy for K epochs
    lr              = 0.0001        # parameters for learning rate
    betas           = (0.9, 0.999)  # Adam β
    gamma           = 0.99          # discount factor
    eps_clip        = 0.2           # clip parameter for CPPO
    vloss_coef      = 0.5           # clip parameter for PPO2
    entropy_coef    = 0.01
    action_std      = 0.5           # constant std for action distribution (Multivariate Normal)
    #############################################

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]
    dim_acts    = env.action_space.shape[0]
    
    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, action_std, h_neurons, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef)
    ppo.policy_ac.eval()

    # map_location=torch.device('cpu') for cpu only if you have cuda then cancel it
    lastname    = args.checkpoint_dir + '/PPO_{}_last.pth'.format(env_name)
    checkpoint  = torch.load(lastname)
    ppo.policy_ac.load_state_dict(checkpoint['state_dict'])

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        envstates = env.reset()
        for ts in range(max_timesteps):
            #action = ppo.select_action(envstates, gamedata)
            action = ppo.policy_ac.interact(envstates, gamedata)
            envstates, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(ts))
            if done:
                break
            
        print('Episode: {} \t Reward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()
