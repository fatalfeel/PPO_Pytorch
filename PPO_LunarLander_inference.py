import argparse
import torch
import gym
from PPO_LunarLander_train import GameContent, CPPO
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
    env_name        = "LunarLander-v2"
    render          = True
    save_gif        = False
    h_neurons       = 1024          # number of variables in hidden layer
    n_episodes      = 200000        # num of episodes to run
    max_timesteps   = 400           # max timesteps in one episode
    train_epochs    = 4             # update policy for K epochs
    lr              = 0.0005        # parameters for learning rate
    betas           = (0.9, 0.999)  # Adam β
    gamma           = 0.99          # discount factor
    eps_clip        = 0.2           # clip parameter for PPO
    vloss_coef      = 0.5           # clip parameter for PPO2
    entropy_coef    = 0.01
    #############################################

    # creating environment
    env         = gym.make(env_name)
    dim_states  = env.observation_space.shape[0]
    dim_acts    = 4

    gamedata    = GameContent()
    ppo         = CPPO(dim_states, dim_acts, h_neurons, lr, betas, gamma, train_epochs, eps_clip, vloss_coef, entropy_coef)
    ppo.policy_ac.eval()

    # map_location=torch.device('cpu') for cpu only if you have cuda then cancel it
    lastname    = args.checkpoint_dir + '/PPO_{}_last.pth'.format(env_name)
    checkpoint  = torch.load(lastname)
    ppo.policy_ac.load_state_dict(checkpoint['state_dict'])

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        estates = env.reset()
        for ts in range(max_timesteps):
            action                      = ppo.policy_ac.interact(estates, gamedata)
            estates, reward, done, _    = env.step(action)
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

    
