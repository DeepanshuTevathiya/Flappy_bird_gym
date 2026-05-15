import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import argparse
import os

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_treshold = params["reward_treshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self,is_training = True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
    
        num_states = env.observation_space.shape[0] #input dim (12)
        num_actions = env.action_space.n            # output dim(2)
        policy_dqn = DQN(num_states, num_actions).to(device)
    
    
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())  #loading weigh and bias of policy dqn

            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters())

            best_reward = float("-inf")
        else:
            # Loading the best model for testing
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

    
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float, device=device)

            ep_reward = 0
            terminated = False
    
            while (not terminated and ep_reward < self.reward_treshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype = torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, _, _ = env.step(action.item())

                ep_reward += reward

                reward = torch.tensor(reward, dtype = torch.float, device=device)
                next_state = torch.tensor(next_state, dtype = torch.float, device=device)
    
                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1
    
                state = next_state
                
            print(f"Episode: {episode+1}  |  Rewards:{ep_reward}")

            # epsilon decay
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    log_msg = f"Best reward: {best_reward}  |  Episode: {episode+1}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

            if is_training and len(memory) > self.mini_batch_size:
                #get sample
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())  #syncing the networks
                    steps = 0
            
        # env.close() => want to stop manually
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor (terminations).float().to (device)

        # calculate target Q-values if terminations=true => zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1) [0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)