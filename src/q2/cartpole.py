'''
cartpole.py

Methods for solving Cartpole using Policy Gradient. Also includes a policy gradient model with a baseline. 
'''

import yaml
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)

### Configs
# Defining the Path to the config
CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
# Opening the Config
with open(CONFIG, "r") as f:
    config = yaml.safe_load(f)
general_config = config["question_2"]["general"]
cartpole_config = config["question_2"]["cartpole"]

### Consts
training_iterations = general_config["train_iterations"]

### Neural Network
# No Baseline
class CartpolePolicyNoBaseline(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class CartpolePolicyBaseline(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super().__init__()
        pass    

### Discounted Reward Function
def discounted_returns(rewards, discount_rate):
    '''
    Discounted reward as specified in the config. The rewards are provided by gymnasium
    '''
    discounted_reward = 0 # current reward at time t
    output = [] # All rewards. Note: This list starts with the reward at the very end, so this should be reverse if index 0 is to be beginning of time. 
    for r in reversed(rewards):
        discounted_reward = r + discount_rate * discounted_reward
        output.append(discounted_reward)
    return torch.tensor(output[::-1]) # by default float32

### Running a single trajectory/episode 
def episode(env, policy, seed=None):
    '''
    Run an episode until termination.
    '''
    obs, _ = env.reset(seed=seed)
    log_prob, rewards = [], []
    completed = False 
    
    while not completed:
        logits = policy(torch.as_tensor(obs, dtype=torch.float32))
        dist = Categorical(logits=logits)
        action = dist.sample() # returns indices
        obs, reward, terminated, truncated, _ = env.step(action.item()) # .item() returns an integer, the next action in the action space
        log_prob.append(dist.log_prob(action))
        rewards.append(reward)
        completed = terminated or truncated

    return torch.stack(log_prob), rewards 

### Training
BATCH = general_config["batch"] # episodes per update

def training(env, policy, optimizer, discount_rate, seed=None):
    '''
    Training policy Neural Network
    '''
    episodic_rewards = []
    n_updates = training_iterations // BATCH
    progress = tqdm(range(n_updates), desc="Training")
    for upd in progress:
        batch_loss = 0.0
        for b in range(BATCH):
            first = (upd == 0 and b == 0)
            log_probs, rewards = episode(env, policy, seed=seed if first else None)
            returns = discounted_returns(rewards, discount_rate)
            batch_loss = batch_loss + -(log_probs * returns).sum()
            episodic_rewards.append(sum(rewards))

        loss = batch_loss / BATCH
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (upd + 1) % 5 == 0:
            avg = np.mean(episodic_rewards[-100:])
            progress.set_postfix(reward=f"{episodic_rewards[-1]:.0f}", avg100=f"{avg:.1f}")
    return episodic_rewards

### Plots
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def plot_training(rewards, name="cartpole_no_baseline"):
    ma = np.convolve(rewards, np.ones(100) / 100, mode="valid")
    plt.figure()
    plt.plot(rewards, alpha=0.4, label="Episode reward")
    plt.plot(np.arange(99, len(rewards)), ma, label="Moving avg (100)")
    plt.xlabel("Training episode")
    plt.ylabel("Episode reward")
    plt.title("CartPole-v1 REINFORCE (no baseline)")
    plt.legend()
    plt.savefig(FIG_DIR / f"{name}_training.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_eval(rewards, name="cartpole_no_baseline"):
    plt.figure()
    plt.hist(rewards, bins=30)
    plt.xlabel("Episode reward")
    plt.ylabel("Count")
    plt.title(f"500 rollouts: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
    plt.savefig(FIG_DIR / f"{name}_eval_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

### Main
if __name__=="__main__":
    policy = CartpolePolicyNoBaseline()
    discount_rate = cartpole_config["discount_factor"]
    learning_rate = general_config["learning_rate"]
    seed = general_config["seed"]
    env = gym.make("CartPole-v1")
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    torch.manual_seed(seed)
    np.random.seed(seed)

    episodic_rewards = training(env, policy, optimizer, discount_rate, seed)

    # after training
    Path("models").mkdir(exist_ok=True)
    torch.save(policy.state_dict(), "models/cartpole_policy_no_baseline.pt")

    # Plot training
    plot_training(episodic_rewards)

    # Eval
    eval_rewards = []
    tqdm(range(n_updates), desc="Training")
    with torch.no_grad():
        for _ in tqdm(range(500), desc="Evaluating"):
            _, rewards = episode(env, policy)
            eval_rewards.append(sum(rewards))
    print(f"Eval: mean={np.mean(eval_rewards):.2f}, std={np.std(eval_rewards):.2f}")
    
    # Plot Eval
    plot_eval(eval_rewards)

        



    