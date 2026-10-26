'''
pong.py

Policy gradient (REINFORCE, no baseline) for Pong-v5 using only the RIGHT and LEFT actions.
Games are played in parallel with a vectorized environment: one update uses BATCH games, one per sub-environment, and each forward pass handles all of them at once.
'''

import copy
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
from torch.distributions import Categorical
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)
gym.register_envs(ale_py)

### Configs
CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG, "r") as f:
    config = yaml.safe_load(f)
general_config = config["question_2"]["general"]
pong_config = config["question_2"]["pong"]

def cfg(key):
    '''Look up a setting in the pong section first, then fall back to the general section.'''
    return pong_config.get(key, general_config.get(key))

### Consts
training_iterations = cfg("train_iterations")  # total training episodes
BATCH = cfg("batch") # episodes per update is the number of parallel environments
RIGHT, LEFT = 2, 3 # From documentation
ACTION_MAP = np.array([RIGHT, LEFT]) # network output index is the Atari action id
CHUNK = 4096 # steps per backward pass to limit GPU memory

LOG_DIR = Path("logs")
MODEL_DIR = Path("models")
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SAVE_EVERY = 20 # save the rewards and best weights every this many updates

train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_device = train_device

### Preprocessing from HW sheet
def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float32).ravel(), [80,80])

### Neural Network
class PongPolicyNoBaseline(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),   # (1,80,80)  to (16,19,19)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (16,19,19) to (32,8,8)
            nn.ReLU(),
            nn.Flatten(start_dim=-3, end_dim=-1),        # (32,8,8) to 2048 (also works with a batch dim)
            nn.Linear(32 * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x is (80,80) for a single frame or (B,80,80) for a batch; add the channel dimension
        return self.net(x.unsqueeze(-3))

### Discounted Reward Function
def discounted_returns(rewards, discount_rate):
    '''
    Discounted return-to-go for every environment at once.
    rewards: numpy array of shape (T, n), already zeroed after each game ended.
    In Pong every nonzero reward ends a point, so the running sum is reset there; later
    points should not affect the credit given to actions in earlier points.
    '''
    T, n = rewards.shape
    returns = np.zeros((T, n), dtype=np.float32)
    running = np.zeros(n, dtype=np.float32)
    for t in reversed(range(T)):
        running = rewards[t] + discount_rate * np.where(rewards[t] != 0, 0.0, running)
        returns[t] = running
    return returns

### Playing one game in every environment
# def rollout(vec_env, policy, device, seed=None, store_states=True):
#     '''
#     Play one full game in each of the n sub-environments at the same time, without tracking gradients. The input to the network is the difference between consecutive preprocessed frames, so the network can see motion.

#     Returns numpy arrays, all with time as the first axis (T = length of the longest game):
#         states  (T, n, 80, 80) int8   frame differences in {-1, 0, 1} (None if store_states=False)
#         actions (T, n) sampled action indices (0 = RIGHT, 1 = LEFT)
#         rewards (T, n)  0 once an environment's game is over
#         mask (T, n) bool True while an environment's game is still being played

#     An environment that finishes early keeps being stepped until all are done, but its data is masked out and never used.
#     '''
#     n = vec_env.num_envs
#     obs, _ = vec_env.reset(seed=seed)
#     alive = np.ones(n, dtype=bool)
#     prev = None
#     state_list, action_list, reward_list, mask_list = [], [], [], []

#     with torch.no_grad():
#         while alive.any():
#             cur = np.stack([preprocess(o) for o in obs])              # (n,80,80)
#             x = cur - prev if prev is not None else np.zeros_like(cur)
#             prev = cur

#             logits = policy(torch.as_tensor(x, dtype=torch.float32, device=device))
#             action = Categorical(logits=logits).sample().cpu().numpy()  # (n,)
#             obs, reward, terminated, truncated, _ = vec_env.step(ACTION_MAP[action])

#             if store_states:
#                 state_list.append(x.astype(np.int8))
#             action_list.append(action)
#             reward_list.append(np.where(alive, reward, 0.0))
#             mask_list.append(alive.copy())

#             alive = alive & ~(terminated | truncated)

#     states = np.stack(state_list) if store_states else None
#     return (states, np.stack(action_list),
#             np.stack(reward_list).astype(np.float32), np.stack(mask_list))


def rollout(vec_env, policy, device, seed=None, store_states=True):
    n = vec_env.num_envs
    obs, _ = vec_env.reset(seed=seed)

    alive = np.ones(n, dtype=bool)
    prev = None
    state_list, action_list, reward_list, mask_list = [], [], [], []

    with torch.no_grad():
        while alive.any():
            cur = np.stack([preprocess(o) for o in obs])

            # First state is zero; later states are frame differences.
            x = cur - prev if prev is not None else np.zeros_like(cur)
            prev = cur

            logits = policy(
                torch.as_tensor(x, dtype=torch.float32, device=device)
            )
            action = Categorical(logits=logits).sample().cpu().numpy()

            obs, reward, terminated, truncated, _ = vec_env.step(
                ACTION_MAP[action]
            )

            if store_states:
                state_list.append(x.astype(np.int8))

            action_list.append(action)
            reward_list.append(np.where(alive, reward, 0.0))
            mask_list.append(alive.copy())

            alive = alive & ~(terminated | truncated)

    states = np.stack(state_list) if store_states else None

    return (
        states,
        np.stack(action_list),
        np.stack(reward_list).astype(np.float32),
        np.stack(mask_list),
    )

### Training
def training(vec_env, policy, optimizer, discount_rate, episodic_rewards, seed=None):
    '''
    Train the policy. Each update plays one game in every sub-environment (BATCH games), averages the policy-gradient loss over those games and takes one optimizer step.
    Returns the per-episode undiscounted rewards, the best weights seen by 100-episode average, and that best average.
    '''
    n_envs = vec_env.num_envs
    # episodic_rewards is created by the caller and filled in place, so it survives a crash
    best_avg = -float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    n_updates = training_iterations // BATCH
    progress = tqdm(range(n_updates), desc="Training")

    for upd in progress:
        states, actions, rewards, mask = rollout(
            vec_env, policy, train_device, seed=seed if upd == 0 else None)
        returns = discounted_returns(rewards, discount_rate)

        # Keep only steps from games that were still running; shapes become (K, ...)
        flat_states = torch.as_tensor(states[mask])    # int8
        flat_actions = torch.as_tensor(actions[mask])
        flat_returns = torch.as_tensor(returns[mask])

        # One batched forward pass WITH gradients, in chunks to limit GPU memory.
        # The loss is a sum over steps, so backward() on each chunk accumulates the same
        # gradient as one big pass. Dividing by n_envs averages over the games.
        optimizer.zero_grad()
        for i in range(0, flat_states.shape[0], CHUNK):
            s = flat_states[i:i + CHUNK].to(train_device).float()
            a = flat_actions[i:i + CHUNK].to(train_device)
            g = flat_returns[i:i + CHUNK].to(train_device)
            log_probs = Categorical(logits=policy(s)).log_prob(a)
            loss = -(log_probs * g).sum() / n_envs
            loss.backward()
        optimizer.step()

        episodic_rewards.extend(rewards.sum(axis=0).tolist())  # undiscounted reward per game

        if len(episodic_rewards) >= 100:
            avg = np.mean(episodic_rewards[-100:])
            if avg > best_avg:
                best_avg = avg
                best_state = copy.deepcopy(policy.state_dict())

        # Periodic save so a hard failure (killed job, node restart) loses little
        if (upd + 1) % SAVE_EVERY == 0:
            np.save(LOG_DIR / "pong_rewards.npy", np.array(episodic_rewards))
            torch.save(best_state, MODEL_DIR / "pong_policy_no_baseline_best.pt")

        if (upd + 1) % 5 == 0:
            progress.set_postfix(reward=f"{episodic_rewards[-1]:.0f}",
                                 avg100=f"{np.mean(episodic_rewards[-100:]):.1f}",
                                 best=f"{best_avg:.1f}")
    return episodic_rewards, best_state, best_avg

### Evaluation
def evaluate(eval_env, policy, n_episodes=500):
    '''Play at least n_episodes games (in parallel) and return exactly n_episodes total rewards.'''
    policy = policy.to(eval_device)
    policy.eval()
    totals = []
    progress = tqdm(total=n_episodes, desc="Evaluating")
    while len(totals) < n_episodes:
        _, _, rewards, _ = rollout(eval_env, policy, eval_device, store_states=False)
        totals.extend(rewards.sum(axis=0).tolist())
        progress.update(eval_env.num_envs)
    progress.close()
    return totals[:n_episodes]

### Plots
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def plot_training(rewards, name="pong_no_baseline"):
    plt.figure()
    plt.plot(rewards, alpha=0.4, label="Episode reward")
    if len(rewards) >= 100:
        ma = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        plt.plot(np.arange(99, len(rewards)), ma, label="Moving avg (100)")
    plt.xlabel("Training episode")
    plt.ylabel("Episode reward")
    plt.title("Pong-v5 REINFORCE (no baseline)")
    plt.legend()
    plt.savefig(FIG_DIR / f"{name}_training.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_eval(rewards, name="pong_no_baseline"):
    plt.figure()
    plt.hist(rewards, bins=30)
    plt.xlabel("Episode reward")
    plt.ylabel("Count")
    plt.title(f"{len(rewards)} rollouts: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
    plt.savefig(FIG_DIR / f"{name}_eval_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

### Main
if __name__ == "__main__":
    seed = cfg("seed")
    torch.manual_seed(seed)
    np.random.seed(seed)

    discount_rate = cfg("discount_factor")
    learning_rate = cfg("learning_rate")

    # Create the worker processes BEFORE anything touches the GPU
    vec_env = gym.make_vec("ALE/Pong-v5", num_envs=BATCH, vectorization_mode="async")

    print("Training on:", train_device)
    policy = PongPolicyNoBaseline().to(train_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)  # created after .to(device)

    episodic_rewards = []  # lives outside training() so the rewards survive a crash
    try:
        episodic_rewards, best_state, best_avg = training(vec_env, policy, optimizer, discount_rate, episodic_rewards, seed)
    finally:
        # Runs even if training raises an error or is stopped with Ctrl+c
        np.save(LOG_DIR / "pong_rewards.npy", np.array(episodic_rewards))
        torch.save(policy.state_dict(), MODEL_DIR / "pong_policy_no_baseline_final.pt")
        vec_env.close()

    torch.save(best_state, MODEL_DIR / "pong_policy_no_baseline_best.pt")
    print(f"Best avg100 during training: {best_avg:.2f}")

    plot_training(episodic_rewards)

    # Evaluate the final policy, then the best policy, with their own parallel environments
    eval_env = gym.make_vec("ALE/Pong-v5", num_envs=cfg("eval_envs") or BATCH, vectorization_mode="async")

    final_rewards = evaluate(eval_env, policy)
    print(f"Final policy: mean={np.mean(final_rewards):.2f}, std={np.std(final_rewards):.2f}")
    plot_eval(final_rewards, name="pong_no_baseline_final")

    policy.load_state_dict(best_state)
    best_rewards = evaluate(eval_env, policy)
    print(f"Best policy:  mean={np.mean(best_rewards):.2f}, std={np.std(best_rewards):.2f}")
    plot_eval(best_rewards, name="pong_no_baseline_best")

    eval_env.close()