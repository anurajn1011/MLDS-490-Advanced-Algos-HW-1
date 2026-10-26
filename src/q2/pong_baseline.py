'''
pong_baseline.py

Pong solved using Advantage Actor-Critic baseline.
'''

import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pong import cfg, discounted_returns, rollout, evaluate, BATCH, CHUNK, LOG_DIR, MODEL_DIR, SAVE_EVERY, training_iterations, train_device

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

### Model
class PongPolicyBaseline(nn.Module):
    '''
    CNN trunk shared by the actor head (2 logits: RIGHT, LEFT) and the critic head (1 value).
    '''
    def __init__(self, hidden_dim=256, output_dim=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(32 * 8 * 8, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x.unsqueeze(-3))
        return self.actor(features), self.critic(features).squeeze(-1)


class ActorOnly(nn.Module):
    '''
    View of an actor critic that returns only the logits, so rollout() and evaluate() from pong.py work unchanged. It shares its parameters with the wrapped model.
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]

### Training
def training(vec_env, policy, optimizer, discount_rate, episodic_rewards, seed=None, value_coef=0.5, entropy_coef=0.0):
    '''
    Train the actor critic. Each update plays one game in every sub-environment, uses the advantage Q - V(s_t) with V detached, and sums the losses over steps and averages over games, as in pong.py.
    Returns the per-episode undiscounted rewards, the best weights seen by 100-episode average, and that best average.
    '''
    actor_view = ActorOnly(policy)
    n_envs = vec_env.num_envs
    best_avg = -float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    n_updates = training_iterations // BATCH
    progress = tqdm(range(n_updates), desc="Training")

    for upd in progress:
        states, actions, rewards, mask = rollout(vec_env, actor_view, train_device, seed=seed if upd == 0 else None)
        returns = discounted_returns(rewards, discount_rate)

        flat_states = torch.as_tensor(states[mask])
        flat_actions = torch.as_tensor(actions[mask])
        flat_returns = torch.as_tensor(returns[mask])

        optimizer.zero_grad()
        critic_total = 0.0
        for i in range(0, flat_states.shape[0], CHUNK):
            s = flat_states[i:i + CHUNK].to(train_device).float()
            a = flat_actions[i:i + CHUNK].to(train_device)
            g = flat_returns[i:i + CHUNK].to(train_device)
            logits, values = policy(s)
            dist = Categorical(logits=logits)
            advantages = g - values.detach()
            actor_loss = -(dist.log_prob(a) * advantages).sum() / n_envs
            critic_loss = 0.5 * (values - g).pow(2).sum() / n_envs
            loss = actor_loss + value_coef * critic_loss - entropy_coef * dist.entropy().sum() / n_envs
            loss.backward()
            critic_total += critic_loss.item()
        optimizer.step()

        episodic_rewards.extend(rewards.sum(axis=0).tolist())

        if len(episodic_rewards) >= 100:
            avg = np.mean(episodic_rewards[-100:])
            if avg > best_avg:
                best_avg = avg
                best_state = copy.deepcopy(policy.state_dict())

        if (upd + 1) % SAVE_EVERY == 0:
            np.save(LOG_DIR / "pong_baseline_rewards.npy", np.array(episodic_rewards))
            torch.save(best_state, MODEL_DIR / "pong_policy_baseline_best.pt")

        if (upd + 1) % 5 == 0:
            progress.set_postfix(reward=f"{episodic_rewards[-1]:.0f}", avg100=f"{np.mean(episodic_rewards[-100:]):.1f}", best=f"{best_avg:.1f}", critic=f"{critic_total:.2f}")
    return episodic_rewards, best_state, best_avg

### Plots
def plot_training(rewards, name="pong_baseline"):
    plt.figure()
    plt.plot(rewards, alpha=0.4, label="Episode reward")
    if len(rewards) >= 100:
        ma = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        plt.plot(np.arange(99, len(rewards)), ma, label="Moving avg (100)")
    plt.xlabel("Training episode")
    plt.ylabel("Episode reward")
    plt.title("Pong-v5 REINFORCE with baseline")
    plt.legend()
    plt.savefig(FIG_DIR / f"{name}_training.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_eval(rewards, name="pong_baseline"):
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

    vec_env = gym.make_vec("ALE/Pong-v5", num_envs=BATCH, vectorization_mode="async")

    print("Training on:", train_device)
    policy = PongPolicyBaseline().to(train_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    episodic_rewards = []
    try:
        episodic_rewards, best_state, best_avg = training(vec_env, policy, optimizer, discount_rate, episodic_rewards, seed)
    finally:
        np.save(LOG_DIR / "pong_baseline_rewards.npy", np.array(episodic_rewards))
        torch.save(policy.state_dict(), MODEL_DIR / "pong_policy_baseline_final.pt")
        vec_env.close()

    torch.save(best_state, MODEL_DIR / "pong_policy_baseline_best.pt")
    print(f"Best avg100 during training: {best_avg:.2f}")

    plot_training(episodic_rewards)

    eval_env = gym.make_vec("ALE/Pong-v5", num_envs= BATCH, vectorization_mode="async")
    actor_view = ActorOnly(policy)

    final_rewards = evaluate(eval_env, actor_view)
    print(f"Final policy: mean={np.mean(final_rewards):.2f}, std={np.std(final_rewards):.2f}")
    plot_eval(final_rewards, name="pong_baseline_final")

    policy.load_state_dict(best_state)
    best_rewards = evaluate(eval_env, actor_view)
    print(f"Best policy:  mean={np.mean(best_rewards):.2f}, std={np.std(best_rewards):.2f}")
    plot_eval(best_rewards, name="pong_baseline_best")

    eval_env.close()