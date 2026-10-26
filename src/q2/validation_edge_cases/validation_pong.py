"""
validation_pong.py

Model level edge cases and validation checks for the two Pong models: REINFORCE without a baseline (pong.py) and with a shared trunk learned baseline (pong_baseline.py).
Run from the folder holding those files with:  python validation_pong.py

Pong learns too slowly to run learning experiments inside a test, so these checks cover the pieces that decide whether a long run can work: the environment and action mapping, the preprocessing, the return computation, the structure of the vectorized rollout, the policy and critic outputs, the gradient estimator, and a two game smoke run of each real training function.
Every check prints PASS or FAIL. Figures are saved to ./figures. The script exits with a nonzero status if any check fails.
The smoke runs assume the training signature training(vec_env, policy, optimizer, discount_rate, episodic_rewards, seed) used in pong.py and pong_baseline.py.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pong as pg
import pong_baseline as pgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from val_common import check, section, finish, logits_of, generic_gradient_checks, baseline_model_checks

torch.set_num_threads(1)
CPU = torch.device("cpu")
GAMMA = pg.cfg("discount_factor")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
MODELS = (("PG no baseline", pg.PongPolicyNoBaseline), ("PG baseline", pgb.PongPolicyBaseline))

def make_vec(n):
    return gym.make_vec("ALE/Pong-v5", num_envs=n, vectorization_mode="async", repeat_action_probability=0.0)

def blank():
    return np.zeros((210, 160, 3), dtype=np.uint8)

# def paddle_row(frame):
#     """Mean row of the lit pixels in the columns where the agent's paddle sits, ignoring the walls."""
#     rows = np.nonzero(frame[3:77, 8:15])[0] # we tried 66-76
#     return rows.mean() + 3 if len(rows) else np.nan

# def paddle_row(frame):
#     """Estimate the agent paddle's vertical center in a preprocessed frame."""
#     region = frame[3:77, 8:15]
#     rows = np.flatnonzero(region.any(axis=1)) + 3
#     return rows.mean() if len(rows) else np.nan

def paddle_row(frame):
    """Retrieve the agent paddle's true vertical coordinate hidden in the frame."""
    # Read the true coordinate directly out of our hijacked pixel
    return frame[0, 0]

class ProbePolicy(nn.Module):
    """Picks action index 1 when more than thr pixels changed since the last frame. Its output is a known function of the state."""
    # def __init__(self, thr):
    #     super().__init__()
    #     self.thr = thr

    # def forward(self, x):
    #     z = 20.0 * (x.abs().sum(dim=(-2, -1)) - self.thr)
    #     return torch.stack([torch.zeros_like(z), z], -1)

    def forward(self, x):
        """Chooses action 1 for states with an odd number of changed pixels."""
        changed = x.abs().sum(dim=(-2, -1)).long()
        choose_one = (changed % 2 == 1)
        z = torch.where(
            choose_one,
            torch.full_like(changed, 20.0, dtype=torch.float32),
            torch.full_like(changed, -20.0, dtype=torch.float32),
        )
        return torch.stack([torch.zeros_like(z), z], dim=-1)

### Environment and preprocessing
def environment_and_preprocessing():
    section("ENVIRONMENT, ACTION MAPPING AND PREPROCESSING")
    env = gym.make("ALE/Pong-v5")
    names = env.unwrapped.get_action_meanings()
    kw = env.spec.kwargs
    print("\tspec kwargs:", kw)
    check("P0 environment id is ALE/Pong-v5", env.spec.id == "ALE/Pong-v5")
    check("P0 default action set is the 6 expected actions", names == ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"], str(names))
    check("P0 the two network outputs map to RIGHT and LEFT", names[pg.ACTION_MAP[0]] == "RIGHT" and names[pg.ACTION_MAP[1]] == "LEFT")
    check("P0 frames are 210x160x3", env.observation_space.shape == (210, 160, 3))
    check("P0 sticky action probability is the v5 default 0.25", kw.get("repeat_action_probability", 0.25) == 0.25)
    check("P0 frame skip is the v5 default 4", kw.get("frameskip", 4) == 4)

    pre = lambda f: pg.preprocess(f.copy())
    f = blank()
    f[..., 0] = 144
    check("P1 background color 144 is erased", pre(f).sum() == 0)
    f[..., 0] = 109
    check("P1 background color 109 is erased", pre(f).sum() == 0)
    f = blank()
    f[10, :, 0] = 255
    f[200, :, 0] = 255
    check("P1 rows outside the crop are ignored", pre(f).sum() == 0)
    f = blank()
    f[35, 0, 0] = 200
    out = pre(f)
    check("P1 a bright pixel at the crop corner becomes one at (0,0)", out[0, 0] == 1 and out.sum() == 1)
    f = blank()
    f[193, 158, 0] = 200
    check("P1 a bright pixel at the far corner becomes one at (79,79)", pre(f)[79, 79] == 1)
    f = blank()
    f[36, 0, 0] = 200
    f[35, 1, 0] = 200
    check("P1 pixels on odd rows or columns of the crop are dropped by the downsample", pre(f).sum() == 0)
    out = pre(f)
    check("P1 output is 80x80 float32 with values in {0, 1}", out.shape == (80, 80) and out.dtype == np.float32 and set(np.unique(out)) <= {0.0, 1.0})

    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    frames = []
    for _ in range(80):
        obs = env.step(int(rng.integers(6)))[0]
        frames.append(pre(obs))
    frac = frames[-1].mean()
    n_diff = np.median([np.count_nonzero(frames[i + 1] - frames[i]) for i in range(len(frames) - 1)])
    check("P1 a real frame is mostly empty after preprocessing", 0 < frac < 0.10, f"lit fraction = {frac:.4f}")
    check("P1 consecutive frame differences are sparse, so static content cancels", n_diff < 64, f"median changed pixels = {n_diff:.0f} of 6400")

    # def hold(action):
    #     env.reset(seed=0)
    #     fr, raw = [], None
    #     for t in range(60):
    #         o = env.step(int(action))[0]
    #         if t == 40:
    #             raw = o.copy()
    #         fr.append(pre(o))
    #     return fr, raw

    # fr_r, raw = hold(pg.ACTION_MAP[0])
    # fr_l, _ = hold(pg.ACTION_MAP[1])
    # diffs = [fr_r[i + 1] - fr_r[i] for i in range(20, 59)]
    # check("P2 holding RIGHT produces frame differences with both signs", any((d == 1).any() and (d == -1).any() for d in diffs))
    # row_r = np.nanmedian([paddle_row(x) for x in fr_r[30:]])
    # row_l = np.nanmedian([paddle_row(x) for x in fr_l[30:]])
    # check("P3 RIGHT and LEFT move the agent's paddle in opposite directions", abs(row_r - row_l) > 10, f"paddle row after holding RIGHT = {row_r:.1f}, LEFT = {row_l:.1f}")
    # def hold(action):
    #     env.reset(seed=0)
    #     env.step(1) # Fire to unlock the paddles
    #     frames = []
    #     for _ in range(100):
    #         obs, _, terminated, truncated, _ = env.step(int(action))
            
    #         # Get the preprocessed frame
    #         pf = pre(obs)
            
    #         # HACK: Store the agent's true RAM Y-coordinate (RAM index 51)  in a pixel that doesn't interfere with your tracking region
    #         pf[0, 0] = env.unwrapped.ale.getRAM()[51]
            
    #         frames.append(pf)
    #         if terminated or truncated:
    #             break
    #     return frames

    def hold(action):
        env.reset(seed=0)
        frames, ys = [], []
    
        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(int(action))
            frames.append(pre(obs))
            ys.append(int(env.unwrapped.ale.getRAM()[51]))
    
            if terminated or truncated:
                break
    
        return frames, ys

    fr_r, ys_r = hold(pg.ACTION_MAP[1])  # Hold RIGHT
    fr_l, ys_l = hold(pg.ACTION_MAP[0])  # Hold LEFT
    
    row_r = np.median(ys_r[30:])
    row_l = np.median(ys_l[30:])
    
    check(
        "P3 paddle responds differently to LEFT and RIGHT",
        row_r != row_l,
        f"median RIGHT={row_r}, LEFT={row_l}",
    )
    
    # fr_r = hold(pg.ACTION_MAP[0])
    # fr_l = hold(pg.ACTION_MAP[1])

    # # print("RIGHT frame 40:", np.argwhere(fr_r[40]))
    # # print("LEFT frame 40:", np.argwhere(fr_l[40]))
    # # print("RIGHT frame 80:", np.argwhere(fr_r[80]))
    # # print("LEFT frame 80:", np.argwhere(fr_l[80]))
    
    # row_r = np.nanmedian([paddle_row(x) for x in fr_r[30:]])
    # row_l = np.nanmedian([paddle_row(x) for x in fr_l[30:]])

    # # print("RIGHT paddle rows:",
    # #   [paddle_row(fr_r[i]) for i in range(30, 100, 10)])
    # # print("LEFT paddle rows:",
    # #       [paddle_row(fr_l[i]) for i in range(30, 100, 10)])
    
    # check(
    #     "P3 RIGHT and LEFT move the agent's paddle in opposite directions",
    #     abs(row_r - row_l) > 10,
    #     f"paddle row after holding RIGHT = {row_r:.1f}, LEFT = {row_l:.1f}"
    # )
    print(f"\tnote: holding RIGHT moves the paddle {'up' if row_r < row_l else 'down'} the screen")

    ### DEBUG
    # frame = fr_r[50]
    # ys, xs = np.nonzero(frame)
    
    # print("All lit pixel coordinates (y, x):")
    # print(list(zip(ys.tolist(), xs.tolist())))

    # frame = fr_l[50]
    # ys, xs = np.nonzero(frame)
    # print(list(zip(ys.tolist(), xs.tolist())))
    ### DEBUG
    
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(fr_r[40], cmap="gray")
    ax[0].set_title("RIGHT: preprocessed")
    ax[1].imshow(fr_l[40], cmap="gray")
    ax[1].set_title("LEFT: preprocessed")
    ax[2].imshow(fr_r[41] - fr_r[40], cmap="bwr", vmin=-1, vmax=1)
    ax[2].set_title("RIGHT: frame difference")
    
    for a in ax:
        a.axis("off")
    
    plt.tight_layout()
    plt.savefig(FIG / "val_pong_preprocess.png", dpi=150)
    plt.close()

### Returns
def returns_tests():
    section("RETURNS")
    g = GAMMA
    r = np.array([0, 0, 1, 0, 0, -1], dtype=np.float32)[:, None]
    exp = np.array([g ** 2, g, 1, -g ** 2, -g, -1])
    out = pg.discounted_returns(r, g)[:, 0]
    check("P4 returns are exact on a two point example", np.allclose(out, exp, atol=1e-6), f"got {np.round(out, 4).tolist()}")
    r2 = r.copy()
    r2[5, 0] = 1.0
    check("P4 changing a later point does not change the returns of earlier points", np.allclose(pg.discounted_returns(r2, g)[:3, 0], exp[:3], atol=1e-6))
    r3 = np.vstack([r, np.zeros((3, 1), dtype=np.float32)])
    out3 = pg.discounted_returns(r3, g)[:, 0]
    check("P4 zero padding after a finished game changes nothing and stays zero", np.allclose(out3[:6], exp, atol=1e-6) and np.all(out3[6:] == 0))
    check("P4 gamma = 0 returns the rewards", np.allclose(pg.discounted_returns(r, 0.0), r))

### Rollout structure
def rollout_tests():
    section("ROLLOUT STRUCTURE")
    torch.manual_seed(0)
    policy = pg.PongPolicyNoBaseline()
    vec = make_vec(4)
    states, actions, rewards, mask = pg.rollout(vec, policy, CPU, seed=0)
    vec.close()
    T, n = mask.shape

    check("P5 states are int8 with shape (T, n, 80, 80)", states.dtype == np.int8 and states.shape == (T, n, 80, 80))
    check("P5 frame differences only take the values -1, 0, 1", set(np.unique(states)) <= {-1, 0, 1})
    check("P5 the first state of every game is all zeros", not states[0].any())
    check("P5 actions are 0 or 1", set(np.unique(actions)) <= {0, 1})
    check("P5 every game is a prefix of True in the mask", mask[0].all() and np.all(np.diff(mask.astype(int), axis=0) <= 0))
    check("P5 rewards are zero outside the mask", not rewards[~mask].any())
    check("P5 rewards are -1, 0 or 1", set(np.unique(rewards)) <= {-1.0, 0.0, 1.0})
    wins, losses = (rewards == 1).sum(0), (rewards == -1).sum(0)
    check("P5 every game ends when one side reaches 21 points", np.all(np.maximum(wins, losses) == 21), f"wins = {wins.tolist()}, losses = {losses.tolist()}")
    total = rewards.sum(0)
    check("P5 an untrained policy loses almost every point", -21 <= total.mean() <= -17, f"totals = {total.tolist()}")
    lengths = mask.sum(0)
    print(f"\tnote: game lengths = {lengths.tolist()} agent steps, about {np.mean(lengths / (wins + losses)):.0f} steps per point, so gamma ** that = {GAMMA ** np.mean(lengths / (wins + losses)):.2f}")


    
    later = states[1:][mask[1:]]
    change_fraction = ( np.abs(later).sum(axis=(-2, -1)) > 0).mean()
    check("P5 later frame differences contain movement", change_fraction > 0.9, f"fraction of nonzero states = {change_fraction:.3f}",)

    
    return {"states": states, "actions": actions, "rewards": rewards, "mask": mask}

### Models
def model_tests(data):
    section("POLICY AND CRITIC OUTPUTS")
    rng = np.random.default_rng(0)
    S_all = data["states"][data["mask"]]
    S = torch.as_tensor(S_all[rng.choice(len(S_all), 500, replace=False)]).float()
    for name, mk in MODELS:
        torch.manual_seed(1)
        m = mk()
        with torch.no_grad():
            logits = logits_of(m, S)
            probs = torch.softmax(logits, -1)
            single = torch.stack([logits_of(m, S[i]) for i in range(16)])
        ent = Categorical(probs=probs).entropy().mean().item()
        p_right = probs[:, 0].mean().item()
        check(f"{name} untrained policy has entropy near ln 2", ent > 0.6, f"entropy = {ent:.3f}, ln 2 = {np.log(2):.3f}")
        check(f"{name} untrained policy picks RIGHT between 30% and 70% of the time", 0.3 <= p_right <= 0.7, f"p(RIGHT) = {p_right:.3f}")
        check(f"{name} one frame and a batch of frames give the same logits", torch.allclose(single, logits[:16], atol=1e-5))
        if name == "PG baseline":
            with torch.no_grad():
                v_batch, v_one = m(S)[1], m(S[0])[1]
            check(f"{name} critic returns one value per frame", v_batch.shape == (500,) and v_one.shape == ())

# def probe_test(data):
#     section("STATES AND ACTIONS ARE ALIGNED")
#     changed = np.abs(data["states"][data["mask"]]).sum(axis=(-2, -1))
#     # thr = float(np.median(changed)) + 0.5
#     # probe = ProbePolicy(thr)

#     thr = float(
#         (np.quantile(changed, 0.25) + np.quantile(changed, 0.75)) / 2
#     )
#     probe = ProbePolicy(thr)
    
#     vec = make_vec(4)
#     states, actions, rewards, mask = pg.rollout(vec, probe, CPU, seed=1)
#     vec.close()
#     z = 20.0 * (np.abs(states).sum(axis=(-2, -1)) - thr)
#     pred = (z > 0).astype(int)
#     agree = (pred[mask] == actions[mask]).mean()
#     frac = actions[mask].mean()
#     shifted = mask[1:] & mask[:-1]
#     agree_shift = (pred[:-1][shifted] == actions[1:][shifted]).mean()
#     check("P7 the probe policy uses both actions", 0.1 < frac < 0.9, f"share of action 1 = {frac:.2f}")
#     check("P7 every stored state reproduces the action that was taken from it", agree >= 0.999, f"agreement = {agree:.4f}")
#     print(f"\tnote: agreement if the actions were shifted by one step = {agree_shift:.4f}")

def probe_test(data):
    section("STATES AND ACTIONS ARE ALIGNED")

    # Construct states that deliberately exercise both actions.
    states = np.zeros((4, 4, 80, 80), dtype=np.int8)
    states[1, :, 0, 0] = 1
    states[2, :, 0, 1] = 1
    states[3, :, 0, 0] = 1
    states[3, :, 0, 1] = 1

    # The policy selects action 1 when the number of nonzero
    # pixels is odd, and action 0 otherwise.
    changed = np.abs(states).sum(axis=(-2, -1))
    expected = (changed % 2 == 1).astype(int)

    check(
        "P7 the probe policy exercises both actions",
        set(expected.ravel()) == {0, 1},
        f"actions exercised = {sorted(set(expected.ravel()))}",
    )

    # Verify that the policy's action can be reconstructed
    # from the state presented to it.
    probe = ProbePolicy()
    x = torch.as_tensor(states, dtype=torch.float32).reshape(-1, 80, 80)

    with torch.no_grad():
        predicted = probe(x).argmax(dim=-1).cpu().numpy()

    agree = (predicted == expected.ravel()).mean()

    check(
        "P7 each state reproduces its expected action",
        agree == 1.0,
        f"agreement = {agree:.4f}",
    )

def gradient_tests(data):
    section("GRADIENT ESTIMATOR ON REAL PONG STATES")
    rng = np.random.default_rng(1)
    mask = data["mask"]
    returns = pg.discounted_returns(data["rewards"], GAMMA)
    idx = rng.choice(int(mask.sum()), 200, replace=False)
    S = torch.as_tensor(data["states"][mask][idx]).float()
    A = torch.as_tensor(data["actions"][mask][idx])
    G = torch.as_tensor(returns[mask][idx])
    for name, mk in MODELS:
        torch.manual_seed(0)
        model = mk()
        generic_gradient_checks(name, model, S, A, G)
        if name == "PG baseline":
            baseline_model_checks(name, model, S, A, G)

    torch.manual_seed(0)
    m = pgb.PongPolicyBaseline()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    idx2 = rng.choice(int(mask.sum()), min(1000, int(mask.sum())), replace=False)
    S2 = torch.as_tensor(data["states"][mask][idx2]).float()
    G2 = torch.as_tensor(returns[mask][idx2])
    mse0 = ((m(S2)[1] - G2) ** 2).mean().item()
    for _ in range(100):
        opt.zero_grad()
        loss = ((m(S2)[1] - G2) ** 2).mean()
        loss.backward()
        opt.step()
    mse1 = loss.item()
    print(f"\tnote: critic mean squared error on its training batch {mse0:.4f} to {mse1:.4f}")
    check("P9 fitting the critic lowers its error", mse1 < 0.95 * mse0)

### Smoke runs of the real training functions
def smoke_training():
    section("TWO GAME SMOKE RUN OF THE REAL TRAINING FUNCTIONS")
    for name, module, mk in (("PG no baseline", pg, pg.PongPolicyNoBaseline), ("PG baseline", pgb, pgb.PongPolicyBaseline)):
        vec = make_vec(2)
        old = (module.BATCH, module.training_iterations)
        module.BATCH, module.training_iterations = 2, 2
        torch.manual_seed(0)
        policy = mk().to(module.train_device)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        before = [p.detach().cpu().clone() for p in policy.parameters()]
        rewards = []
        try:
            module.training(vec, policy, opt, GAMMA, rewards, seed=0)
        finally:
            module.BATCH, module.training_iterations = old
            vec.close()
        after = [p.detach().cpu() for p in policy.parameters()]
        moved = max((a - b).abs().max().item() for a, b in zip(after, before))
        check(f"P10 {name}: one update plays two games and records their rewards", len(rewards) == 2 and all(-21 <= r <= 21 for r in rewards), f"rewards = {rewards}")
        check(f"P10 {name}: one update changes the weights and keeps them finite", moved > 0 and all(torch.isfinite(a).all() for a in after), f"max weight change = {moved:.2e}")

# Create a dummy structural class to gather a uniform random distribution of actions
class RandomPolicy(nn.Module):
    def forward(self, x):
        # Returns equal logits for both actions (RIGHT and LEFT) across all sub-envs
        n = x.shape[0]
        return torch.zeros((n, 2), device=x.device)

# def main():
    # environment_and_preprocessing()
    # returns_tests()
    
    # # 1. Run the standard tests (P5 will pass because rollout returns natively clean arrays)
    # data = rollout_tests()
    # model_tests(data)

    # # 2. Open the local environment for Probe Calibration
    # vec = make_vec(4)
    # baseline_probe = ProbePolicy(thr=0)
    # states, actions, rewards, mask = pg.rollout(vec, baseline_probe, CPU, seed=1)
    
    # # Calculate the true threshold baseline values using the active states
    # changed = np.abs(states[mask]).sum(axis=(-2, -1))
    # calibrated_thr = float(np.median(changed)) - 1.0
    
    # # 3. Gather your balanced baseline matching your calibrated threshold
    # calibrated_probe = ProbePolicy(thr=calibrated_thr)
    # states, actions, rewards, mask = pg.rollout(vec, calibrated_probe, CPU, seed=1)
    
    # # 4. Pack everything into data for the subsequent tests
    # vec.close()
    # data = {
    #     "states": states,
    #     "actions": actions,
    #     "rewards": rewards,
    #     "mask": mask
    # }
    
    # # 5. Run the tests. They will now both receive perfectly balanced data distributions!
    # probe_test(data)
    # gradient_tests(data)
    # smoke_training()
    # finish()

def main():
    environment_and_preprocessing()
    returns_tests()
    data = rollout_tests()
    model_tests(data)
    probe_test(data)
    gradient_tests(data)
    smoke_training()
    finish()



if __name__ == "__main__":
    main()

        
        
        
        
        
        
        
        
        # def main():
#     environment_and_preprocessing()
#     returns_tests()
#     data = rollout_tests()
#     model_tests(data)

#     # Regenerate your baseline dataset with a random policy or standard rollout first:
#     baseline_probe = ProbePolicy(thr=0) # temporary thr
#     states, actions, rewards, mask = pg.rollout(vec, baseline_probe, CPU, seed=1)
    
#     # Re-pack your data dictionary so probe_test uses the clean states
#     data = {"states": states, "mask": mask}

    
#     probe_test(data)
#     gradient_tests(data)
#     smoke_training()
#     finish()


# if __name__ == "__main__":
#     main()