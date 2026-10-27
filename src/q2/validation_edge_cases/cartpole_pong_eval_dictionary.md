# Test Dictionary: CartPole and Pong Validation

Every `check()` call in `validation_cartpole.py`, `validation_pong.py` and the shared `val_common.py`, 110 in total. Each entry gives what the test does, the pass condition, why passing it is evidence of correctness, and the code it exercises.

**Key**
- **Edge** tests (CartPole `C`, Pong `P0` to `P5`) probe boundaries and environment facts. **Validation** tests (CartPole `V`, Pong model, gradient and smoke tests) probe whether the algorithm behaves as theory predicts.
- Two model pairs are covered: REINFORCE without a baseline (`cartpole.py`, `pong.py`) and with a shared-trunk learned baseline (`cartpole_baseline.py`, `pong_baseline.py`). Model tags in check names: `CP no baseline`, `CP baseline`, `PG no baseline`, `PG baseline`.
- Notation: `gamma` = discount factor, `G` = discounted return, `V` = critic value, `ln 2` ≈ 0.693 is the entropy of a uniform two-action policy.

## Totals

| Group | Checks |
|---|---|
| CartPole edge cases (C0 to C10), script-specific | 21 |
| CartPole gradient and baseline checks from `val_common` (C8) | 18 |
| CartPole learning and variance experiments (V1 to V3) | 9 |
| Pong environment, preprocessing, returns, rollout, model, probe and smoke tests | 44 |
| Pong gradient and baseline checks from `val_common` | 18 |
| **Total** | **110** |

Of these, 36 come from `val_common` (6 gradient checks × 4 model tags, plus 6 baseline checks × 2 baseline tags) and 74 are specific to the two validation scripts.

---

## Shared Checks from `val_common.py` (36 check lines)

All gradient checks run in double precision on the CPU. The two functions take real states, actions and returns from rollouts of the model under test.

### Gradient estimator checks: `generic_gradient_checks` (6 checks × 4 models = 24)

Run for `CP no baseline`, `CP baseline` (on 10 CartPole episodes), `PG no baseline` and `PG baseline` (on 200 sampled Pong states). Each check name is `<tag> <ID> <text>`.

| Test | What it does | Passes when | Why this validates |
|---|---|---|---|
| G1 expected score function is zero | Computes the gradient of `sum_a pi(a) log pi(a)` and compares its norm to the gradient of the taken-action log probabilities | Relative norm < 1e-8 | `E[∇ log π] = 0` is the identity that makes baseline subtraction unbiased, so the model's log probabilities and probabilities must be consistent |
| G2 a state dependent baseline term has zero expectation | Uses a random per-state baseline `b ~ N(0, 25)` and compares the expected baseline gradient to the sampled one | Relative norm < 1e-8 | Confirms that any state-dependent baseline adds no bias to the policy gradient |
| G3 summed loss equals the sum of single step gradients | For 32 samples, compares one summed-loss gradient to the return-weighted sum of single-step gradients | Relative difference < 1e-9 | The batched loss must equal the per-step REINFORCE sum `Σ G_t ∇log π(a_t|s_t)` |
| G4 chunked backward equals one pass | Splits the batch into chunks of 64, accumulates gradients, and compares to a single pass | Relative difference < 1e-9 | Chunking (used to save memory) must not change the gradient |
| G5 zero returns give zero update | Runs one Adam step (lr 0.01) with all returns set to 0 | Max weight change is exactly 0 | With no learning signal there should be no update, so nothing leaks in from regularization or a stray loss term |
| G6 autograd gradient matches a finite difference | Compares the gradient norm to a central-difference directional derivative along the gradient (eps 1e-6) | Relative error < 5e-4 | Independent numerical confirmation that the autograd gradient is the true derivative of the loss |

### Actor-critic checks: `baseline_model_checks` (6 checks × 2 baseline models = 12)

Run for `CP baseline` and `PG baseline`. They require the model to expose `.actor`, `.critic` and `.shared`.

| Test | What it does | Passes when | Why this validates |
|---|---|---|---|
| B1 advantage carries no gradient | Checks `adv = G - V.detach()` | `adv.requires_grad` is false | The advantage is a fixed weight in the actor loss; letting gradients flow through it would also train the critic through the policy loss |
| B2 actor loss does not train the critic head | Backpropagates the actor loss and inspects the critic head | Critic weight and bias gradients are zero or `None` | The policy loss must not move the critic head |
| B2 actor loss trains the actor head | Same backward pass, inspects the actor head | Actor weight gradient is nonzero | Confirms the policy loss is connected to the actor |
| B3 critic loss does not train the actor head | Backpropagates the critic loss and inspects the actor head | Actor weight and bias gradients are zero or `None` | The value loss must not move the policy head |
| B3 critic loss trains the critic head | Same backward pass, inspects the critic head | Critic weight gradient is nonzero | Confirms the value loss is connected to the critic |
| B4 a step along the critic gradient lowers the critic loss | Takes a normalized step of size 1e-5 along the critic gradient | New critic loss < old critic loss | Confirms the critic gradient points downhill, so the critic can learn |

The cosine between the actor and critic gradients on the shared trunk is printed as a note and is not checked.

---

## CartPole (`validation_cartpole.py`): 48 checks

### Edge cases (C0 to C10)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| C0 observation has 4 entries and there are 2 actions | Reads the environment spaces | Observation shape `(4,)` and 2 actions | The network input and output sizes must match the environment | `gym.make("CartPole-v1")` |
| C0 episode cap is 500 steps | Reads the environment spec | `max_episode_steps == 500` | The maximum possible reward is the reference point for every learning result | `gym.make("CartPole-v1")` |
| C1 uniform random policy scores about 22 | Runs 500 episodes with uniformly random actions | Mean length in [19, 26] | The random floor that any learned policy must beat | `run_length` |
| C2 untrained no baseline network scores like a random policy | Runs 300 episodes with a freshly initialized network | Mean length in [15, 35] | A correct initialization should not start with a strong or degenerate policy | `CartpolePolicyNoBaseline`, `collect` |
| C2 untrained no baseline network has entropy near ln 2 | Measures mean policy entropy on one episode | Entropy > 0.65 | The initial policy should be nearly uniform so exploration starts unbiased | `CartpolePolicyNoBaseline`, `logits_of` |
| C2 untrained baseline network scores like a random policy | Same as above | Mean length in [15, 35] | Same argument for the actor-critic network | `CartpolePolicyBaseline`, `collect` |
| C2 untrained baseline network has entropy near ln 2 | Same as above | Entropy > 0.65 | Same argument for the actor-critic network | `CartpolePolicyBaseline`, `logits_of` |
| C3 always pushing action 0 ends within about 10 steps | Runs 100 episodes pushing one way | Mean length in [8, 12] | This is what a fully collapsed policy would score, so a collapse is recognizable in the learning curves | `run_length` |
| C3 always pushing action 1 ends within about 10 steps | Same, other direction | Mean length in [8, 12] | Same | `run_length` |
| C4 action 1 pushes the cart right and action 0 pushes it left | Takes one step with each action from the same seed and compares cart velocity | Velocity after action 1 > after action 0 | Confirms the meaning of the network's output indices | `gym` step |
| C5 angular velocity controller clearly beats the random policy | Runs 100 episodes of a hand-written controller through the same interface | Mean length > 50 | Shows the task is solvable through the policy interface and that the episode helper returns correct lengths | `HeuristicPolicy`, `cartpole.episode` |
| C6 no baseline loop ends cleanly on truncation | Runs episodes capped at 20 steps with a controller that does not fail | Every episode has exactly 20 steps | Truncation (time limit) must end the loop just like termination | `cartpole.episode` |
| C6 baseline loop ends cleanly on truncation | Same for the actor-critic loop | Every episode has exactly 20 steps | Same | `cartpole_baseline.collect_episode` |
| C7 both return implementations match the definition at gamma = 0 | Compares both `discounted_returns` to a direct double sum on 50 random rewards | `allclose` (atol 1e-4) | The returns are the learning signal; they must equal the definition | `cartpole.discounted_returns`, `cartpole_baseline.discounted_returns` |
| C7 both return implementations match the definition at gamma = 0.5 | Same | `allclose` | Same, with real discounting | same |
| C7 both return implementations match the definition at gamma = configured value | Same | `allclose` | Same, at the value used in training | same |
| C7 both return implementations match the definition at gamma = 1 | Same | `allclose` | Same, with no discounting | same |
| C7 returns never exceed 1 / (1 - gamma) | Computes returns for 500 rewards of 1 | Max ≤ bound + 1e-5 | The geometric-series upper bound holds, so no accumulation bug inflates returns | `cartpole.discounted_returns` |
| C9 same seed gives identical episode rewards | Runs 20 episodes twice with the same seed | Lists equal | Results must be reproducible | `CartpolePolicyNoBaseline`, `cartpole.episode` |
| C9 different seeds give different episode rewards | Runs with seeds 5 and 6 | Lists differ | Seeding must actually change the run | same |
| C10 batched recomputation matches cartpole.episode | Recomputes log probabilities with the batched `train_nb` pattern and compares to `cartpole.episode` | Rewards equal and log probabilities `allclose` (atol 1e-6) | Ties the experiment trainer to the real episode function, so the learning experiments test the same loss | `collect`, `cartpole.episode` |

Not a check: C7 also prints notes that returns are within 1% of `1/(1-gamma)` for most of a 500-step episode (so only the last few steps carry a learning signal), and saves `val_cartpole_returns.png`.

**C8** runs the shared gradient checks on real CartPole states: G1 to G6 for `CP no baseline` and `CP baseline`, plus B1 to B4 for `CP baseline` (18 checks, defined above).

### Validation experiments (V1 to V3)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| V1 no baseline learns: final avg100 is at least 1.5x the starting avg100 | Trains 3 seeds for 2000 episodes and compares the first and last 100-episode averages | Mean final > 1.5 × mean start | Learning means the reward goes up above the random floor | `train_nb` |
| V1 baseline learns: final avg100 is at least 1.5x the starting avg100 | Same for the actor-critic model, using the real training update | Mean final > 1.5 × mean start | Same, for the real `train_update` | `train_b`, `cartpole_baseline.train_update` |
| V2 no baseline: with the reward negated the policy learns to fail faster | Trains 2 seeds for 1000 episodes with the reward multiplied by -1 | Final length < 0.75 × starting length | If the update truly ascends the reward, flipping the sign must reverse learning | `train_nb`, `SignFlip` |
| V2 baseline: with the reward negated the policy learns to fail faster | Same for the actor-critic model | Final length < 0.75 × starting length | Same | `train_b`, `SignFlip` |
| V2 no baseline: gamma = 0 learns less than the configured gamma | Trains with `gamma = 0`, where every return equals the constant 1 | Final avg100 below the normal run | With constant returns, the expected gradient is zero, so nothing systematic can be learned | `train_nb` |
| V2 baseline: gamma = 0 learns less than the configured gamma | Same for the actor-critic model | Final avg100 below the normal run | Same | `train_b` |
| V3 the critic explains part of the return variance on fresh episodes | Uses 400 fresh episodes from a trained baseline model and computes explained variance of `G` by `V` | Explained variance > 0.2 | A useful baseline must predict returns better than a constant | `collect`, `CartpolePolicyBaseline` |
| V3 subtracting the baseline lowers the gradient variance | Compares total per-episode gradient variance with raw returns and with advantages | Variance ratio < 0.8 | The reason for a baseline: lower variance with no added bias | `flat_grad` |
| V3 averaging BATCH episodes cuts the variance by about BATCH | Compares the variance of batch-mean gradients to per-episode variance divided by `BATCH` | Observed / expected in [0.5, 2] | Independent episodes should average down at rate `1/BATCH`, confirming the estimator is a plain Monte Carlo average | `flat_grad` |

Not a check: **V4 learning-rate sweep** trains both models at three learning rates, prints a table of final average, collapse fraction and minimum entropy, and saves `val_cartpole_lr_sweep.png`. It has no pass condition. V1 also saves `val_cartpole_learning.png`, and V3 saves `val_cartpole_gradient_variance.png`.

---

## Pong (`validation_pong.py`): 62 checks

Pong learns too slowly for learning experiments inside a test, so the checks cover the pieces that decide whether a long run can work.

### Environment and action mapping (P0)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| P0 environment id is ALE/Pong-v5 | Reads the environment spec | Id matches | The right game and version | `gym.make("ALE/Pong-v5")` |
| P0 default action set is the 6 expected actions | Reads `get_action_meanings()` | `NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE` | The action indices the code maps to must correspond to these names | `ale` action meanings |
| P0 the two network outputs map to RIGHT and LEFT | Looks up `ACTION_MAP[0]` and `ACTION_MAP[1]` | Names are RIGHT and LEFT | The 2-output network drives the correct two moves | `pong.ACTION_MAP` |
| P0 frames are 210x160x3 | Reads the observation space | Shape `(210, 160, 3)` | Preprocessing crops assume this layout | `gym.make("ALE/Pong-v5")` |
| P0 sticky action probability is the v5 default 0.25 | Reads the spec kwargs | Value is 0.25 | Documents the stochasticity the agent faces | env spec |
| P0 frame skip is the v5 default 4 | Reads the spec kwargs | Value is 4 | Documents the time scale of one agent step | env spec |

### Preprocessing (P1) and action direction (P3)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| P1 background color 144 is erased | Feeds a frame filled with 144 | Output sums to 0 | Background must not appear as signal | `pong.preprocess` |
| P1 background color 109 is erased | Same with 109 | Output sums to 0 | Same, second background color | `pong.preprocess` |
| P1 rows outside the crop are ignored | Lights rows 10 and 200 | Output sums to 0 | Score banner and bottom border must be cropped | `pong.preprocess` |
| P1 a bright pixel at the crop corner becomes one at (0,0) | Lights pixel (35, 0) | Output has a single 1 at (0,0) | Pins the crop start and the downsample alignment | `pong.preprocess` |
| P1 a bright pixel at the far corner becomes one at (79,79) | Lights pixel (193, 158) | Output has 1 at (79,79) | Pins the crop end and downsample alignment | `pong.preprocess` |
| P1 pixels on odd rows or columns of the crop are dropped by the downsample | Lights (36, 0) and (35, 1) | Output sums to 0 | Documents the stride-2 downsample | `pong.preprocess` |
| P1 output is 80x80 float32 with values in {0, 1} | Checks shape, dtype and values | All hold | Network input contract | `pong.preprocess` |
| P1 a real frame is mostly empty after preprocessing | Preprocesses 80 real frames from random play | Lit fraction in (0, 0.10) | Real frames should reduce to ball, paddles and a few marks, not noise | `pong.preprocess` |
| P1 consecutive frame differences are sparse, so static content cancels | Counts changed pixels between consecutive preprocessed frames | Median < 64 of 6400 | The network input is a frame difference; static content must cancel so the signal is motion | `pong.preprocess` |
| P3 RIGHT and LEFT drive the paddle to opposite ends | Holds each action for up to 100 steps and reads the paddle y from emulator RAM (byte 51) | Median y after step 30 differs by more than 10 | Confirms the action mapping physically moves the paddle in opposite directions | `pong.ACTION_MAP`, ALE RAM |

Not a check: P3 prints which direction increases the RAM value and saves `val_pong_preprocess.png` (the preprocessed frames for RIGHT and LEFT and a frame difference).

### Returns (P4)

The return function resets its running sum at every scored point, so each point is its own short episode.

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| P4 returns are exact on a two point example | Rewards `[0,0,1,0,0,-1]` | Returns equal `[γ², γ, 1, -γ², -γ, -1]` | Hand-computed answer for the per-point discounting | `pong.discounted_returns` |
| P4 changing a later point does not change the returns of earlier points | Changes the final reward from -1 to +1 | First three returns unchanged | Confirms the running sum resets at each point, so credit does not leak across points | `pong.discounted_returns` |
| P4 zero padding after a finished game changes nothing and stays zero | Appends three zero rewards | Original returns unchanged and padding returns are 0 | Vectorized rollouts pad finished games with zeros; padding must not add signal | `pong.discounted_returns` |
| P4 gamma = 0 returns the rewards | Uses `gamma = 0` | Returns equal rewards | Limiting case with a known answer | `pong.discounted_returns` |

### Rollout structure (P5)

Runs a 4-game vectorized rollout with an untrained policy.

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| P5 states are int8 with shape (T, n, 80, 80) | Checks dtype and shape | Both hold | Storage contract for the batched states | `pong.rollout` |
| P5 frame differences only take the values -1, 0, 1 | Lists unique state values | Subset of {-1, 0, 1} | Difference of two binary frames | `pong.rollout` |
| P5 the first state of every game is all zeros | Checks `states[0]` | No nonzero pixel | No previous frame exists at the start, so the first difference must be empty | `pong.rollout` |
| P5 actions are 0 or 1 | Lists unique actions | Subset of {0, 1} | Matches the two network outputs | `pong.rollout` |
| P5 every game is a prefix of True in the mask | Checks the mask is True at t=0 and never turns back on | First row all True and mask non-increasing in time | A finished game stays finished, so padding is only at the end | `pong.rollout` |
| P5 rewards are zero outside the mask | Checks rewards after game end | None nonzero | Padding must not carry reward | `pong.rollout` |
| P5 rewards are -1, 0 or 1 | Lists unique rewards | Subset of {-1, 0, 1} | Pong rewards per point | `pong.rollout` |
| P5 every game ends when one side reaches 21 points | Counts +1 and -1 per game | Max of the two equals 21 in every game | A game must run to completion, not stop early or run long | `pong.rollout` |
| P5 an untrained policy loses almost every point | Averages total score across games | Mean in [-21, -17] | The untrained policy should be near the worst score, the baseline for any learning curve | `pong.rollout`, `PongPolicyNoBaseline` |
| P5 later frame differences contain movement | Computes the share of valid non-initial states with any nonzero pixel | Share > 0.9 | The state should show motion nearly always, otherwise the network sees blank input | `pong.rollout` |

Not a check: P5 prints game lengths, steps per point, and `gamma ** steps_per_point`, which shows how much credit survives across one point.

### Policy and critic outputs (model tests, names without an ID)

Run on 500 real states from the rollout.

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| PG no baseline untrained policy has entropy near ln 2 | Mean entropy on real states | > 0.6 | Initial policy should be close to uniform | `PongPolicyNoBaseline` |
| PG no baseline untrained policy picks RIGHT between 30% and 70% of the time | Mean `p(RIGHT)` | In [0.3, 0.7] | No strong initial bias toward one action | `PongPolicyNoBaseline` |
| PG no baseline one frame and a batch of frames give the same logits | Compares single and batched forward passes on 16 states | `allclose` (atol 1e-5) | Batch handling must not change outputs, since rollouts use single states and training uses batches | `PongPolicyNoBaseline`, `logits_of` |
| PG baseline untrained policy has entropy near ln 2 | Same | > 0.6 | Same | `PongPolicyBaseline` |
| PG baseline untrained policy picks RIGHT between 30% and 70% of the time | Same | In [0.3, 0.7] | Same | `PongPolicyBaseline` |
| PG baseline one frame and a batch of frames give the same logits | Same | `allclose` | Same | `PongPolicyBaseline`, `logits_of` |
| PG baseline critic returns one value per frame | Checks the critic output shape for a batch and a single frame | Shape `(500,)` and scalar | Critic output contract used by the advantage computation | `PongPolicyBaseline` |

### States and actions alignment (P7)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| P7 the probe policy exercises both actions | Builds four synthetic states with 0, 1, 1 and 2 lit pixels and applies the parity rule | Expected actions include both 0 and 1 | The probe must produce both actions to be a meaningful test | `ProbePolicy` |
| P7 each state reproduces its expected action | Runs `ProbePolicy` on the synthetic states | Predicted actions equal the parity rule | Confirms the probe's output is a known function of the state. As written, this uses synthetic states and does not touch the rollout, so rollout state-action alignment is not exercised | `ProbePolicy` |

### Gradient estimator on real Pong states and critic fit

The shared checks run on 200 sampled valid Pong states: G1 to G6 for `PG no baseline` and `PG baseline`, plus B1 to B4 for `PG baseline` (18 checks, defined above). One further check:

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| P9 fitting the critic lowers its error | Trains a fresh critic for 100 Adam steps (lr 1e-3) on 1000 states and returns | Final MSE < 0.95 × initial MSE | The critic must be able to fit the returns, otherwise it cannot serve as a baseline | `PongPolicyBaseline` |

### Smoke runs of the real training functions (P10)

Each runs 2 games for 2 iterations by temporarily setting `BATCH` and `training_iterations` on the module.

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| P10 PG no baseline: one update plays two games and records their rewards | Runs `training` with 2 environments | Two rewards recorded, each in [-21, 21] | The real training loop completes and logs episode scores | `pong.training` |
| P10 PG no baseline: one update changes the weights and keeps them finite | Compares weights before and after | Some change > 0 and all finite | The update actually modifies the network without producing NaN or inf | `pong.training` |
| P10 PG baseline: one update plays two games and records their rewards | Same | Same | Same, for the actor-critic loop | `pong_baseline.training` |
| P10 PG baseline: one update changes the weights and keeps them finite | Same | Same | Same | `pong_baseline.training` |

---

## Notes on the Scripts

- **`algorithmic_policy_direction_check` in `validation_pong.py` is never called from `main()`.** It also uses `F` (`torch.nn.functional`), which is not imported, and reports with `assert` and `print` rather than `check()`, so it would not appear in the PASS/FAIL count even if wired in. `RandomPolicy` and `paddle_row` are likewise defined and unused.
- **Check IDs P2, P6 and P8 do not exist.** The numbering skips them; no checks were dropped from the counts above.
- **The P7 section title says states and actions are aligned, but the checks use synthetic states.** They validate `ProbePolicy`, not the alignment of rollout states with actions.
- **`QUICK` is described in the CartPole docstring but commented out.** The script always runs the full 2000-episode, 3-seed experiments.
- **The CartPole learning checks (V1 to V3) are statistical.** They depend on 2 to 3 seeds and fixed thresholds, so they are the most likely to vary between runs.
- **The Pong smoke tests depend on module globals** (`BATCH`, `training_iterations`, `train_device`) and on the signature `training(vec_env, policy, optimizer, discount_rate, episodic_rewards, seed)`.

## Running

Place `validation_cartpole.py`, `validation_pong.py` and `val_common.py` together, with the model files (`cartpole.py`, `cartpole_baseline.py`, `pong.py`, `pong_baseline.py`) in the parent folder, then run:

```
python validation_cartpole.py
python validation_pong.py
```

Each prints PASS or FAIL per check, saves figures to `./figures`, and exits with a nonzero status if any check fails.