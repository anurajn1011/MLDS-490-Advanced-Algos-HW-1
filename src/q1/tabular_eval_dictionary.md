# Test Dictionary

Every `check()` call in the script, 102 in total. Each entry gives what the test does, the pass condition, why passing it is evidence of correctness, and the code it exercises.

**Key**
- **E tests** support the "Edge cases" requirement. **V tests** support the "Tests for validating correctness" requirement.
- **Code exercised** names the functions a test relies on:
  - `helper`, `enumeration`, `value_iteration` and `policy_iteration` are imported from `enumeration_methods`.
  - **Reference solvers** are defined in the script itself: `build_model`, `pi_exact`, `vi`, `backward`, `evaluate`, `greedy` and `simulate`.
- Notation: `K` = shuttle capacity, `cf` = dispatch cost, `ch` = holding cost per waiting customer, `CAP` = 200, `gamma` = discount rate, `T` = horizon.

---

## E: Edge Cases (28 checks)

### E0: Model construction

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E0 helper matches independent model builder | Compares `helper(K, cf, ch)` output (`P0`, `P1`, `R`) to the separately written `build_model` | `np.allclose` on all three arrays | Two independent implementations agreeing makes a shared transcription mistake unlikely, and every solver downstream depends on these matrices | `helper` vs `build_model` |

### E1: Transition matrix structure

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E1 both matrices are 201x201 | Checks the shapes of `P0` and `P1` | Shape is `(201, 201)` | A wrong size would mean a state was dropped or an arrival indexed past the cap | `helper` |
| E1 rows sum to one | Sums each row of both matrices | All row sums ≈ 1 | Valid transition rows must be probability distributions; any lost or duplicated mass shows up here | `helper` |
| E1 entries are nonnegative | Checks for negative probabilities | All entries ≥ 0 | Negative entries would be an arithmetic error in the construction | `helper` |

### E2: Shuttle larger than the queue

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E2 dispatch cost is exactly cf for every s <= K | Looks at `R[1, 0..K]` | Every value equals `cf` | Confirms the post-dispatch queue is clipped at 0 rather than going negative, so no holding cost is charged | `helper` |
| E2 dispatch from any s <= K leads to the same distribution | Compares rows `P1[0..K]` to `P1[0]` | All rows identical | If the queue is fully cleared, the next state cannot depend on how many were waiting | `helper` |
| E2 dispatching an empty station only wastes cf | Checks `R[1,0]` and `P1[0]` against `P0[0]` | Cost is `cf` and the next-state distribution equals waiting | Dispatch from state 0 changes nothing except the cost, so it must never be preferred | `helper` |

### E3: Partial dispatch

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E3 cost at s=50 is cf + ch*(50-K) | Checks one concrete state with a remainder | `R[1,50] == cf + ch*(50-K)` | A hand-computed value at a state where the shuttle does not clear the queue checks the remainder logic | `helper` |
| E3 next state after dispatch from 50 is 50-K+A | Builds the expected row by hand (arrivals 1 to 5 at 0.2 each) | `P1[50]` equals the hand-built row | Confirms arrivals are added after the dispatch, to the remaining queue | `helper` |

### E4: Capacity overflow

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E4 P0[200,200] == 1 | At the cap, every arrival overflows | Probability is 1.0 | Hand-computed value at the extreme boundary | `helper` |
| E4 P0[199,200] == 1 | From 199, any arrival (1 to 5) reaches or exceeds 200 | Probability is 1.0 | Hand-computed value one step inside the cap | `helper` |
| E4 P0[198,200] == 0.8 | From 198, arrivals 2 to 5 land on 200 | Probability is 0.8 | Only arrivals of 2 or more reach the cap, so 4 of 5 outcomes pool | `helper` |
| E4 P0[197,200] == 0.6 | From 197, arrivals 3 to 5 land on 200 | Probability is 0.6 | Same logic, 3 of 5 outcomes pool | `helper` |
| E4 P0[195,200] == 0.2 and P0[194,200] == 0 | Boundary of where pooling starts | 0.2 at 195 and 0 at 194 | Pins down the exact state where overflow first occurs, catching off-by-one errors | `helper` |
| E4 no overflow mass is lost | Rows still sum to 1 after pooling | All row sums ≈ 1 | Overflow arrivals must be pooled on the cap state, not discarded | `helper` |

### E5: Cost and timing convention

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E5 wait cost is ch*s | Checks the whole `R[0]` vector | Equals `ch * arange(N)` | Holding cost under waiting depends only on the current queue | `helper` |
| E5 dispatch cost is cf + ch*max(s-K,0) | Checks the whole `R[1]` vector | Holding cost is charged on those left behind | Verifies the timing convention: dispatch first, holding cost on the remainder, arrivals afterward | `helper` |

### E6: One shuttle per period

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E6 from 200 a dispatch reaches only 200-K+1 through 200-K+5 | Looks at the support of `P1[200]` | Min support is `200-K+1`, max is `200-K+5` | Shows exactly K are removed per period, not more and not fewer | `helper` |

### E7: Arrivals are never zero

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E7 under wait the queue never decreases | Checks the strict lower triangle of `P0` | All zeros | With no dispatch the queue cannot shrink | `helper` |
| E7 under wait the queue strictly grows except at the cap | Checks the diagonal of `P0` except the last state | All zeros | Arrivals are 1 to 5, so staying in place is impossible below the cap | `helper` |

### E8: Finite-horizon boundary

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E8 last period never dispatches, since V_T is zero | Looks at `pol_e[T-1]` from `enumeration()` | No state dispatches | With terminal value 0, dispatching can only add cost, so the last-period decision tests the boundary condition | `enumeration` |
| E8 policy[0] equals the stationary policy | Compares the time-0 enumeration policy to the infinite-horizon policy | Arrays equal | With a long horizon the time-0 decision should match the stationary optimum | `enumeration`, `pi_exact` |
| E8 early rows are all identical, only the tail differs | Compares each row to row 0 for `t < T-100` | All early rows match | Far from the end the policy should be time-invariant; only the last periods feel the boundary | `enumeration` |

### E9: Ties between actions

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E9 smallest gap between actions is not near zero | Computes `min |v1 - v0|` over states | Gap > 1e-8 | Shows the policy is not decided by floating-point noise | reference solvers |
| E9 flipping the tie rule leaves V unchanged | Evaluates the policy where dispatch wins ties and compares V | `allclose` to `V_STAR` | The optimal value must not depend on how ties are broken | reference solvers |

### E10: Is the cap of 200 distorting the answer?

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| E10 raising the cap to 300 leaves V unchanged for s <= 150 | Re-solves with cap 300 and compares low states | Max difference < 1e-6 | If the cap were driving the result, moving it would change values at low states | reference solvers |
| E10 raising the cap to 300 leaves the policy unchanged for s <= 150 | Same re-solve, compares policies | Arrays equal on states 0 to 150 | Same argument, applied to decisions | reference solvers |
| E10 optimal threshold sits far below the cap | Compares the threshold to `CAP // 2` | Threshold < 100 | The optimal policy dispatches long before the cap is reachable | reference solvers |

---

## V1: Cross-Method Agreement (7 checks)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| V1 enumeration V matches exact reference | `max|V_e - V*|` | < 1e-6 | The exact linear-solve answer involves no iteration or truncation, so agreement shows the enumeration recursion is correct | `enumeration` vs `pi_exact` |
| V1 value iteration V matches exact reference | `max|V_v - V*|` | < 1e-6 | Same argument for value iteration | `value_iteration` vs `pi_exact` |
| V1 policy iteration V matches exact reference | `max|V_p - V*|` | < 1e-6 | Same argument for policy iteration | `policy_iteration` vs `pi_exact` |
| V1 enumeration policy[0] equals VI policy | Compares the time-0 enumeration policy to the VI policy | Arrays equal | Different algorithms reaching the same decisions is strong evidence both are right | `enumeration`, `value_iteration` |
| V1 VI policy equals PI policy | Compares VI and PI policies | Arrays equal | Same argument | `value_iteration`, `policy_iteration` |
| V1 all policies equal the exact reference policy | Compares the PI policy to the exact policy | Arrays equal | Ties the three algorithms to one ground truth | `policy_iteration` vs `pi_exact` |
| V1 enumeration with T=500 and T=1000 agree within gamma**T * norm V | Runs enumeration at two horizons | Difference ≤ `gamma^T * max|V*|` (×1.01) | Doubling the horizon should change V by at most the discounted tail, so this checks the truncation behavior is as theory predicts | `enumeration` |

## V2: Bellman Optimality Residual (2 checks)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| V2 residual of the VI answer is tiny | Plugs the VI value function into the Bellman optimality equation | `max|V - TV| < 1e-7` | The optimal value function is the fixed point of the Bellman operator, so a tiny residual confirms convergence to it | `value_iteration`, `greedy` |
| V2 true error is within gamma/(1-gamma) times the tolerance | Compares the VI value function to the exact V* | Error ≤ `gamma/(1-gamma) * 1e-8` (×1.01) | Checks the stopping rule against the theoretical bound; a printed note states that `delta < tol` only guarantees error ≤ 19×tol | `value_iteration`, `pi_exact` |

## V3: Optimality Against Competing Policies (7 checks + plot)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| V3 optimal policy has threshold structure | Checks the policy is "wait below θ, dispatch at or above θ" | `is_threshold_policy` is true | Matches the expected structure of the problem: more waiting customers should never flip the decision back to wait | `pi_exact` |
| V3 best threshold in the sweep equals the algorithm threshold | Evaluates every threshold policy 0 to 201 and takes the argmin of V(0) | Best θ equals the algorithm θ | Brute-force search over all threshold policies independently finds the same optimum | `evaluate`, `pi_exact` |
| V3 minimum swept cost equals V*[0] | Compares the best swept cost to V*[0] | Within 1e-6 | The brute-force minimum cost and the solver value agree numerically | `evaluate`, `pi_exact` |
| V3 optimal V is below every threshold policy at every state | Compares V* to each swept V | `Vs ≥ V* - 1e-6` everywhere | An optimal value function cannot be beaten by any competing policy at any state | `evaluate`, `pi_exact` |
| V3 always dispatch is worse | Compares θ=0 cost to V*[0] | Strictly greater | Trivial baseline must lose | `evaluate`, `pi_exact` |
| V3 never dispatch is worse | Compares θ=201 cost to V*[0] | Strictly greater | Trivial baseline must lose | `evaluate`, `pi_exact` |
| V3 no single state deviation improves any state | Flips the action at each state, re-evaluates exactly, and takes the minimum improvement | Most negative improvement ≥ -1e-6 | The policy improvement theorem: a policy with no improving single-state change is optimal | `evaluate`, `pi_exact` |

Plot: `v3_threshold_sweep.png` shows V(0) against θ with the algorithm θ marked.

## V4: Monte Carlo Against the Solvers (11 checks)

The simulator is written directly from the problem text and never touches `P0`, `P1` or `r`. This is the only test family that can catch a modeling error shared by both model builders. Each check passes when `|z| < 4`, where `z = (MC mean − exact V) / standard error`, with 20,000 episodes and a 300-step horizon (a T-step horizon for the time-indexed enumeration policy).

| Test | What it does | Why this validates | Code exercised |
|---|---|---|---|
| V4 optimal, start 0 | Simulates the optimal policy from state 0 | Sampled discounted cost matching the solver value shows the model matches the problem text | `simulate`, `evaluate` |
| V4 optimal, start 30 | Same, from state 30 | Tests a mid-range state | `simulate`, `evaluate` |
| V4 optimal, start 100 | Same, from state 100 | Tests a high state, near dispatch territory | `simulate`, `evaluate` |
| V4 never dispatch, start 0 | Simulates "always wait" | A non-optimal policy checks the evaluation machinery, not just the optimum, and exercises the cap overflow | `simulate`, `evaluate` |
| V4 never dispatch, start 30 | Same, from state 30 | Same | `simulate`, `evaluate` |
| V4 never dispatch, start 100 | Same, from state 100 | Same | `simulate`, `evaluate` |
| V4 threshold θ+10, start 0 | Simulates a deliberately suboptimal threshold | Confirms the model is right for policies other than the optimal one | `simulate`, `evaluate` |
| V4 threshold θ+10, start 30 | Same, from state 30 | Same | `simulate`, `evaluate` |
| V4 threshold θ+10, start 100 | Same, from state 100 | Same | `simulate`, `evaluate` |
| V4 enumeration time indexed policy, start 0 | Simulates the full T-step time-varying policy and compares to the enumeration V0 | Validates the time-indexed policy and values from enumeration directly | `enumeration`, `simulate` |
| V4 enumeration time indexed policy, start 100 | Same, from state 100 | Same | `enumeration`, `simulate` |

## V5: Limiting Cases With Known Answers (17 checks)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| V5 gamma=0 VI matches closed form | VI with `discount_rate=0` | V equals `min(ch*s, cf + ch*max(s-K,0))` | With no future, V is just the cheaper immediate cost, which can be computed by hand | `value_iteration` |
| V5 T=1 enumeration matches closed form | Enumeration with a one-period horizon | Same closed form | A one-period problem is the same as γ=0 | `enumeration` |
| V5 cf > ch*K so the myopic policy never dispatches | At γ=0, dispatching never pays since `cf > ch*K` | Policy never dispatches and V = `ch*s` | Dispatch saves at most `ch*K` this period, which is less than its cost | `value_iteration` |
| V5 cf=0 VI: dispatch for every s >= 1 | Free dispatch | Dispatches at all states ≥ 1 | If dispatch costs nothing, removing waiting customers is always beneficial | `value_iteration` |
| V5 cf=0 VI: state 0 waits since the tie goes to wait | Dispatching an empty station is a tie | State 0 waits | Confirms the tie-breaking rule behaves as documented | `value_iteration` |
| V5 cf=0 PI: dispatch for every s >= 1 | Same as above, for PI | Dispatches at all states ≥ 1 | Same argument, second algorithm | `policy_iteration` |
| V5 cf=0 PI: state 0 waits since the tie goes to wait | Same as above, for PI | State 0 waits | Same argument, second algorithm | `policy_iteration` |
| V5 cf=1e6 VI: never dispatch | Prohibitive dispatch cost | No dispatch anywhere | An enormous fixed cost must make dispatching never worthwhile | `value_iteration` |
| V5 cf=1e6 PI: never dispatch | Same, for PI | No dispatch anywhere | Same argument, second algorithm | `policy_iteration` |
| V5 ch=0 VI: V is zero and never dispatch | No holding cost | V ≈ 0 and no dispatch | With no waiting cost and a positive dispatch cost, the cheapest plan is to do nothing | `value_iteration` |
| V5 ch=0 PI: V is zero and never dispatch | Same, for PI | V ≈ 0 and no dispatch | Same argument, second algorithm | `policy_iteration` |
| V5 K=200 every dispatch row is identical | Shuttle as large as the station | All `P1` rows equal `P1[0]` | A dispatch always empties the station, so the next state is independent of the current one | `helper` |
| V5 K=200 dispatch cost is exactly cf everywhere | Same setup | `R[1]` is constant at `cf` | Nobody is left behind, so no holding cost | `helper` |
| V5 deterministic model has a single 1 in every row | Builds the model with exactly 3 arrivals per period | Every row max is 1 | Confirms the stochasticity was actually removed | `build_model` |
| V5 deterministic: VI and PI agree | Solves the near-deterministic problem with both reference solvers | V values and policies match | Two algorithms agreeing on a simplified problem | `vi`, `pi_exact` |
| V5 deterministic: algorithm threshold attains the brute force minimum cost | Compares the found θ cost to the minimum over all θ | Costs equal | In a deterministic problem brute force is exact; costs are compared rather than θ values because different θ can tie when the path 0, 3, 6, … skips states | `pi_exact`, `evaluate` |
| V5 deterministic: minimum swept cost equals V*[0] | Compares the brute-force minimum to the solver V(0) | Within 1e-6 | Brute force and the solver agree on the optimal cost | `pi_exact`, `evaluate` |

## V6: Comparative Statics (8 checks + plot)

Sweeps `cf ∈ {10, 25, 50, 100, 200, 400}`, `ch ∈ {0.5, 1, 2, 4, 8}` and `K ∈ {1, 5, 10, 15, 25, 50, 100, 200}`. All sweeps use `pi_exact` on models from `build_model`.

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| V6 threshold is nondecreasing in cf | Raises the dispatch cost | θ never falls as `cf` rises | Costlier dispatches should be delayed until more people wait | `pi_exact` |
| V6 threshold is nonincreasing in ch | Raises the holding cost | θ never rises as `ch` rises | Costlier waiting should trigger earlier dispatch | `pi_exact` |
| V6 V[0] is nondecreasing in cf | Raises the dispatch cost | Differences ≥ -1e-9 | Higher costs cannot lower total cost | `pi_exact` |
| V6 V[0] is nondecreasing in ch | Raises the holding cost | Differences ≥ -1e-9 | Same argument | `pi_exact` |
| V6 V[0] is nonincreasing in K, a bigger shuttle can only help | Raises the shuttle capacity | Differences ≤ 1e-9 | A larger shuttle never hurts, since it can always carry fewer people | `pi_exact` |
| V6 V is nondecreasing in s in every sweep | Checks V across states in all 19 solves | Differences ≥ -1e-9 | More waiting customers should never lower cost | `pi_exact` |
| V6 optimal policy has threshold structure whenever K >= 10 | Checks every policy where the shuttle can keep up with arrivals (mean 3 per period) | All are threshold policies | Threshold structure is the expected form when capacity exceeds arrivals | `pi_exact` |
| V6 K=5 departure from threshold structure is confined to states within 10 of the cap | Documents an artifact at small K | Any re-wait state is ≥ 190 | Near the cap, overflow arrivals are lost for free, which makes waiting cheaper; this is an artifact of the no-penalty overflow assumption, not a bug | `pi_exact` |

Plot: `v6_comparative_statics.png`.

## V7: Convergence Behavior (10 checks + plot)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| V7 VI log delta slope is close to log gamma | Fits a line to `log(delta)` over the second half of iterations | Slope within 0.005 of `log(gamma)` | Value iteration is a γ-contraction, so the update size must shrink at rate γ | `vi` |
| V7 VI error obeys the contraction bound at every iteration | Compares true error to `gamma^n * max|V*|` | Holds at every iteration | The theoretical error bound must hold at every step, not just at the end | `vi` |
| V7 enumeration error decays like gamma**T | Computes error at T = 1, 5, 10, 25, 50, 100, 500 | Each error ≤ `gamma^k * max|V*|` | Finite-horizon error against the infinite-horizon answer should shrink geometrically | recursion over `build_model` output |
| V7 T=500 truncation error is negligible, below 1e-7 | Checks the error at T=500 | < 1e-7 | Shows T=500 is long enough to be treated as infinite horizon | recursion over `build_model` output |
| V7 enumeration T=1 matches independent backward induction | `enumeration(T=1)` vs `backward` | `allclose` | A separately written backward induction should produce identical values | `enumeration`, `backward` |
| V7 enumeration T=5 matches independent backward induction | Same, T=5 | `allclose` | Same argument | `enumeration`, `backward` |
| V7 enumeration T=25 matches independent backward induction | Same, T=25 | `allclose` | Same argument | `enumeration`, `backward` |
| V7 enumeration T=100 matches independent backward induction | Same, T=100 | `allclose` | Same argument | `enumeration`, `backward` |
| V7 PI converges in a small number of rounds | Counts policy-evaluation rounds | Fewer than 15 | Policy iteration converges in few rounds when correct; many rounds would signal a flaw | `pi_exact` |
| V7 V of each PI iterate is nonincreasing at every state | Checks monotone improvement across rounds | All differences ≤ 1e-9 | Policy improvement guarantees the cost never rises between rounds | `pi_exact` |

Plot: `v7_convergence.png`.

## V8: Initialization and Tolerance Robustness (12 checks)

| Test | What it does | Passes when | Why this validates | Code exercised |
|---|---|---|---|---|
| V8 VI from random start of scale 1 | VI from `V0 ~ U(-1, 1)` | Error within the bound and policy equals the reference | The fixed point is unique, so the starting point must not matter | `vi` |
| V8 VI from random start of scale 100 | Same, scale 100 | Same | Same argument with a worse starting guess | `vi` |
| V8 VI from random start of scale 10000 | Same, scale 10,000 | Same | Same argument with a far worse starting guess | `vi` |
| V8 PI from 10 random initial policies reaches the same answer | Runs PI from 10 random initial policies | All reach V* (atol 1e-9) and the reference policy | Policy iteration must reach the same optimum from any initial policy | `pi_exact` |
| V8 tol=0.0001: error is within the bound | Tolerance sweep | Error ≤ `gamma/(1-gamma) * tol` | The stopping rule must respect the theoretical bound at any tolerance | `vi` |
| V8 tol=1e-06: error is within the bound | Same | Same | Same | `vi` |
| V8 tol=1e-08: error is within the bound | Same | Same | Same | `vi` |
| V8 tol=1e-10: error is within the bound | Same | Same | Same | `vi` |
| V8 tol=1e-12: error is within the bound | Same | Same | Same | `vi` |
| V8 tol=1e-08: policy equals the reference | Policy check at tight tolerance | Arrays equal | A tight tolerance should recover the exact optimal policy | `vi` |
| V8 tol=1e-10: policy equals the reference | Same | Arrays equal | Same | `vi` |
| V8 tol=1e-12: policy equals the reference | Same | Arrays equal | Same | `vi` |

---

## Not Checks: Required Plots (parts A, B, C)

`required_plots()` saves the three figures the assignment asks for and runs no assertions.

- `part_a_enumeration_V0.png`: optimal value at time 0 from enumeration.
- `part_b_value_iteration_V.png`: optimal value from value iteration.
- `part_c_policy_iteration_policy.png`: optimal policy from policy iteration.

## Totals

| Group | Checks |
|---|---|
| E0 to E10 | 28 |
| V1 | 7 |
| V2 | 2 |
| V3 | 7 |
| V4 | 11 |
| V5 | 17 |
| V6 | 8 |
| V7 | 10 |
| V8 | 12 |
| **Total** | **102** |

## Running
```python
if __name__ == "__main__":
    main()
```