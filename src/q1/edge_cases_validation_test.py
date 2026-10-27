"""
edge_cases_validation_test.py

Edge case verification and validation experiments for the shuttle dispatch MDP. Run from the same folder as enumeration_methods.py with:  python validation.py
Every check prints PASS or FAIL. Figures are saved to ./figures.

The script exits with a nonzero status if any check fails.

Convention used throughout: action 1 means dispatch, action 0 means wait, and r holds costs, so every algorithm minimizes.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, figures go straight to disk
import matplotlib.pyplot as plt

from enumeration_methods import helper, enumeration, value_iteration, policy_iteration, q1

### Config Constants
K = q1["K"]
CF = q1["c_f"]
CH = q1["c_h"]
GAMMA = q1["discount_rate"]
T = int(q1["time_period"])
CAP = q1["station_capacity"]
N = CAP + 1  # number of states, 0 through CAP inclusive

EPISODES = 20000  # Monte Carlo episodes per estimate
MC_HORIZON = 300 # gamma**300 is about 2e-7, so truncation bias is negligible
TOL_V = 1e-6 # allowed gap between two value functions that should agree
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

RNG = np.random.default_rng(0)  # fixed seed so the run is reproducible
RESULTS = []

def check(name, ok, detail=""):
    """Record and print one PASS or FAIL line."""
    RESULTS.append((name, bool(ok)))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"\t{detail}" if detail else ""))

def section(title):
    print("\n" + "#" * 10 + f"\n{title}\n" + "#" * 10)

### Second implementation of the models with the ability to sweep over different capacities and arrival rules. 
def build_model(K_, cf_, ch_, cap, arrivals=None):
    """Return P0, P1, r. arrivals maps arrival count to probability."""
    if arrivals is None:
        arrivals = {a: 0.2 for a in range(1, 6)}
    n = cap + 1
    P0 = np.zeros((n, n))
    P1 = np.zeros((n, n))
    r = np.zeros((2, n))
    for s in range(n):
        post = max(s - K_, 0)  # queue left after a dispatch, never negative
        r[0, s] = ch_ * s
        r[1, s] = cf_ + ch_ * post  # holding cost is charged on those left behind
        for a, p in arrivals.items():
            P0[s, min(s + a, cap)] += p      # min pools overflow mass on the cap state
            P1[s, min(post + a, cap)] += p
    return P0, P1, r

def greedy(P0, P1, r, gamma, V):
    """One step lookahead values and the greedy policy. Strict < means ties wait."""
    v0 = r[0] + gamma * P0 @ V
    v1 = r[1] + gamma * P1 @ V
    return v0, v1, v1 < v0

def evaluate(policy, P0, P1, r, gamma):
    """Exact V under a fixed policy through a linear solve, no iteration. Part of Policy Iteration."""
    pol = np.asarray(policy).astype(bool)
    P = np.where(pol[:, None], P1, P0)  # column vector mask picks whole rows
    c = np.where(pol, r[1], r[0])
    return np.linalg.solve(np.eye(len(c)) - gamma * P, c)

def vi(P0, P1, r, gamma, tol=1e-8, V0=None, keep=False):
    """Value iteration. Returns V, policy, list of deltas, optional iterate history."""
    V = np.zeros(r.shape[1]) if V0 is None else np.array(V0, dtype=float)
    deltas, hist = [], []
    for _ in range(200000):  # hard cap so a bug cannot hang the script
        V_new = np.minimum(r[0] + gamma * P0 @ V, r[1] + gamma * P1 @ V)
        d = np.max(np.abs(V_new - V))
        deltas.append(d)
        V = V_new
        if keep:
            hist.append(V.copy())
        if d < tol:
            break
    _, _, pol = greedy(P0, P1, r, gamma, V)
    return V, pol, deltas, hist

def pi_exact(P0, P1, r, gamma, pol0=None):
    """Policy iteration with exact evaluation. Returns V, policy, list of V per round."""
    n = r.shape[1]
    pol = np.zeros(n, dtype=bool) if pol0 is None else np.asarray(pol0).astype(bool)
    hist = []
    for _ in range(1000):
        V = evaluate(pol, P0, P1, r, gamma)
        hist.append(V)
        _, _, new = greedy(P0, P1, r, gamma, V)
        if np.array_equal(new, pol):
            break
        pol = new
    return V, pol, hist

def backward(P0, P1, r, gamma, horizon):
    """Finite horizon backward induction with terminal value zero."""
    V = np.zeros(r.shape[1])
    pol = np.zeros((horizon, r.shape[1]), dtype=bool)
    for t in reversed(range(horizon)):
        v0, v1, pol[t] = greedy(P0, P1, r, gamma, V)
        V = np.minimum(v0, v1)
    return V, pol

def threshold(policy):
    """First state where the policy dispatches. Equals len when it never dispatches."""
    idx = np.flatnonzero(policy)
    return int(idx[0]) if len(idx) else len(policy)

def is_threshold_policy(policy):
    th = threshold(policy)
    return np.array_equal(np.asarray(policy).astype(bool), np.arange(len(policy)) >= th)

def simulate(policy, s0, episodes, horizon, rng, time_indexed=False):
    """
    Monte Carlo of the discounted cost, written straight from the problem text. It never touches P0, P1 or r, so it can expose a modeling mistake that a
    self consistent solver would miss. Returns mean and standard error.
    """
    s = np.full(episodes, s0)
    total = np.zeros(episodes)
    for t in range(horizon):
        pol_t = policy[t] if time_indexed else policy
        a = pol_t[s].astype(bool)
        post = np.where(a, np.maximum(s - K, 0), s) # dispatch removes up to K
        total += GAMMA ** t * (CF * a + CH * post) # pay dispatch and holding cost
        s = np.minimum(post + rng.integers(1, 6, episodes), CAP)  # integers upper bound is exclusive
    return total.mean(), total.std() / np.sqrt(episodes)

# Shared Vals
P0, P1, R = helper(K, CF, CH)
Q0, Q1, QR = build_model(K, CF, CH, CAP)
V_STAR, POL_STAR, _ = pi_exact(Q0, Q1, QR, GAMMA)  # exact reference answer for policy iteration, solved using linear algebra.

### Edge Case Testing
def edge_cases():
    section("EDGE CASES")

    # E0 the helper agrees with an independent construction
    check("E0 helper matches independent model builder",
          np.allclose(P0, Q0) and np.allclose(P1, Q1) and np.allclose(R, QR))

    # E1 structural validity of the transition matrices
    check("E1 both matrices are 201x201", P0.shape == (N, N) and P1.shape == (N, N),
          "a column index outside 0..CAP would have raised an IndexError or changed the shape")
    check("E1 rows sum to one", np.allclose(P0.sum(1), 1) and np.allclose(P1.sum(1), 1))
    check("E1 entries are nonnegative", (P0 >= 0).all() and (P1 >= 0).all())

    # E2 dispatch with s <= K clears everyone but still pays cf
    check("E2 dispatch cost is exactly cf for every s <= K", np.allclose(R[1, :K + 1], CF))
    check("E2 dispatch from any s <= K leads to the same distribution",
          np.allclose(P1[:K + 1], P1[0]))
    check("E2 dispatching an empty station only wastes cf",
          R[1, 0] == CF and np.allclose(P1[0], P0[0]))

    # E3 dispatch with s > K leaves the remainder behind
    s_test = 50
    expected = np.zeros(N)
    for a in range(1, 6):
        expected[s_test - K + a] = 0.2
    check("E3 cost at s=50 is cf + ch*(50-K)", np.isclose(R[1, s_test], CF + CH * (s_test - K)))
    check("E3 next state after dispatch from 50 is 50-K+A", np.allclose(P1[s_test], expected))

    # E4 station cap and partial overflow
    check("E4 P0[200,200] == 1", np.isclose(P0[200, 200], 1.0))
    check("E4 P0[199,200] == 1", np.isclose(P0[199, 200], 1.0))
    check("E4 P0[198,200] == 0.8", np.isclose(P0[198, 200], 0.8),
          "arrivals 2 to 5 all land on 200")
    check("E4 P0[197,200] == 0.6", np.isclose(P0[197, 200], 0.6))
    check("E4 P0[195,200] == 0.2 and P0[194,200] == 0",
          np.isclose(P0[195, 200], 0.2) and np.isclose(P0[194, 200], 0.0))
    check("E4 no overflow mass is lost anywhere", np.allclose(P0.sum(1), 1))

    # E5 costs depend only on state and action, arrivals join after the dispatch
    check("E5 wait cost is ch*s", np.allclose(R[0], CH * np.arange(N)))
    check("E5 dispatch cost is cf + ch*max s-K 0",
          np.allclose(R[1], CF + CH * np.maximum(np.arange(N) - K, 0)))

    # E6 only one shuttle per period
    post = CAP - K
    support = np.flatnonzero(P1[CAP])
    check("E6 from 200 a dispatch reaches only 200-K+1 through 200-K+5",
          support.min() == post + 1 and support.max() == post + 5)

    # E7 arrivals are never zero, so waiting can never shrink the queue
    check("E7 under wait the queue never decreases", np.allclose(np.tril(P0, -1), 0))
    check("E7 under wait the queue strictly grows except at the cap",
          np.allclose(np.diag(P0)[:-1], 0))

    # E8 horizon boundary of the finite horizon problem
    V_e, pol_e = enumeration()
    pol_e = pol_e.astype(bool)
    check("E8 last period never dispatches, since V_T is zero", not pol_e[T - 1].any())
    same = np.array([np.array_equal(pol_e[t], pol_e[0]) for t in range(T)])
    first_diff = int(np.argmax(~same)) if (~same).any() else T
    check("E8 policy[0] equals the stationary policy", np.array_equal(pol_e[0], POL_STAR))
    check("E8 early rows are all identical, only the tail differs",
          same[:T - 100].all(), f"rows match policy[0] up to t={first_diff - 1} of {T - 1}")

    # E9 ties between actions
    v0, v1, pol = greedy(Q0, Q1, QR, GAMMA, V_STAR)
    gap = np.abs(v1 - v0).min()
    check("E9 smallest gap between actions is not near zero", gap > 1e-8, f"min gap = {gap:.3e}")
    pol_alt = v1 <= v0  # opposite tie rule, dispatch wins ties
    V_alt = evaluate(pol_alt, Q0, Q1, QR, GAMMA)
    check("E9 flipping the tie rule leaves V unchanged", np.allclose(V_alt, V_STAR, atol=1e-9))

    # E10 the cap of 200 is harmless when it is not binding
    P0b, P1b, Rb = build_model(K, CF, CH, 300)
    V_big, pol_big, _ = pi_exact(P0b, P1b, Rb, GAMMA)
    lim = 150  # compare only low states, where the cap cannot matter
    check("E10 raising the cap to 300 leaves V unchanged for s <= 150",
          np.allclose(V_big[:lim + 1], V_STAR[:lim + 1], atol=1e-6),
          f"max diff = {np.abs(V_big[:lim + 1] - V_STAR[:lim + 1]).max():.2e}")
    check("E10 raising the cap to 300 leaves the policy unchanged for s <= 150",
          np.array_equal(pol_big[:lim + 1], POL_STAR[:lim + 1]))
    check("E10 optimal threshold sits far below the cap", threshold(POL_STAR) < CAP // 2,
          f"threshold = {threshold(POL_STAR)}")

### Methods for validating results
def v1_cross_method():
    '''
    We check our results from the three algorithms as well as against a solution solved using linear algebra in Policy Iteration. We also check that the solutions
    at time horizons 500 and 1000 agree by some power of the discount factor. 
    '''
    section("V1 CROSS METHOD AGREEMENT")
    V_e, pol_e = enumeration()
    V_v, pol_v = value_iteration()
    V_p, pol_p = policy_iteration()
    pol_e0 = pol_e[0].astype(bool)

    check("V1 enumeration V matches exact reference", np.abs(V_e - V_STAR).max() < TOL_V,
          f"{np.abs(V_e - V_STAR).max():.2e}")
    check("V1 value iteration V matches exact reference", np.abs(V_v - V_STAR).max() < TOL_V,
          f"{np.abs(V_v - V_STAR).max():.2e}")
    check("V1 policy iteration V matches exact reference", np.abs(V_p - V_STAR).max() < TOL_V,
          f"{np.abs(V_p - V_STAR).max():.2e}")
    check("V1 enumeration policy[0] equals VI policy", np.array_equal(pol_e0, pol_v.astype(bool)))
    check("V1 VI policy equals PI policy", np.array_equal(pol_v.astype(bool), pol_p.astype(bool)))
    check("V1 all policies equal the exact reference policy",
          np.array_equal(pol_p.astype(bool), POL_STAR))

    # Convergence in T, a longer horizon should change V0 by roughly gamma**T
    V_1000, _ = enumeration(T=1000)
    trunc_bound = GAMMA ** T * np.abs(V_STAR).max()  # error of a horizon T answer is at most this
    check("V1 enumeration with T=500 and T=1000 agree within gamma**T * norm V",
          np.abs(V_e - V_1000).max() <= trunc_bound * 1.01,
          f"diff = {np.abs(V_e - V_1000).max():.2e}, bound = {trunc_bound:.2e}")
    return V_e, pol_e, V_v, pol_v, V_p, pol_p

def v2_bellman_residual(V_v):
    '''
    Checks the solution provided by value iteration and plugs it into the Bellman Equation to verify it. Also checks the error is within some distance of the tolerance. 
    '''
    section("V2 BELLMAN OPTIMALITY RESIDUAL")
    v0, v1, _ = greedy(Q0, Q1, QR, GAMMA, V_v)
    res = np.abs(V_v - np.minimum(v0, v1)).max()
    true_err = np.abs(V_v - V_STAR).max()
    bound = GAMMA / (1 - GAMMA) * 1e-8
    check("V2 residual of the VI answer is tiny", res < 1e-7, f"residual = {res:.2e}")
    check("V2 true error is within gamma/(1-gamma) times the tolerance", true_err <= bound * 1.01,
          f"true error = {true_err:.2e}, bound = {bound:.2e}")
    # Stopping on delta below tol does not mean the error is below tol, the true error can be up to 19 times larger at gamma 0.95
    print(f"\tnote: delta < tol only guarantees error <= {GAMMA / (1 - GAMMA):.0f} * tol")

def v3_optimality():
    '''
    Test our policy found via the three methods against other competing policies and see if we can find one better. We also check what happens if we change 
    the policy taken at each state one by one. We also check our policy against the trivial policies of always dispatch and never dispatch. 
    '''
    section("V3 OPTIMALITY AGAINST COMPETING POLICIES")
    pol_star = POL_STAR
    check("V3 optimal policy has threshold structure", is_threshold_policy(pol_star),
          f"dispatch iff s >= {threshold(pol_star)}")

    # Sweep every threshold policy, n means never dispatch and 0 means always dispatch
    Vs = np.array([evaluate(np.arange(N) >= th, Q0, Q1, QR, GAMMA) for th in range(N + 1)])
    costs0 = Vs[:, 0]
    best_th = int(np.argmin(costs0))
    check("V3 best threshold in the sweep equals the algorithm threshold",
          best_th == threshold(pol_star), f"sweep = {best_th}, algorithm = {threshold(pol_star)}")
    check("V3 minimum swept cost equals V*[0]", np.isclose(costs0.min(), V_STAR[0], atol=1e-6))
    check("V3 optimal V is below every threshold policy at every state",
          (Vs >= V_STAR - 1e-6).all())
    check("V3 always dispatch is worse", costs0[0] > V_STAR[0] + 1e-6, f"{costs0[0]:.2f} vs {V_STAR[0]:.2f}")
    check("V3 never dispatch is worse", costs0[N] > V_STAR[0] + 1e-6, f"{costs0[N]:.2f} vs {V_STAR[0]:.2f}")

    # Single state deviations, flip the action at one state and re-evaluate
    worst = np.inf
    for s in range(N):
        p = pol_star.copy()
        p[s] = ~p[s]
        worst = min(worst, (evaluate(p, Q0, Q1, QR, GAMMA) - V_STAR).min())
    check("V3 no single state deviation improves any state", worst >= -1e-6,
          f"most negative improvement = {worst:.2e}")

    plt.figure(figsize=(7, 4))
    plt.plot(range(N + 1), costs0, lw=1.5)
    plt.axvline(threshold(pol_star), color="r", ls="--", label=f"algorithm threshold = {threshold(pol_star)}")
    plt.ylim(V_STAR[0] * 0.9, V_STAR[0] * 2)  # zoom so the basin is visible
    plt.xlabel("dispatch threshold theta")
    plt.ylabel("exact V(0) under threshold policy")
    plt.title("V3: value at state 0 for every threshold policy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "v3_threshold_sweep.png", dpi=150)
    plt.close()

def v4_monte_carlo(pol_e):
    '''
    A monte carlo simulation of the problem with the given parameters. Just to verify if a simulation matches the algorithm solution we devised. 
    '''
    section("V4 MONTE CARLO AGAINST THE SOLVERS")
    th = threshold(POL_STAR)
    policies = {
        "optimal": POL_STAR,
        "never dispatch": np.zeros(N, dtype=bool),
        f"threshold {th + 10}": np.arange(N) >= th + 10,
    }
    for name, pol in policies.items():
        V_exact = evaluate(pol, Q0, Q1, QR, GAMMA)
        for s0 in (0, 30, 100):
            m, se = simulate(pol, s0, EPISODES, MC_HORIZON, RNG)
            z = (m - V_exact[s0]) / se
            check(f"V4 {name}, start {s0}: simulation matches exact V", abs(z) < 4,
                  f"MC = {m:.2f} +/- {se:.2f}, exact = {V_exact[s0]:.2f}, z = {z:.2f}")

    # Enumeration needs the time indexed policy over the full horizon
    pol_t = pol_e.astype(bool)
    V_e, _ = enumeration()
    for s0 in (0, 100):
        m, se = simulate(pol_t, s0, EPISODES, T, RNG, time_indexed=True)
        z = (m - V_e[s0]) / se
        check(f"V4 enumeration time indexed policy, start {s0}", abs(z) < 4,
              f"MC = {m:.2f} +/- {se:.2f}, V0 = {V_e[s0]:.2f}, z = {z:.2f}")

def v5_limiting_cases():
    '''
    Verifies cases where we know the answer such as when gamma is 0, when cf is 0 or very large, when ch is 0, and what happens when the capacity of a shuttle is greater than the capacity of the station. We also check if our algorithms agree when there is a determined number of arrivals every period, say 3. 
    '''
    section("V5 LIMITING CASES WITH KNOWN ANSWERS")
    s = np.arange(N)

    # gamma = 0 means a one step problem, V is the cheaper of the two immediate costs
    closed = np.minimum(CH * s, CF + CH * np.maximum(s - K, 0))
    V0g, pol0g = value_iteration(discount_rate=0.0)
    check("V5 gamma=0 VI matches closed form", np.allclose(V0g, closed))
    V_t1, pol_t1 = enumeration(T=1)
    check("V5 T=1 enumeration matches closed form", np.allclose(V_t1, closed))
    check("V5 cf > ch*K so the myopic policy never dispatches",
          CF > CH * K and not pol0g.any() and np.allclose(V0g, CH * s),
          f"cf = {CF}, ch*K = {CH * K}")

    # cf = 0, dispatch is free so dispatch whenever anyone is waiting
    for name, fn in (("VI", value_iteration), ("PI", policy_iteration)):
        _, p = fn(cf=0)
        p = p.astype(bool)
        check(f"V5 cf=0 {name}: dispatch for every s >= 1", p[1:].all())
        check(f"V5 cf=0 {name}: state 0 waits since the tie goes to wait", not p[0])

    # cf huge, never dispatch
    for name, fn in (("VI", value_iteration), ("PI", policy_iteration)):
        _, p = fn(cf=1e6)
        check(f"V5 cf=1e6 {name}: never dispatch", not p.astype(bool).any())

    # ch = 0, waiting is free so V is zero and nothing is ever dispatched
    for name, fn in (("VI", value_iteration), ("PI", policy_iteration)):
        V, p = fn(ch=0)
        check(f"V5 ch=0 {name}: V is zero and never dispatch",
              np.allclose(V, 0) and not p.astype(bool).any())

    # K >= cap, a dispatch always empties the station
    Pk0, Pk1, Rk = helper(K=CAP, cf=CF, ch=CH)
    check("V5 K=200 every dispatch row is identical", np.allclose(Pk1, Pk1[0]))
    check("V5 K=200 dispatch cost is exactly cf everywhere", np.allclose(Rk[1], CF))

    # Deterministic arrivals, exactly three per period
    D0, D1, DR = build_model(K, CF, CH, CAP, arrivals={3: 1.0})
    check("V5 deterministic model has a single 1 in every row", (D0.max(1) == 1).all() and (D1.max(1) == 1).all())
    Vd, pold, _ = pi_exact(D0, D1, DR, GAMMA)
    Vv, polv, _, _ = vi(D0, D1, DR, GAMMA)
    Vs = np.array([evaluate(np.arange(N) >= th, D0, D1, DR, GAMMA)[0] for th in range(N + 1)])
    check("V5 deterministic: VI and PI agree", np.allclose(Vd, Vv, atol=1e-6) and np.array_equal(pold, polv))
    # Different thresholds can give identical cost from state 0 when the deterministic path 0,3,6,... skips over some states, so compare costs and not threshold values.
    check("V5 deterministic: algorithm threshold attains the brute force minimum cost",
          np.isclose(Vs[threshold(pold)], Vs.min(), atol=1e-6),
          f"best theta = {int(np.argmin(Vs))}, algorithm theta = {threshold(pold)}")
    check("V5 deterministic: minimum swept cost equals V*[0]", np.isclose(Vs.min(), Vd[0], atol=1e-6))


def v6_comparative_statics():
    '''
    Sweeps of cf, ch, and K. Plots the results. 
    '''
    section("V6 COMPARATIVE STATICS")

    def solve(K_, cf_, ch_):
        a, b, c = build_model(K_, cf_, ch_, CAP)
        V, pol, _ = pi_exact(a, b, c, GAMMA)
        return V, pol

    cfs = [10, 25, 50, 100, 200, 400]
    chs = [0.5, 1, 2, 4, 8]
    Ks = [1, 5, 10, 15, 25, 50, 100, 200]

    cf_res = [solve(K, c, CH) for c in cfs]
    ch_res = [solve(K, CF, c) for c in chs]
    K_res = [solve(k, CF, CH) for k in Ks]

    cf_th = [threshold(p) for _, p in cf_res]
    ch_th = [threshold(p) for _, p in ch_res]
    K_th = [threshold(p) for _, p in K_res]

    print("cf  :", cfs, "\nthr :", cf_th, "\n V[0]:", [round(v[0], 1) for v, _ in cf_res])
    print("ch:", chs, "\nthr :", ch_th, "\nV[0]:", [round(v[0], 1) for v, _ in ch_res])
    print("K :", Ks, "\nthr :", K_th, "\nV[0]:", [round(v[0], 1) for v, _ in K_res])

    check("V6 threshold is nondecreasing in cf", np.all(np.diff(cf_th) >= 0))
    check("V6 threshold is nonincreasing in ch", np.all(np.diff(ch_th) <= 0))
    check("V6 V[0] is nondecreasing in cf", np.all(np.diff([v[0] for v, _ in cf_res]) >= -1e-9))
    check("V6 V[0] is nondecreasing in ch", np.all(np.diff([v[0] for v, _ in ch_res]) >= -1e-9))
    check("V6 V[0] is nonincreasing in K, a bigger shuttle can only help",
          np.all(np.diff([v[0] for v, _ in K_res]) <= 1e-9))

    every = cf_res + ch_res + K_res
    check("V6 V is nondecreasing in s in every sweep", all(np.all(np.diff(v) >= -1e-9) for v, _ in every))
    # Threshold structure is expected whenever the shuttle can keep up with arrivals.
    # K=1 and K=5 sit near or below the mean arrival of 3 per period, so the queue pins at the cap.
    keep_up = cf_res + ch_res + [res for k, res in zip(Ks, K_res) if k >= 10]
    check("V6 optimal policy has threshold structure whenever K >= 10", all(is_threshold_policy(p) for _, p in keep_up))
    # With K=5 waiting becomes cheaper right next to the cap, because overflow arrivals are lost
    # for free. This is a consequence of the no penalty overflow assumption, so it is documented.
    p5 = K_res[Ks.index(5)][1]
    idx5 = np.flatnonzero(~p5[threshold(p5):]) + threshold(p5)
    check("V6 K=5 departure from threshold structure is confined to states within 10 of the cap",
          len(idx5) == 0 or idx5.min() >= CAP - 10, f"waits again from state {idx5.min() if len(idx5) else None}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for a, xs, ys, lab in ((ax[0], cfs, cf_th, "cf"), (ax[1], chs, ch_th, "ch"), (ax[2], Ks, K_th, "K")):
        a.plot(xs, ys, "o-")
        a.set_xlabel(lab)
        a.set_ylabel("dispatch threshold")
        a.set_title(f"threshold vs {lab}")
    plt.tight_layout()
    plt.savefig(FIG / "v6_comparative_statics.png", dpi=150)
    plt.close()


def v7_convergence():
    '''
    Checks proper convergence behavior for each algo. For value iteration that the difference in new value functions decreases by a power of gamma. Likewise with enumeration whilst for policy iteration that the value function continuously improves. 
    '''
    section("V7 CONVERGENCE BEHAVIOR")

    # Value iteration should be a gamma contraction
    V, _, deltas, hist = vi(Q0, Q1, QR, GAMMA, tol=1e-10, keep=True)
    deltas = np.array(deltas)
    half = len(deltas) // 2
    slope = np.polyfit(np.arange(half, len(deltas)), np.log(deltas[half:]), 1)[0]
    check("V7 VI log delta slope is close to log gamma", abs(slope - np.log(GAMMA)) < 0.005,
          f"slope = {slope:.4f}, log gamma = {np.log(GAMMA):.4f}")
    err = np.array([np.abs(h - V_STAR).max() for h in hist])
    n_idx = np.arange(1, len(hist) + 1)
    bound = GAMMA ** n_idx * np.abs(V_STAR).max()  # V0 is zero so the initial error is norm of V*
    check("V7 VI error obeys the contraction bound at every iteration", (err <= bound * (1 + 1e-9) + 1e-9).all())

    # Enumeration with horizon k equals k VI sweeps from zero, so error shrinks like gamma**k
    ks = [1, 5, 10, 25, 50, 100, 500]
    errs = []
    Vk = np.zeros(N)
    for k in range(1, 501):
        Vk = np.minimum(QR[0] + GAMMA * Q0 @ Vk, QR[1] + GAMMA * Q1 @ Vk)
        if k in ks:
            errs.append(np.abs(Vk - V_STAR).max())
    check("V7 enumeration error decays like gamma**T",
          all(e <= GAMMA ** k * np.abs(V_STAR).max() * (1 + 1e-9) + 1e-9 for e, k in zip(errs, ks)))
    check("V7 T=500 truncation error is negligible, below 1e-7", errs[-1] < 1e-7, f"{errs[-1]:.2e}")
    # Cross check the user's enumeration at several horizons against the independent recursion
    for k in (1, 5, 25, 100):
        Vu, _ = enumeration(T=k)
        Vi, _ = backward(Q0, Q1, QR, GAMMA, k)
        check(f"V7 enumeration T={k} matches independent backward induction", np.allclose(Vu, Vi))

    # Policy iteration, outer loop count and monotone improvement for a cost problem
    _, _, pi_hist = pi_exact(Q0, Q1, QR, GAMMA)
    H = np.array(pi_hist)
    check("V7 PI converges in a small number of rounds", len(H) < 15, f"{len(H)} evaluation rounds")
    check("V7 V of each PI iterate is nonincreasing at every state", np.all(np.diff(H, axis=0) <= 1e-9))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(deltas, label="delta")
    ax[0].semilogy(err, label="true error")
    ax[0].semilogy(bound, "--", label="gamma^n bound")
    ax[0].set_xlabel("iteration")
    ax[0].set_title("V7: value iteration")
    ax[0].legend()
    ax[1].semilogy(ks, errs, "o-", label="error at V0")
    ax[1].semilogy(ks, [GAMMA ** k * np.abs(V_STAR).max() for k in ks], "--", label="gamma^T bound")
    ax[1].set_xlabel("horizon T")
    ax[1].set_title("V7: enumeration truncation error")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "v7_convergence.png", dpi=150)
    plt.close()


def v8_robustness():
    '''
    What happens in value iteration if we start with different value functions and what happens in policy iterations if we start from random policies. Will we still
    converge to the same output?
    '''
    section("V8 INITIALIZATION AND TOLERANCE ROBUSTNESS")

    # Value iteration from wild starting points should reach the same answer
    bound = GAMMA / (1 - GAMMA) * 1e-8
    for i, scale in enumerate((1.0, 1e2, 1e4)):
        V0 = RNG.uniform(-scale, scale, N)
        V, pol, _, _ = vi(Q0, Q1, QR, GAMMA, tol=1e-8, V0=V0)
        check(f"V8 VI from random start of scale {scale:g}",
              np.abs(V - V_STAR).max() < max(TOL_V, bound * 1.01) and np.array_equal(pol, POL_STAR),
              f"error = {np.abs(V - V_STAR).max():.2e}")

    # Policy iteration from random initial policies
    rounds = []
    ok = True
    for _ in range(10):
        p0 = RNG.random(N) < 0.5
        V, pol, h = pi_exact(Q0, Q1, QR, GAMMA, pol0=p0)
        rounds.append(len(h))
        ok &= np.allclose(V, V_STAR, atol=1e-9) and np.array_equal(pol, POL_STAR)
    check("V8 PI from 10 random initial policies reaches the same answer", ok, f"rounds used = {rounds}")

    # Tolerance sweep, the error must stay inside gamma/(1-gamma)*tol
    print("\ttol\ttrue error\tbound\tsame policy")
    for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        V, pol, _, _ = vi(Q0, Q1, QR, GAMMA, tol=tol)
        err = np.abs(V - V_STAR).max()
        b = GAMMA / (1 - GAMMA) * tol
        same = np.array_equal(pol, POL_STAR)
        print(f"{tol:<10.0e} {err:<14.3e} {b:<14.3e} {same}")
        check(f"V8 tol={tol:g}: error is within the bound", err <= b * 1.01 + 1e-9)
        if tol <= 1e-8:
            check(f"V8 tol={tol:g}: policy equals the reference", same)


def required_plots(V_e, V_v, pol_p):
    section("REQUIRED PLOTS FOR PARTS A B C")
    s = np.arange(N)
    for name, y, lab, title in (
        ("part_a_enumeration_V0.png", V_e, "V0", "Enumeration, optimal value at time 0"),
        ("part_b_value_iteration_V.png", V_v, "V", "Value iteration, optimal value"),
    ):
        plt.figure(figsize=(6, 4))
        plt.plot(s, y)
        plt.xlabel("customers waiting")
        plt.ylabel(lab)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIG / name, dpi=150)
        plt.close()
    plt.figure(figsize=(6, 4))
    plt.step(s, pol_p.astype(int), where="post")
    plt.yticks([0, 1], ["wait", "dispatch"])
    plt.xlabel("customers waiting")
    plt.title("Policy iteration, optimal policy")
    plt.tight_layout()
    plt.savefig(FIG / "part_c_policy_iteration_policy.png", dpi=150)
    plt.close()
    print("saved to ./figures")


def main():
    edge_cases()
    V_e, pol_e, V_v, pol_v, V_p, pol_p = v1_cross_method()
    v2_bellman_residual(V_v)
    v3_optimality()
    v4_monte_carlo(pol_e)
    v5_limiting_cases()
    v6_comparative_statics()
    v7_convergence()
    v8_robustness()
    required_plots(V_e, V_v, pol_p)

    failed = [n for n, ok in RESULTS if not ok]
    section("SUMMARY")
    print(f"{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
    for n in failed:
        print("  FAILED:", n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()