"""
val_common.py

Helpers shared by validation_cartpole.py and validation_pong.py. Every check prints PASS or FAIL and is recorded for the summary.
The gradient checks work on any of the four models. A model may return logits (no baseline) or a tuple (logits, values) (shared trunk baseline); logits_of hides the difference.
"""

import sys
import copy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

RESULTS = []

def check(name, ok, detail=""):
    """Record and print one PASS or FAIL line."""
    RESULTS.append((name, bool(ok)))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"\t{detail}" if detail else ""))

def section(title):
    print("\n" + "#" * 10 + f"\n{title}\n" + "#" * 10)

def finish():
    failed = [n for n, ok in RESULTS if not ok]
    section("SUMMARY")
    print(f"{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
    for n in failed:
        print("  FAILED:", n)
    sys.exit(1 if failed else 0)

def logits_of(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out

def flat_grad(model, loss):
    """Gradient of loss with respect to every parameter as one vector, zeros for unused parameters."""
    model.zero_grad()
    loss.backward(retain_graph=True)
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in model.parameters()]).detach().clone()

def _logp(model, S, A):
    la = torch.log_softmax(logits_of(model, S), -1)
    return la, la.gather(-1, A[:, None]).squeeze(-1)

def generic_gradient_checks(tag, model, states, actions, returns, chunk=64):
    """
    Model level checks of the policy gradient estimator, run in double precision on the CPU.
    G1 the expected score function is zero, which is why subtracting any state dependent baseline cannot bias the gradient
    G2 the baseline term b(s) * grad log pi has zero expectation for an arbitrary b
    G3 one summed loss gives the same gradient as adding up the gradients of single steps
    G4 splitting the batch into chunks gives the same gradient as one pass
    G5 with all returns zero there is no gradient and Adam leaves the weights unchanged
    G6 the autograd gradient matches a finite difference of the loss
    """
    pol = copy.deepcopy(model).cpu().double()
    S = states.detach().cpu().double()
    A = actions.detach().cpu().long()
    G = returns.detach().cpu().double()
    K = len(S)

    la, lp = _logp(pol, S, A)
    probs = la.exp().detach()
    g_taken = flat_grad(pol, lp.sum())

    g_exp = flat_grad(pol, (probs * la).sum())
    r1 = (g_exp.norm() / g_taken.norm()).item()
    check(f"{tag} G1 expected score function is zero", r1 < 1e-8, f"relative norm = {r1:.1e}")

    b = torch.randn(K, dtype=torch.double) * 5
    g_ref = flat_grad(pol, (b * lp).sum())
    g_b = flat_grad(pol, (b * (probs * la).sum(-1)).sum())
    r2 = (g_b.norm() / g_ref.norm()).item()
    check(f"{tag} G2 a state dependent baseline term has zero expectation", r2 < 1e-8, f"relative norm = {r2:.1e}")

    M = min(32, K)
    _, lp_m = _logp(pol, S[:M], A[:M])
    g_sum = flat_grad(pol, (lp_m * G[:M]).sum())
    g_loop = torch.zeros_like(g_sum)
    for i in range(M):
        _, l = _logp(pol, S[i:i + 1], A[i:i + 1])
        g_loop += G[i] * flat_grad(pol, l.sum())
    r3 = ((g_sum - g_loop).norm() / g_sum.norm()).item()
    check(f"{tag} G3 summed loss equals the sum of single step gradients", r3 < 1e-9, f"relative diff = {r3:.1e}")

    full = flat_grad(pol, -(lp * G).sum())
    acc = torch.zeros_like(full)
    for i in range(0, K, chunk):
        _, l = _logp(pol, S[i:i + chunk], A[i:i + chunk])
        acc += flat_grad(pol, -(l * G[i:i + chunk]).sum())
    r4 = ((full - acc).norm() / full.norm()).item()
    check(f"{tag} G4 chunked backward equals one pass", r4 < 1e-9, f"relative diff = {r4:.1e}")

    p2 = copy.deepcopy(pol)
    opt = torch.optim.Adam(p2.parameters(), lr=0.01)
    before = [p.detach().clone() for p in p2.parameters()]
    _, l = _logp(p2, S, A)
    opt.zero_grad()
    (-(l * torch.zeros_like(G)).sum()).backward()
    opt.step()
    moved = max((p - q).abs().max().item() for p, q in zip(p2.parameters(), before))
    check(f"{tag} G5 zero returns give zero update", moved == 0.0, f"max weight change = {moved:.1e}")

    _, lp_f = _logp(pol, S, A)
    g = flat_grad(pol, -(lp_f * G).sum())
    u = g / g.norm()
    theta = parameters_to_vector(pol.parameters()).detach().clone()
    eps = 1e-6 # -6 for tighter 

    def loss_at(vec):
        vector_to_parameters(vec, pol.parameters())
        with torch.no_grad():
            _, l_ = _logp(pol, S, A)
            return -(l_ * G).sum().item()

    fd = (loss_at(theta + eps * u) - loss_at(theta - eps * u)) / (2 * eps)
    vector_to_parameters(theta, pol.parameters())
    r6 = abs(fd - g.norm().item()) / g.norm().item()
    check(f"{tag} G6 autograd gradient matches a finite difference", r6 < 5e-4, f"relative error = {r6:.1e}")

def baseline_model_checks(tag, model, states, actions, returns):
    """
    Checks for the shared trunk actor critic models, which return (logits, values) and have .actor, .critic and .shared.
    B1 the advantage carries no gradient
    B2 the actor loss trains the actor head and the trunk but not the critic head
    B3 the critic loss trains the critic head and the trunk but not the actor head
    B4 a small step along the critic gradient lowers the critic loss
    The cosine between the two gradients on the shared trunk is printed, not checked.
    """
    m = copy.deepcopy(model).cpu().double()
    S = states.detach().cpu().double()
    A = actions.detach().cpu().long()
    G = returns.detach().cpu().double()

    def forward():
        logits, values = m(S)
        lp = torch.log_softmax(logits, -1).gather(-1, A[:, None]).squeeze(-1)
        return lp, values, G - values.detach()

    def zero_or_none(t):
        return t is None or t.abs().max().item() == 0.0

    def nonzero(t):
        return t is not None and t.abs().max().item() > 0.0

    lp, values, adv = forward()
    check(f"{tag} B1 advantage carries no gradient", not adv.requires_grad)

    m.zero_grad()
    (-(lp * adv).sum()).backward(retain_graph=True)
    check(f"{tag} B2 actor loss does not train the critic head", zero_or_none(m.critic.weight.grad) and zero_or_none(m.critic.bias.grad))
    check(f"{tag} B2 actor loss trains the actor head", nonzero(m.actor.weight.grad))
    g_actor = torch.cat([p.grad.flatten() for p in m.shared.parameters()]).clone()

    m.zero_grad()
    critic_loss = 0.5 * (values - G).pow(2).sum()
    critic_loss.backward(retain_graph=True)
    check(f"{tag} B3 critic loss does not train the actor head", zero_or_none(m.actor.weight.grad) and zero_or_none(m.actor.bias.grad))
    check(f"{tag} B3 critic loss trains the critic head", nonzero(m.critic.weight.grad))
    g_critic = torch.cat([p.grad.flatten() for p in m.shared.parameters()]).clone()
    cos = (g_actor @ g_critic / (g_actor.norm() * g_critic.norm())).item()
    print(f"\tnote: cosine between actor and critic gradients on the shared trunk = {cos:.3f}")

    gc = flat_grad(m, critic_loss)
    theta = parameters_to_vector(m.parameters()).detach().clone()
    vector_to_parameters(theta - 1e-5 * gc / gc.norm(), m.parameters())
    with torch.no_grad():
        new_loss = 0.5 * (m(S)[1] - G).pow(2).sum().item()
    check(f"{tag} B4 a step along the critic gradient lowers the critic loss", new_loss < critic_loss.item(), f"{critic_loss.item():.6f} to {new_loss:.6f}")