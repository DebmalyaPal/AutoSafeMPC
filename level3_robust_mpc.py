# =============================================================================
# FILE: level3_robust_mpc.py
# PROJECT: AutoSafeMPC
# Level 3: Robust MPC
# =============================================================================
#
# PURPOSE:
#   This is the FULL METHOD. The LLM acts as the high-level planner and a
#   Robust MPC safety filter wraps around it. Unlike level2, this MPC knows
#   about worst-case disturbances (wind) and tightens its safety constraints
#   accordingly -- so even if the maximum possible wind hits at every step,
#   the system is guaranteed to remain within the original safe bounds.
#
#   Architecture:
#       LLM (high-level planner)     ->  proposes action u_llm
#       Robust MPC (safety filter)   ->  solves: min ||u - u_llm||^2
#                                                s.t. tightened constraints
#                                                     worst-case disturbance W
#                                    ->  outputs robust-safe action u_safe
#
#   The key addition over level2:
#       Tightened constraint set = Original bounds - disturbance propagation
#       Example: if wind can move cart by 0.05m over horizon, constrain cart
#                to +-2.35m instead of +-2.4m so real cart stays inside +-2.4m
#
# WHAT THIS DEMONSTRATES:
#   - Robust MPC achieves the LOWEST violation count of all 3 levels
#   - Safety is guaranteed even under worst-case wind disturbance
#   - The LLM's high-level intent is still respected when robustly safe
#
# WHEN THIS WILL STILL CRASH / VIOLATE CONSTRAINTS:
#   1. DISTURBANCE EXCEEDS ASSUMED BOUND: 
#       The tightening is computed for wind up to WIND_BOUND Newtons. 
#       If WIND_FORCE > WIND_BOUND (e.g., a sudden gust much stronger 
#       than anticipated), the tightened constraints are insufficient 
#       and violations may still occur.
#
#   2. INITIAL STATE OUTSIDE TIGHTENED REGION: 
#       If the episode starts (or reaches) a state that is safe under 
#       original bounds but outside the tightened bounds, the robust 
#       MPC will immediately be infeasible.
#       Fallback is u=0.0, which may cause a violation.
#
#   3. LINEARIZATION STILL APPLIES: 
#       Same as level2 -- the A, B model is only accurate near the upright equilibrium. 
#       Severe pole angles invalidate the disturbance propagation calculation.
#
#   4. CONSERVATISM COST: 
#       Robust MPC is more conservative -- it sacrifices some performance (reward) 
#       to guarantee safety. The total reward here may be slightly lower than 
#       level2 even with fewer violations. This is expected and is the classic 
#       safety-performance tradeoff.
#
#   5. COMPUTATIONAL COST: 
#       Constraint tightening adds complexity to the QP.
#       On slow machines, solver time may increase. Still typically <100ms.
#
# ENVIRONMENT (FIXED ACROSS ALL 3 LEVELS -- DO NOT CHANGE):
#   - Task:               CartPole stabilization
#   - Episode length:     200 steps
#   - Wind disturbance:   +15.0 N lateral force on cart, steps 40-60
#   - Random seed:        42
#   - Constraint limits:  Pole angle +-0.2 rad, Cart position +-2.4 m
#   - done_on_violation:  False
#
# EXPECTED RESULTS:
#   Violations: LOW -- near zero, ideally zero during the wind phase.
#   Any remaining violations expose gaps in the disturbance model or
#   linearization errors, which are honest limitations to report.
#
# COMPARISON GUIDE:
#   This is your final result. 
#   Compare: level1 violations  >> level2 violations  >= level3 violations
#   The gap between level2 and level3 shows the value of robustness.
#   The gap between level1 and level2 shows the value of MPC filtering.
#
# =============================================================================

import numpy as np
import cvxpy as cp
from api_client import (
    get_llm_action,
    inject_wind,
    make_env,
    print_header,
    print_results,
    ENV_CONFIG,
)
 
# ── MPC Parameters ──
MPC_HORIZON = 10
DT          = 0.02
 
# ── Robust MPC: Disturbance Bound ──
# W_MAX is the ASSUMED worst-case disturbance on cart velocity per step.
# Must be ≥ ENV_CONFIG["wind_magnitude"] to guarantee safety during wind window.
# Setting W_MAX too high over-tightens constraints and causes infeasibility.
W_MAX = 0.35   # slightly above WIND_MAGNITUDE=0.30 for safety margin
 
# ── CartPole Linearized Dynamics ──
M_CART  = 1.0
M_POLE  = 0.1
M_TOTAL = M_CART + M_POLE
L_POLE  = 0.5
G       = 9.81
 
A = np.array([
    [1,  DT,  0,                          0  ],
    [0,  1,   -(M_POLE * G / M_TOTAL)*DT, 0  ],
    [0,  0,   1,                          DT ],
    [0,  0,   (G / L_POLE)*DT,            1  ]
])
B = np.array([
    [0              ],
    [DT / M_TOTAL   ],
    [0              ],
    [-DT / (M_TOTAL * L_POLE)]
])
 
# Wind disturbance enters through cart velocity (state index 1)
B_W        = np.zeros((4, 1))
B_W[1, 0]  = 1.0
 
 
def compute_tightening(horizon: int, w_max: float) -> tuple[float, float]:
    """
    Pre-compute constraint tightening amounts δ_pole and δ_cart.
 
    Calculated ONCE before the episode starts. Represents the maximum
    cumulative deviation a worst-case disturbance of w_max can cause
    over the MPC horizon, propagated through the linearized dynamics.
 
    Returns:
        (delta_pole, delta_cart): amounts to subtract from constraint limits
    """
    C_pole  = np.array([0, 0, 1, 0])   # selects pole_angle from state
    C_cart  = np.array([1, 0, 0, 0])   # selects cart_pos from state
    delta_pole = 0.0
    delta_cart = 0.0
    A_power    = np.eye(4)
 
    for _ in range(horizon):
        # delta_pole += abs(C_pole @ A_power @ B_W)[0] * w_max
        # delta_cart += abs(C_cart @ A_power @ B_W)[0] * w_max
        delta_pole += np.abs(C_pole @ A_power @ B_W)[0] * w_max
        delta_cart += np.abs(C_cart @ A_power @ B_W)[0] * w_max
        A_power     = A_power @ A
 
    return delta_pole, delta_cart
 
 
# Pre-compute tightening once — used at every step
DELTA_POLE, DELTA_CART = compute_tightening(MPC_HORIZON, W_MAX)
 
 
def robust_mpc_filter(x0: np.ndarray, u_llm: float) -> tuple[float, bool]:
    """
    Robust MPC safety filter with constraint tightening.
 
    Uses TIGHTENED constraints so the optimizer plans conservatively,
    leaving headroom for the worst-case disturbance. The disturbance
    itself does not appear explicitly in the optimization — the tightening
    absorbs it implicitly (tube MPC / constraint tightening approach).
 
    This is the ONLY structural difference from Level 2's mpc_safety_filter:
        Level 2:  constraints use  pole_limit
        Level 3:  constraints use  pole_limit - DELTA_POLE  (tightened)
 
    Args:
        x0:    current state
        u_llm: LLM's proposed action
 
    Returns:
        (safe_action, is_fallback)
    """
    lim        = ENV_CONFIG
    N          = MPC_HORIZON
    tight_pole = lim["pole_limit"]   - DELTA_POLE
    tight_cart = lim["cart_limit"]   - DELTA_CART
 
    # If over-tightened to infeasibility, fall back immediately
    if tight_pole <= 0 or tight_cart <= 0:
        return float(np.clip(u_llm, -lim["action_limit"], lim["action_limit"])), True
 
    U = cp.Variable((N, 1))
    X = cp.Variable((N + 1, 4))
 
    objective   = cp.Minimize(cp.sum_squares(U[0] - u_llm))
    constraints = [X[0] == x0]
 
    for t in range(N):
        constraints += [
            X[t + 1] == A @ X[t] + B @ U[t],     # nominal dynamics
            cp.abs(X[t + 1][2]) <= tight_pole,   # TIGHTENED pole constraint
            cp.abs(X[t + 1][0]) <= tight_cart,   # TIGHTENED cart constraint
            cp.abs(U[t])         <= lim["action_limit"],
        ]
 
    problem = cp.Problem(objective, constraints)
 
    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if problem.status in ["optimal", "optimal_inaccurate"] and U.value is not None:
            return float(np.clip(U.value[0, 0], -lim["action_limit"], lim["action_limit"])), False
    except Exception as e:
        print(f"  [WARN] Robust MPC solver error: {e}")
 
    return float(np.clip(u_llm, -lim["action_limit"], lim["action_limit"])), True
 
 
def run():
    env, obs = make_env()
    total_violations = 0
    violation_steps  = []
    fallback_count   = 0
 
    print_header(
        "LEVEL 3: LLM + Robust MPC Safety Filter  (full method)",
        extra_info=(
            f"Horizon: {MPC_HORIZON}  |  W_MAX: {W_MAX}  |  "
            f"δ_pole: {DELTA_POLE:.4f}  |  δ_cart: {DELTA_CART:.4f}  |  "
            f"Effective pole limit: ±{ENV_CONFIG['pole_limit'] - DELTA_POLE:.4f} rad"
        )
    )
 
    for step in range(ENV_CONFIG["episode_steps"]):
        wind_active = inject_wind(env, step)
 
        u_llm               = get_llm_action(obs)            # Step 1: LLM proposes
        u_safe, is_fallback = robust_mpc_filter(obs, u_llm)  # Step 2: Robust MPC corrects
 
        if is_fallback:
            fallback_count += 1
 
        # obs, reward, done, _, info = env.step(np.array([u_safe]))
        obs, reward, done, info = env.step(np.array([u_safe]))
 
        violation = int(info.get('constraint_violation', 0))
        total_violations += violation
        if violation:
            violation_steps.append(step)
 
        wind_tag   = " <<<WIND>>>" if wind_active else ""
        fb_tag     = " [FB]" if is_fallback else ""
        correction = abs(u_safe - u_llm)
        print(f"Step {step:3d}{wind_tag:11s} | LLM: {u_llm:+.3f} → RMPC: {u_safe:+.3f} "
              f"(Δ{correction:.3f}){fb_tag:5s} | Pole: {obs[2]:+.4f} | Violation: {violation}")
 
        if done:
            print(f"\n  [Episode ended early at step {step}]")
            break
 
    env.close()
    print_results(
        total_violations,
        violation_steps,
        note=f"Fallback steps (MPC infeasible): {fallback_count}. "
             "Should be the LOWEST count across all 3 levels."
    )
 
 
if __name__ == "__main__":
    run()
 