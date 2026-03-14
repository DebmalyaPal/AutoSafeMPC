# =============================================================================
# FILE: level2_classical_mpc.py
# PROJECT: AutoSafeMPC
# Level 2: Classical MPC
# =============================================================================
#
# PURPOSE:
#   This is the CLASSICAL MPC experiment. The LLM still acts as the high-level
#   planner (proposing an action each step), but a Model Predictive Controller
#   acts as a safety filter -- it accepts the LLM's suggestion only if it is
#   safe, and overrides it with the nearest safe action if it is not.
#
#   Architecture:
#       LLM (high-level planner)  ->  proposes action u_llm
#       MPC (safety filter)       ->  solves: min ||u - u_llm||^2
#                                             s.t. dynamics + constraints
#                                 ->  outputs safe action u_safe
#
# WHAT THIS DEMONSTRATES:
#   - MPC significantly reduces constraint violations vs. the LLM alone
#   - The LLM's intent is preserved when safe; minimally corrected when not
#   - This is the "middle ground" between unconstrained LLM (level1) and
#     fully robust MPC with disturbance handling (level3)
#
# WHEN THIS WILL STILL CRASH / VIOLATE CONSTRAINTS:
#   1. NOMINAL DYNAMICS ONLY: 
#       The MPC uses a linearized model of CartPole (matrices A, B). 
#       This model assumes NO disturbances. When wind hits (steps 40-60), 
#       the model is wrong -- it predicts the cart will stay put, 
#       but in reality it is being pushed sideways. Because there is no
#       constraint tightening (that comes in level3), the wind can still push
#       the system past the safe boundary.
#
#   2. LINEARIZATION ERROR: 
#       The A, B matrices are only valid near the upright equilibrium. 
#       If the pole falls far (|angle| > ~0.3 rad), the linear
#       model diverges from reality, MPC predictions become inaccurate, and
#       the safety guarantee breaks down.
#
#   3. HORIZON TOO SHORT: 
#       With N=5 steps (0.1s lookahead), the MPC cannot
#       see far enough ahead to avoid fast-developing violations. A large LLM
#       error AND a wind gust in the same window may still cause a breach.
#
#   4. SOLVER INFEASIBILITY: 
#       If the current state is already near or outside the constraint boundary, 
#       CVXPY may find the problem infeasible (no u exists that keeps all future 
#       states inside constraints). The fallback is u=0.0, which is often the 
#       worst possible action at that moment.
#
# ENVIRONMENT (FIXED ACROSS ALL 3 LEVELS -- For Comparison):
#   - Task:               CartPole stabilization
#   - Episode length:     200 steps
#   - Wind disturbance:   +15.0 N lateral force on cart, steps 40-60
#   - Random seed:        42
#   - Constraint limits:  Pole angle +-0.2 rad, Cart position +-2.4 m
#   - done_on_violation:  False
#
# EXPECTED RESULTS:
#   Violations: MEDIUM -- significantly fewer than level1, but non-zero
#   because the MPC has no model of the wind disturbance.
#   Most remaining violations will occur during or just after the wind phase.
#
# COMPARISON GUIDE:
#   Compare total_violations here vs. level1 to confirm MPC filtering works.
#   -> Remaining violations during wind phase (steps 40-60) motivate level3.
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
 
# ── MPC Parameters ──────────────────────────────────────────────────────────────
MPC_HORIZON = 10      # steps to look ahead
DT          = 0.02    # seconds per timestep (safe-control-gym default)
 
# ── CartPole Linearized Dynamics (valid near upright equilibrium) ───────────────
# State:  x = [cart_pos, cart_vel, pole_angle, pole_angular_velocity]
# Input:  u = scalar force in [-1.0, 1.0]
# Only accurate for small pole angles (|theta| < ~0.3 rad).
M_CART  = 1.0    # cart mass (kg)
M_POLE  = 0.1    # pole mass (kg)
M_TOTAL = M_CART + M_POLE
L_POLE  = 0.5    # pole half-length (m)
G       = 9.81   # gravity (m/s²)
 
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
 
 
def mpc_safety_filter(x0: np.ndarray, u_llm: float) -> float:
    """
    Classical MPC safety filter.
 
    Finds the action closest to u_llm that keeps the system within
    constraint bounds over the next MPC_HORIZON steps, using the NOMINAL
    dynamics model (no disturbance term — that's the Level 2 limitation).
 
    Args:
        x0:    current state [cart_pos, cart_vel, pole_angle, pole_vel]
        u_llm: LLM's proposed action (may be unsafe)
 
    Returns:
        Corrected safe action. Falls back to clipped u_llm if solver fails.
    """
    N  = MPC_HORIZON
    lim = ENV_CONFIG
 
    U = cp.Variable((N, 1))
    X = cp.Variable((N + 1, 4))
 
    objective   = cp.Minimize(cp.sum_squares(U[0] - u_llm))
    constraints = [X[0] == x0]
 
    for t in range(N):
        constraints += [
            X[t + 1] == A @ X[t] + B @ U[t],           # nominal dynamics
            cp.abs(X[t + 1][2]) <= lim["pole_limit"],   # pole angle
            cp.abs(X[t + 1][0]) <= lim["cart_limit"],   # cart position
            cp.abs(U[t])         <= lim["action_limit"], # control effort
        ]
 
    problem = cp.Problem(objective, constraints)
 
    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if problem.status in ["optimal", "optimal_inaccurate"] and U.value is not None:
            return float(np.clip(U.value[0, 0], -lim["action_limit"], lim["action_limit"]))
    except Exception as e:
        print(f"  [WARN] MPC solver error: {e}")
 
    # Fallback: clip LLM action to action bounds and use directly
    return float(np.clip(u_llm, -lim["action_limit"], lim["action_limit"]))
 
 
def run():
    env, obs = make_env()
    total_violations = 0
    violation_steps  = []
 
    print_header(
        "LEVEL 2: LLM + Classical MPC Safety Filter",
        extra_info=f"Horizon: {MPC_HORIZON} steps  |  dt: {DT}s"
    )
 
    for step in range(ENV_CONFIG["episode_steps"]):
        wind_active = inject_wind(env, step)
 
        u_llm  = get_llm_action(obs)              # Step 1: LLM proposes
        u_safe = mpc_safety_filter(obs, u_llm)    # Step 2: MPC corrects

        obs, reward, done, info = env.step(np.array([u_safe]))
 
        violation = int(info.get('constraint_violation', 0))
        total_violations += violation
        if violation:
            violation_steps.append(step)
 
        wind_tag   = " <<<WIND>>>" if wind_active else ""
        correction = abs(u_safe - u_llm)
        print(f"Step {step:3d}{wind_tag:11s} | LLM: {u_llm:+.3f} → MPC: {u_safe:+.3f} "
              f"(Δ{correction:.3f}) | Pole: {obs[2]:+.4f} | Violation: {violation}")
 
        if done:
            print(f"\n  [Episode ended early at step {step}]")
            break
 
    env.close()
    print_results(total_violations, violation_steps,
                  note="Should be lower than Level 1; compare to Level 3")
 
 
if __name__ == "__main__":
    run()
 