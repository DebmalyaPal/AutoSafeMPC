# =============================================================================
# FILE: level2_classical_mpc.py
# PROJECT: AutoSafeMPC
# Level 2: LLM + Classical MPC Safety Filter
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

# ─────────────────────────────────────────────────────────────────────────────
# MPC PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

MPC_HORIZON = 8     # slightly longer horizon (better than 5)
DT = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# CartPole Linearized Dynamics
# ─────────────────────────────────────────────────────────────────────────────

M_CART = 1.0
M_POLE = 0.1
M_TOTAL = M_CART + M_POLE
L_POLE = 0.5
G = 9.81

A = np.array([
    [1, DT, 0, 0],
    [0, 1, -(M_POLE * G / M_TOTAL) * DT, 0],
    [0, 0, 1, DT],
    [0, 0, (G / L_POLE) * DT, 1],
])

B = np.array([
    [0],
    [DT / M_TOTAL],
    [0],
    [-DT / (M_TOTAL * L_POLE)],
])


# ─────────────────────────────────────────────────────────────────────────────
# MPC SAFETY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def mpc_safety_filter(x0: np.ndarray, u_llm: float) -> float:

    N = MPC_HORIZON
    lim = ENV_CONFIG

    U = cp.Variable((N, 1))
    X = cp.Variable((N + 1, 4))

    # State stabilization weights
    Q = np.diag([10, 1, 200, 5])   # strong pole penalty
    R = 0.1
    LLM_WEIGHT = 0.5

    cost = 0

    for t in range(N):
        # Dynamics
        cost += cp.quad_form(X[t], Q)
        cost += R * cp.sum_squares(U[t])
        cost += LLM_WEIGHT * cp.sum_squares(U[t] - u_llm)

    objective = cp.Minimize(cost)

    constraints = [X[0] == x0]

    for t in range(N):
        constraints += [
            X[t + 1] == A @ X[t] + B @ U[t],

            # Safety tightening
            cp.abs(X[t + 1][2]) <= 0.90 * lim["pole_limit"],
            cp.abs(X[t + 1][0]) <= 0.90 * lim["cart_limit"],

            cp.abs(U[t]) <= lim["action_limit"],
        ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"] and U.value is not None:
            return float(
                np.clip(U.value[0, 0], -lim["action_limit"], lim["action_limit"])
            )

    except Exception as e:
        print(f"[WARN] MPC solver error: {e}")

    # Fallback to LLM if solver fails
    return float(np.clip(u_llm, -lim["action_limit"], lim["action_limit"]))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run():

    env, obs = make_env()

    total_violations = 0
    violation_steps = []

    print_header(
        "LEVEL 2: LLM + Classical MPC Safety Filter",
        extra_info=f"Horizon: {MPC_HORIZON} steps | dt: {DT}s",
    )

    for step in range(ENV_CONFIG["episode_steps"]):

        wind_active = inject_wind(env, step)

        # LLM action
        u_llm = get_llm_action(obs)

        # MPC correction
        u_safe = mpc_safety_filter(obs, u_llm)

        obs, reward, done, info = env.step(np.array([u_safe]))

        # Manual violation check
        cart_pos = obs[0]
        pole_ang = obs[2]

        violation = int(
            abs(cart_pos) > ENV_CONFIG["cart_limit"]
            or abs(pole_ang) > ENV_CONFIG["pole_limit"]
        )

        total_violations += violation

        if violation:
            violation_steps.append(step)

        wind_tag = " <<<WIND>>>" if wind_active else ""
        correction = abs(u_safe - u_llm)

        print(
            f"Step {step:3d}{wind_tag:11s} | "
            f"LLM: {u_llm:+.3f} -> MPC: {u_safe:+.3f} "
            f"(corr {correction:.3f}) | Pole: {obs[2]:+.4f} | Violation: {violation}"
        )

        if done:
            print(f"\n[Episode ended early at step {step}]")
            break

    env.close()

    print_results(
        total_violations,
        violation_steps,
        note="Should be lower than Level 1; compare to Level 3",
    )


if __name__ == "__main__":
    run()