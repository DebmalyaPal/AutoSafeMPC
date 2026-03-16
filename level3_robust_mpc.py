# =============================================================================
# FILE: level3_robust_mpc.py
# PROJECT: AutoSafeMPC
# LEVEL 3: LLM + Robust MPC Safety Filter
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

# -----------------------------------------------------------------------------
# Dynamics Constants
# -----------------------------------------------------------------------------

MPC_HORIZON = 8
DT = 0.02

M_CART = 1.0
M_POLE = 0.1
L_POLE = 0.5
G = 9.81
M_TOTAL = M_CART + M_POLE

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
    [-DT / (M_TOTAL * L_POLE)]
])


# -----------------------------------------------------------------------------
# Robust MPC Safety Filter
# -----------------------------------------------------------------------------

def mpc_safety_filter_robust(x0: np.ndarray, u_llm: float) -> float:

    x0 = np.array(x0).flatten()[:4]

    N = MPC_HORIZON
    lim = ENV_CONFIG

    U = cp.Variable((N, 1))
    X = cp.Variable((N + 1, 4))
    slack = cp.Variable((N + 1, 1), nonneg=True)

    Q = np.diag([10, 1, 500, 10])
    R = 0.01

    LLM_WEIGHT = 0.2
    SLACK_PENALTY = 1e6

    cost = 0

    for t in range(N):

        cost += cp.quad_form(X[t], Q)
        cost += R * cp.sum_squares(U[t])
        cost += LLM_WEIGHT * cp.sum_squares(U[t] - u_llm)
        cost += SLACK_PENALTY * cp.sum_squares(slack[t])

    constraints = [X[0] == x0]

    for t in range(N):

        constraints += [

            X[t + 1] == A @ X[t] + B @ U[t],

            cp.abs(X[t + 1][2]) <= (lim["pole_limit"] - 0.05) + slack[t+1],
            cp.abs(X[t + 1][0]) <= (lim["cart_limit"] - 0.30) + slack[t+1],

            cp.abs(U[t]) <= lim["action_limit"],
        ]

    prob = cp.Problem(cp.Minimize(cost), constraints)

    try:

        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            return float(U.value[0, 0])

    except:
        pass

    return float(np.clip(u_llm, -1, 1))


# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------

def run():

    env, initial_obs = make_env()

    # handle reset tuple
    if isinstance(initial_obs, tuple):
        obs = initial_obs[0]
    else:
        obs = initial_obs

    total_violations = 0
    violation_steps = []

    print_header("LEVEL 3: LLM + Robust MPC (Final)")

    for step in range(ENV_CONFIG["episode_steps"]):

        wind_active = inject_wind(env, step)

        u_llm = get_llm_action(obs)

        u_safe = mpc_safety_filter_robust(obs, u_llm)

        # ---------------------------------------------------------
        # ENV STEP (Safe-Control-Gym uses 4 return values)
        # ---------------------------------------------------------

        obs, reward, done, info = env.step(np.array([u_safe]))

        # ---------------------------------------------------------
        # SAFETY VIOLATION CHECK
        # ---------------------------------------------------------

        violation = int(
            abs(obs[0]) > ENV_CONFIG["cart_limit"] or
            abs(obs[2]) > ENV_CONFIG["pole_limit"]
        )

        if violation:
            total_violations += 1
            violation_steps.append(step)

        # ---------------------------------------------------------
        # LOGGING
        # ---------------------------------------------------------

        if step % 10 == 0 or wind_active:

            tag = " <<<WIND>>>" if wind_active else ""

            print(
                f"Step {step:3d}{tag:11s} | "
                f"LLM:{u_llm:+.3f} -> RMPC:{u_safe:+.3f} | "
                f"Pole:{obs[2]:+.4f} | V:{violation}"
            )

        # ---------------------------------------------------------
        # RESET ENV IF DONE (DO NOT STOP EXPERIMENT)
        # ---------------------------------------------------------

        if done:

            print(f"[Info] Environment reset at step {step}")

            reset_out = env.reset()

            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out

    env.close()

    print_results(total_violations, violation_steps)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run()