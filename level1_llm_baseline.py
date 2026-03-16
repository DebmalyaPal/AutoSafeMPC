# =============================================================================
# FILE: level1_llm_baseline.py
# PROJECT: AutoSafeMPC
# Level 1: LLM Baseline
# =============================================================================

import numpy as np
from api_client import (
    get_llm_action,
    inject_wind,
    make_env,
    print_header,
    print_results,
    ENV_CONFIG,
)


def run():
    env, obs = make_env()

    total_violations = 0
    violation_steps = []

    print_header("LEVEL 1: LLM-Only Baseline (no safety filter)")

    env_name = ENV_CONFIG.get("env_name", "cartpole")

    for step in range(ENV_CONFIG["episode_steps"]):

        wind_active = inject_wind(env, step)

        # LLM proposes and applies action directly
        action = get_llm_action(obs)

        step_result = env.step(np.array([action]))

        # Handle different env return formats
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        # ─────────────────────────────────────
        # Violation Check
        # ─────────────────────────────────────

        if env_name == "cartpole":

            cart_pos = obs[0]
            pole_ang = obs[2]

            violation = int(
                abs(cart_pos) > ENV_CONFIG["cart_limit"]
                or abs(pole_ang) > ENV_CONFIG["pole_limit"]
            )

            state_str = f"Pole: {obs[2]:+.4f} rad"

        elif env_name == "quadrotor-2d":

            x = obs[0]
            y = obs[1]
            angle = obs[4]

            violation = int(
                abs(x) > ENV_CONFIG["x_limit"]
                or abs(y) > ENV_CONFIG["y_limit"]
                or abs(angle) > ENV_CONFIG["angle_limit"]
            )

            state_str = f"X: {x:+.4f}, Y: {y:+.4f}, Angle: {angle:+.4f} rad"

        else:
            violation = 0
            state_str = str(obs)

        total_violations += violation

        if violation:
            violation_steps.append(step)

        wind_tag = " <<<WIND>>>" if wind_active else ""

        print(
            f"Step {step:3d}{wind_tag:11s} | Action: {action:+.3f} | "
            f"{state_str} | Violation: {violation}"
        )

        if done:
            print(f"\n[Episode ended early at step {step}]")
            break

    env.close()

    print_results(
        total_violations,
        violation_steps,
        note="Compare this count to Level 2 and Level 3",
    )


if __name__ == "__main__":
    run()