# =============================================================================
# FILE: level1_llm_baseline.py
# PROJECT: AutoSafeMPC
# Level 1: LLM Baseline
# =============================================================================
#
# PURPOSE:
#   This is the BASELINE experiment. A large language model acts as
#   the sole controller for a dynamic system (CartPole or Quadrotor 2D).
#   There is NO safety filter, NO constraint enforcement, and NO fallback mechanism.
#   The LLM reads the current physical state and outputs a raw action.
#
# WHAT THIS DEMONSTRATES:
#   - How well an LLM understands control tasks through prompt engineering alone
#   - The violation count here is your UPPER BOUND (worst case)
#   - Results from this file justify why a safety layer is needed at all
#
# WHEN THIS WILL CRASH / VIOLATE CONSTRAINTS:
#   1. LLM LATENCY LAG:
#       Each step calls the API (~0.5-2s per step). Fast dynamics may
#       move the system before the action is applied.
#
#   2. PROMPT HALLUCINATION:
#       The LLM may output physically invalid actions (outside allowed range)
#       or fail to respond correctly. The fallback is action=0.0, which may
#       cause constraint violations.
#
#   3. NO INTERNAL MODEL OF DYNAMICS:
#       The LLM does not model system physics, so it guesses actions from
#       text patterns rather than solving differential equations.
#
#   4. WIND DISTURBANCE:
#       External disturbances are applied during steps defined by ENV_CONFIG
#       (`wind_start` to `wind_end`) with magnitude `wind_magnitude`.
#       The LLM cannot anticipate them, only react one step later.
#
#   5. ACCUMULATED ERROR:
#       Unlike MPC, which plans multiple steps ahead, the LLM only reacts
#       to the current state. Small errors can compound until constraints
#       are violated.
#
# ENVIRONMENT CONFIG (from ENV_CONFIG):
#   - Task:               CartPole stabilization (or Quadrotor 2D)
#   - Episode length:     ENV_CONFIG["episode_steps"] steps
#   - Wind disturbance:   magnitude ENV_CONFIG["wind_magnitude"], 
#                         steps ENV_CONFIG["wind_start"]–ENV_CONFIG["wind_end"]
#   - Random seed:        ENV_CONFIG["seed"]
#   - Constraint limits:  environment-specific (cart_limit/pole_limit/x_limit/y_limit/angle_limit)
#   - done_on_violation:  False (episode continues to count all violations)
#
# EXPECTED RESULTS:
#   Violations: HIGH for LLM-only baseline.
#   The system may survive early steps but is likely to fail during wind
#   or high-speed dynamics due to lack of planning and constraint enforcement.
#
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
    violation_steps  = []
 
    print_header("LEVEL 1: LLM-Only Baseline  (no safety filter)")

    env_name = ENV_CONFIG.get("env_name", "cartpole")
 
    for step in range(ENV_CONFIG["episode_steps"]):
        wind_active = inject_wind(env, step)
 
        # LLM proposes AND applies the action — nothing in between
        action = get_llm_action(obs)
        # obs, reward, done, _, info = env.step(np.array([action]))
        obs, reward, done, info = env.step(np.array([action]))
 
        violation = int(info.get('constraint_violation', 0))
        total_violations += violation
        if violation:
            violation_steps.append(step)
 
        wind_tag = " <<<WIND>>>" if wind_active else ""
        
        # Create environment-specific state string for printing
        if env_name == "cartpole":
            state_str = f"Pole: {obs[2]:+.4f} rad"
        elif env_name == "quadrotor-2d":
            state_str = f"X: {obs[0]:+.4f}, Y: {obs[1]:+.4f}, Angle: {obs[4]:+.4f} rad"
        else:
            state_str = str(obs)  # fallback

        print(f"Step {step:3d}{wind_tag:11s} | Action: {action:+.3f} | {state_str} | Violation: {violation}")
 
        if done:
            print(f"\n  [Episode ended early at step {step}]")
            break
 
    env.close()
    print_results(total_violations, violation_steps,
                  note="Compare this count to Level 2 and Level 3")
 
 
if __name__ == "__main__":
    run()