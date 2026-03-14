"""
================================================================================
FILE: run_experiment.py
PROJECT: AutoSafeMPC
ROLE: Run experiments for different control levels
================================================================================

PURPOSE:
    Run experiments for different control levels
        - LLM baseline
        - Classical MPC
        - Robust MPC

USAGE:
    python run_experiment.py --level <level-num> --env <env-name>
    <level-num> can be 1, 2, or 3
    <env-name> can be cartpole or quadrotor-2d

    Example:
    python run_experiment.py --level 1 --env cartpole
    python run_experiment.py --level 2 --env quadrotor-2d
    python run_experiment.py --level 3 --env quadrotor-2d

================================================================================
"""



import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Run Safe Control Gym experiments")
    parser.add_argument("--level", choices=["1", "2", "3"], required=True,
                        help="Which control level to run")
    parser.add_argument("--env", choices=["cartpole", "quadrotor-2d"], default="cartpole",
                        help="Environment to use")
    args = parser.parse_args()

    # Update api_client ENV_CONFIG dynamically
    import api_client
    if args.env == "cartpole":
        api_client.make_env = lambda: api_client.make_env_cartpole()
        api_client.ENV_CONFIG.update({
            "env_name": "cartpole",
            "cart_limit": 2.4,
            "pole_limit": 0.2095,
            "action_limit": 1.0,
            "episode_steps": 50,
            "wind_magnitude": 0.3,
            "wind_start": 5,
            "wind_end": 8,
        })
    elif args.env == "quadrotor-2d":
        api_client.make_env = lambda: api_client.make_env_quadrotor()
        api_client.ENV_CONFIG.update({
            "env_name": "quadrotor-2d",
            "x_limit": 5.0,
            "y_limit": 5.0,
            "angle_limit": 0.5,
            "action_limit": 1.0,
            "episode_steps": 50,
            "wind_magnitude": 0.3,
            "wind_start": 10,
            "wind_end": 30,
        })

    # Dynamically import the right level file
    level_map = {
        "1": "level1_llm_baseline",
        "2": "level2_classical_mpc",
        "3": "level3_robust_mpc",
    }
    module_name = level_map[args.level]
    level_module = importlib.import_module(module_name)

    # Run the experiment
    level_module.run()


if __name__ == "__main__":
    main()