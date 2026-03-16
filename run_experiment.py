"""
================================================================================
FILE: run_experiment.py
PROJECT: AutoSafeMPC
ROLE: Run experiments for different control levels
================================================================================
"""

import argparse
import importlib
import api_client


def configure_environment(env_name):

    # In run_experiment.py, update these lines:

    if env_name == "cartpole":
        # ... other config ...
        api_client.ENV_CONFIG.update({
            "env_name": "cartpole",
            "seed": 42, 
            "cart_limit": 2.4,
            "pole_limit": 0.2095,
            "action_limit": 1.0,
            "episode_steps": 200,
            "wind_start": 40,
            "wind_end": 80,
            "wind_magnitude": 0.4, # <--- CHANGE THIS FROM 2.0 TO 0.4
        })

    elif env_name == "quadrotor-2d":
        # ... other config ...
        api_client.ENV_CONFIG.update({
            "env_name": "quadrotor-2d",
            "seed": 42, 
            "x_limit": 5.0,
            "y_limit": 5.0,
            "angle_limit": 0.5,
            "action_limit": 1.0,
            "episode_steps": 200,
            "wind_start": 40,
            "wind_end": 80,
            "wind_magnitude": 0.4, # <--- CHANGE THIS FROM 2.0 TO 0.4
        })
        


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--level", choices=["1", "2", "3"], required=True)
    parser.add_argument("--env", choices=["cartpole", "quadrotor-2d"], default="cartpole")

    args = parser.parse_args()

    configure_environment(args.env)

    level_map = {
        "1": "level1_llm_baseline",
        "2": "level2_classical_mpc",
        "3": "level3_robust_mpc",
    }

    module = importlib.import_module(level_map[args.level])

    module.run()


if __name__ == "__main__":
    main()