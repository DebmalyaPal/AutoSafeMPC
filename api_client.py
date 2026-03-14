"""
================================================================================
FILE: api_client.py
PROJECT: AutoSafeMPC
ROLE: Centralized API client and shared utilities
================================================================================

PURPOSE:
    Single source of truth for everything shared across all 3 levels:
        - Groq API client initialization
        - LLM action query (identical logic in all 3 levels)
        - Shared environment config (seed, wind, episode length, limits)
        - Wind injection utility (same disturbance profile in all 3 levels)
        - Environment factory (same env settings in all 3 levels)
        - Consistent header/results printing

    If you need to change the model, prompt, wind profile, or environment
    settings — change it HERE only. All 3 level files update automatically.

USAGE:
    from api_client import get_llm_action, make_env, inject_wind
    from api_client import ENV_CONFIG, print_header, print_results

================================================================================
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from safe_control_gym.utils.registration import make

# ── API Client ──
load_dotenv()

_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

LLM_MODEL      = "llama-3.3-70b-versatile" # "llama-3.3-70b-versatile", "openai/gpt-oss-120b"
LLM_MAX_TOKENS = 64

# ── Shared Environment Configuration ──
ENV_CONFIG = {
    "seed": 42,             # 42
    "episode_steps": 10,    # 200
    "wind_start": 2,        # 5
    "wind_end": 8,          # 8
    "wind_magnitude": 2.0,  # 0.3 m/s
    "action_limit": 1.0,    # 1.0
    # CartPole limits
    "cart_limit": 2.4,      # 2.4
    "pole_limit": 0.2,      # 0.2
    # Quadrotor 2D limits
    "x_limit": 5.0,         # 5.0
    "y_limit": 5.0,         # 5.0
    "angle_limit": 0.5,     # 0.5
    # active environment
    "env_name": "cartpole",  # default
}


# ── LLM Action Query ──
def get_llm_action(obs: np.ndarray) -> float:
    env_name = ENV_CONFIG.get("env_name", "cartpole")
    lim = ENV_CONFIG

    if env_name == "cartpole":
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        prompt = f"""You are controlling a CartPole balancing task.
            Current state:
            - Cart position:          {cart_pos:.4f}  (safe: -{lim['cart_limit']} to +{lim['cart_limit']})
            - Cart velocity:          {cart_vel:.4f}
            - Pole angle (radians):   {pole_angle:.4f}  (safe: -{lim['pole_limit']} to +{lim['pole_limit']})
            - Pole angular velocity:  {pole_vel:.4f}

            Output ONLY a single float between -1.0 and 1.0.
            Negative = push cart left. Positive = push cart right.
            No explanation — just the number.
            """
    elif env_name == "quadrotor-2d":
        x, y, x_dot, y_dot, theta, theta_dot = obs
        prompt = f"""You are controlling a 2D Quadrotor.
            Current state:
            - X position:          {x:.4f}  (safe: -{lim['x_limit']} to +{lim['x_limit']})
            - Y position:          {y:.4f}  (safe: -{lim['y_limit']} to +{lim['y_limit']})
            - X velocity:          {x_dot:.4f}
            - Y velocity:          {y_dot:.4f}
            - Angle (radians):     {theta:.4f}  (safe: -{lim['angle_limit']} to +{lim['angle_limit']})
            - Angular velocity:    {theta_dot:.4f}

            Output ONLY a single float between -1.0 and 1.0.
            Negative = push/tilt left. Positive = push/tilt right.
            No explanation — just the number.
            """
    else:
        print(f"[ERROR] Unknown environment: {env_name}. Fallback action 0.0")
        return 0.0

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.choices[0].message.content.strip()
        print("LLM raw output:", raw)
        return float(np.clip(float(raw), -lim['action_limit'], lim['action_limit']))

    except (ValueError, IndexError):
        print("  [WARN] LLM parse failed — fallback action: 0.0")
        return 0.0
    except Exception as e:
        print(f"  [ERROR] LLM API call failed: {e} — fallback action: 0.0")
        return 0.0


# ── Wind Injection ──
def inject_wind(env, step: int) -> bool:
    cfg = ENV_CONFIG
    if cfg["wind_start"] <= step < cfg["wind_end"]:
        state = list(env.state)
        env_name = cfg.get("env_name", "cartpole")

        if env_name == "cartpole":
            state[1] += cfg["wind_magnitude"]  # cart velocity
        elif env_name == "quadrotor-2d":
            state[2] += cfg["wind_magnitude"]  # x_dot
            state[3] += cfg["wind_magnitude"]  # y_dot

        env.state = np.array(state)
        return True
    return False


# ── Environment Factory ──
def make_env_cartpole():
    """
    Creates a CartPole environment using safe-control-gym's make().
    Uses ENV_CONFIG seed so all 3 levels start from the exact same state.
    NOTE: safe-control-gym's reset() returns (obs, info) — unpack accordingly.
    """
    env = make('cartpole', **{
        'task_config': {
            'task':              'stabilization',
            'done_on_violation': False,
        }
    })
    obs, _ = env.reset(seed=ENV_CONFIG["seed"])
    return env, obs
 
 
def make_env_quadrotor():
    """
    Creates a Quadrotor2D environment using safe-control-gym's make().
    Uses ENV_CONFIG seed so all 3 levels start from the exact same state.
    """
    env = make('quadrotor', **{
        'task_config': {
            'task':              'stabilization',
            'done_on_violation': False,
        }
    })
    obs, _ = env.reset(seed=ENV_CONFIG["seed"])
    return env, obs
 
 
def make_env():
    env_name = ENV_CONFIG.get("env_name", "cartpole")
    if env_name == "cartpole":
        return make_env_cartpole()
    elif env_name == "quadrotor-2d":
        return make_env_quadrotor()
    else:
        raise ValueError(f"Unknown environment: {env_name}")


# ── Logging Helpers ──
def print_header(level_name: str, extra_info: str = ""):
    cfg = ENV_CONFIG
    sep = "=" * 68
    print(sep)
    print(f"  {level_name} | Env: {cfg.get('env_name','unknown')}")
    print(f"  Seed: {cfg['seed']}  | Wind: steps {cfg['wind_start']}–{cfg['wind_end']} | Magnitude: {cfg['wind_magnitude']}")
    if extra_info:
        print(f"  {extra_info}")
    print(sep)


def print_results(total_violations: int, violation_steps: list, note: str = ""):
    sep = "=" * 68
    print(f"\n{sep}")
    print("  RESULTS")
    print(f"  Total violations : {total_violations}")
    print(f"  Violation steps  : {violation_steps if violation_steps else 'none'}")
    if note:
        print(f"  Note             : {note}")
    print(sep)