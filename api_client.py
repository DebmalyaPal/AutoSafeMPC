"""
================================================================================
FILE: api_client.py
PROJECT: AutoSafeMPC
ROLE: Centralized API client and shared utilities
================================================================================
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from safe_control_gym.utils.registration import make

# ─────────────────────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────────────────────

load_dotenv()

_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=30.0,
    base_url="https://api.groq.com/openai/v1"
)

LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 64

# Call LLM every N steps (reduces API usage)
LLM_CALL_INTERVAL = 5

_last_action = 0.0
_step_counter = 0


# ─────────────────────────────────────────────────────────────
# SHARED EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────

ENV_CONFIG = {
    "seed": 42,

    "episode_steps": 200,

    "wind_start": 40,
    "wind_end": 80,
    "wind_magnitude": 0.4,

    "action_limit": 1.0,

    "cart_limit": 2.4,
    "pole_limit": 0.2,

    "x_limit": 5.0,
    "y_limit": 5.0,
    "angle_limit": 0.5,

    "env_name": "cartpole",
}

# ─────────────────────────────────────────────────────────────
# LLM ACTION (with caching)
# ─────────────────────────────────────────────────────────────

def get_llm_action(obs: np.ndarray) -> float:
    global _last_action, _step_counter

    env_name = ENV_CONFIG["env_name"]
    lim = ENV_CONFIG

    # Reuse previous action to reduce API calls
    if _step_counter % LLM_CALL_INTERVAL != 0:
        _step_counter += 1
        return _last_action

    if env_name == "cartpole":
        cart_pos, cart_vel, pole_angle, pole_vel = obs

        prompt = f"""
You are controlling a CartPole system.

Goal:
- Keep the pole upright (angle near 0)
- Keep the cart near position 0

Cart position: {cart_pos:.4f}
Pole angle: {pole_angle:.4f}

Output ONLY a single float between -1.0 and 1.0.
"""

    elif env_name == "quadrotor-2d":
        x, y, x_dot, y_dot, theta, theta_dot = obs

        prompt = f"""
You are controlling a Quadrotor.

Goal:
- Stabilize position and orientation

X: {x:.4f}
Y: {y:.4f}
Angle: {theta:.4f}

Output ONLY a single float between -1.0 and 1.0.
"""

    else:
        return 0.0

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.choices[0].message.content.strip()

        action = float(np.clip(float(raw), -lim["action_limit"], lim["action_limit"]))

        _last_action = action
        _step_counter += 1

        return action

    except Exception:
        _step_counter += 1
        return _last_action


# ─────────────────────────────────────────────────────────────
# WIND INJECTION
# ─────────────────────────────────────────────────────────────

def inject_wind(env, step: int) -> bool:
    cfg = ENV_CONFIG

    if cfg["wind_start"] <= step < cfg["wind_end"]:
        state = list(env.state)
        env_name = cfg["env_name"]

        if env_name == "cartpole":
            state[1] += cfg["wind_magnitude"]

        elif env_name == "quadrotor-2d":
            state[2] += cfg["wind_magnitude"]
            state[3] += cfg["wind_magnitude"]

        env.state = np.array(state)
        return True

    return False


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────

def make_env_cartpole():
    env = make('cartpole', task_config={
        'task': 'stabilization',
        'done_on_violation': False,
    })

    obs, _ = env.reset(seed=ENV_CONFIG["seed"])
    return env, obs


def make_env_quadrotor():
    env = make('quadrotor', task_config={
        'task': 'stabilization',
        'done_on_violation': False,
    })

    obs, _ = env.reset(seed=ENV_CONFIG["seed"])
    return env, obs


def make_env():
    if ENV_CONFIG["env_name"] == "cartpole":
        return make_env_cartpole()
    elif ENV_CONFIG["env_name"] == "quadrotor-2d":
        return make_env_quadrotor()
    else:
        raise ValueError("Unknown environment")


# ─────────────────────────────────────────────────────────────
# PRINTING HELPERS
# ─────────────────────────────────────────────────────────────

def print_header(level_name: str, extra_info: str = ""):
    cfg = ENV_CONFIG
    print("=" * 68)
    print(f"{level_name} | Env: {cfg['env_name']}")
    print(f"Seed: {cfg['seed']} | Wind: {cfg['wind_start']}-{cfg['wind_end']} | Magnitude: {cfg['wind_magnitude']}")
    if extra_info:
        print(extra_info)
    print("=" * 68)


def print_results(total_violations: int, violation_steps: list, note: str = ""):
    print("=" * 68)
    print("RESULTS")
    print("Total violations:", total_violations)
    print("Violation steps:", violation_steps if violation_steps else "none")
    if note:
        print("Note:", note)
    print("=" * 68)