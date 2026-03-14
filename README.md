# UCSD-WI26-CSE203B-Convex-Optimization-Project-
Guaranteed Safety in Agentic Planning: Coupling Large Language Models (LLM) with Robust Model Predictive Control (MPC)


SETups Step:
```bash
conda create -n ams python=3.10
conda activate ams

pip install -r requirements.txt

git clone https://github.com/Stanford-ILIAD/safe-control-gym.git
cd safe-control-gym

pip install -e .


# if problem with pybullet
conda install -c conda-forge pybullet

conda install -c anaconda gmp

(optional) Additional requirements for MPC
You may need to separately install acados for fast MPC implementations.

To build and install acados, see their installation guide.
To set up the acados python interface, check out these installation steps.

https://docs.acados.org/installation/index.html
```

---

## The Three Levels

### Level 1 — `level1_llm_baseline.py` — LLM Only (No Safety Filter)

Claude receives the CartPole state as natural language and returns a control
action. There is no mathematical safety guarantee. Errors compound, and the
pole falls reliably during and after the wind window.

**Crashes when:** Wind hits at step 50. The LLM reacts too slowly, has no
predictive model of dynamics, and occasionally fails to parse a clean float.
Constraint violations are frequent.

**Expected violations:** HIGH (15–40+)

---

### Level 2 — `level2_classical_mpc.py` — LLM + Classical MPC Filter

The LLM still proposes an action, but a Classical MPC optimizer corrects it
before it reaches the environment. MPC solves a short-horizon quadratic
program to find the action closest to the LLM's intent that stays within
constraint bounds over the next 10 steps.

**Crashes when:** Wind hits at step 50. Classical MPC has no disturbance model.
Its predicted trajectory is immediately wrong once wind perturbs the cart. It
cannot tighten constraints proactively and only reacts after the state has
already been pushed toward the boundary.

**Expected violations:** MEDIUM (3–10)

---

### Level 3 — `level3_robust_mpc.py` — LLM + Robust MPC + Wind Model

The LLM still proposes an action, but a **Robust MPC** filter corrects it.
Robust MPC pre-computes constraint tightening amounts (δ_pole, δ_cart) based
on the worst-case disturbance bound W_MAX. The optimizer then plans
conservatively within the tightened safe set, leaving headroom so that even
a worst-case wind push cannot violate the original constraint.

**Crashes when:** The actual disturbance exceeds W_MAX (the assumed bound),
or when tightening makes the safe set infeasible. Neither happens under
normal conditions here since W_MAX = 0.35 > WIND_MAGNITUDE = 0.30.

**Expected violations:** LOW (0–3)

---

#### Summary of the Three Levels

> Level 1:   state → [LLM] → action (no model needed).  
> Level 2:   state → [LLM] → [MPC: A, B, N, dt] → safe action.  
> Level 3:   state → [LLM] → [RMPC: A, B, N, dt, W_max, δ] → safe action

| Level | Method | What the controller knows |
|-------|--------|---------------------------|
| 1 | LLM only | Nothing about physics — just language |
| 2 | LLM + Classical MPC | Nominal physics model (A, B) |
| 3 | LLM + Robust MPC | Physics model + disturbance bound (W_max → δ) |

---

## Shared Environment Settings

| Parameter | Value |
|-----------|-------|
| Task | CartPole stabilization |
| Safe pole range | ±0.2 rad |
| Safe cart range | ±2.4 m |
| Episode length | 200 steps |
| Random seed | 42 |
| Wind disturbance | +0.3 on cart velocity, steps 50–80 |
| done_on_violation | False (count all violations) |

---

## Results Table (TO BE FILLED)

| Level | Method | Violations | Violation Steps |
|-------|--------|------------|-----------------|
| 1 | LLM only | ___ | ___ |
| 2 | LLM + Classical MPC | ___ | ___ |
| 3 | LLM + Robust MPC | ___ | ___ |

---

## Setup

```bash
conda activate asm
```

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

Run each level:
```bash
# Level 1 LLM baseline, CartPole
python run_experiment.py --level level1 --env cartpole

# Level 2 classical MPC, Quadrotor 2D
python run_experiment.py --level level2 --env quadrotor-2d

# Level 3 robust MPC, Quadrotor 2D
python run_experiment.py --level level3 --env quadrotor-2d
```

---

## Project Structure

```
AutoSafeMPC/
├── .env                        ← API keys (no commit - added to gitignore)
|                                (to be created individually during setup)
├── safe-control-gym            ← safe-control-gym library (no commit - added to gitignore)
|                                (no commit - cloned from github as part of setup)
├── .env-sample                 ← sample env file with API keys
├── LICENSE                     ← license file
├── .gitignore                  ← must include .env, safe-control-gym directory
├── requirements.txt            ← dependencies
├── README.md
├── env_test.py                 ← initial environment sanity check
├── api_client.py               ← API client for LLM
├── level1_llm_baseline.py      ← LLM only, no safety
├── level2_classical_mpc.py     ← LLM + Classical MPC
├── level3_robust_mpc.py        ← LLM + Robust MPC
└── run_experiment.py           ← Run experiment

```
