import safe_control_gym
from safe_control_gym.utils.registration import make

env = make('cartpole', **{
    'task_config': {
        'task': 'stabilization',
        'done_on_violation': False,
    }
})

obs, info = env.reset()
done = False

for i in range(200):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Obs:", obs, "| Constraint violation:", info.get('constraint_violation', None))
    if done:
        break

env.close()