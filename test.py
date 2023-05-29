import torch
import gymnasium as gym
import time
from ppo_gymreach import Agent, _preproc_inputs
import numpy as np

seed = 1
env = gym.make("FetchReachDense-v3",render_mode="human")
# env = gym.wrappers.NormalizeReward(env)
# env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

env.action_space.seed(seed)
env.observation_space.seed(seed)

obs = env.observation_space
input_shape = obs["observation"].shape[0] + obs["desired_goal"].shape[0]

print("envs.observation_space.shape", input_shape)
print("envs.action_space.n", env.action_space)

agent = torch.load("save_dir/FetchReachDense-v3_ppo_gymreach_toy_1_1685330224/model_1.0.pt")

test_epoch = 20
local_steps_per_epoch = 100

o, _ = env.reset()

for epoch in range(test_epoch):
    observation, _ = env.reset()
    for step in range(100):
        input_obs = _preproc_inputs(observation['observation'], observation['desired_goal'])

        next_obs = torch.Tensor(input_obs).to("cuda")  # store initial observation

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)  # roll out phase, no need gradient
            action = action.cpu().numpy()
        action = action.flatten()

        observation_new, _, _, _, info = env.step(action)
        if info["is_success"] == 1:
            print(f"success in {step}")
            break
        observation = observation_new
    env.render()