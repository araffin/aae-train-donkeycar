import os

import cv2
import gym

frame_skip = 4
total_frames = 2000
render = True
output_folder = "logs/carracing-v0"

# Create folder if needed
os.makedirs(output_folder, exist_ok=True)

env = gym.make("CarRacing-v0")
obs = env.reset()
for frame_num in range(total_frames):
    action = env.action_space.sample()
    # do not break too much
    # steer, gas, brake
    action[2] = max(action[2], 0.1)
    for _ in range(frame_skip):
        obs, _, done, _ = env.step(action)
        if done:
            break
    if render:
        env.render()
    path = os.path.join(output_folder, f"{frame_num}.jpg")
    cv2.imwrite(path, obs)
    if done:
        obs = env.reset()
