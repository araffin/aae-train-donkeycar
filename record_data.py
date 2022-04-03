"""
Simple script to manually record data
by driving around.
"""
import argparse
import os
from typing import Tuple

import cv2
import gym
import gym_donkeycar  # noqa: F401
import numpy as np
import pygame
from pygame.locals import *  # noqa: F403

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Path to folder where images will be saved", type=str, required=True)
parser.add_argument("-n", "--max-steps", help="Max number of steps", type=int, default=10000)
args = parser.parse_args()

UP = (1, 0)
LEFT = (0, 1)
RIGHT = (0, -1)
DOWN = (-1, 0)

MAX_TURN = 1
MAX_THROTTLE = 0.5
# Smoothing constants
STEP_THROTTLE = 0.8
STEP_TURN = 0.8

frame_skip = 2
total_frames = args.max_steps
render = True
output_folder = args.folder

# Create folder if needed
os.makedirs(output_folder, exist_ok=True)


def control(
    x,
    theta: float,
    control_throttle: float,
    control_steering: float,
) -> Tuple[float, float]:
    """
    Smooth control.

    :param x:
    :param theta:
    :param control_throttle:
    :param control_steering:
    :return:
    """
    target_throttle = x * MAX_THROTTLE
    target_steering = MAX_TURN * theta
    if target_throttle > control_throttle:
        control_throttle = min(target_throttle, control_throttle + STEP_THROTTLE)
    elif target_throttle < control_throttle:
        control_throttle = max(target_throttle, control_throttle - STEP_THROTTLE)
    else:
        control_throttle = target_throttle

    if target_steering > control_steering:
        control_steering = min(target_steering, control_steering + STEP_TURN)
    elif target_steering < control_steering:
        control_steering = max(target_steering, control_steering - STEP_TURN)
    else:
        control_steering = target_steering
    return control_throttle, control_steering


# pytype: disable=name-error
moveBindingsGame = {K_UP: UP, K_LEFT: LEFT, K_RIGHT: RIGHT, K_DOWN: DOWN}  # noqa: F405
WHITE = (230, 230, 230)
pygame.font.init()
FONT = pygame.font.SysFont("Open Sans", 25)

pygame.init()
window = pygame.display.set_mode((400, 400), RESIZABLE)
# pytype: enable=name-error

control_throttle, control_steering = 0, 0

env = gym.make("donkey-mountain-track-v0")
obs = env.reset()
for frame_num in range(total_frames):
    x, theta = 0, 0
    # Record pressed keys
    keys = pygame.key.get_pressed()
    for keycode in moveBindingsGame.keys():
        if keys[keycode]:
            x_tmp, th_tmp = moveBindingsGame[keycode]
            x += x_tmp
            theta += th_tmp

    # Smooth control for teleoperation
    control_throttle, control_steering = control(x, theta, control_throttle, control_steering)

    window.fill((0, 0, 0))
    pygame.display.flip()
    # Limit FPS
    # pygame.time.Clock().tick(1 / TELEOP_RATE)
    for event in pygame.event.get():
        if (event.type == QUIT or event.type == KEYDOWN) and event.key in [  # pytype: disable=name-error
            K_ESCAPE,  # pytype: disable=name-error
            K_q,  # pytype: disable=name-error
        ]:
            env.close()
            exit()

    window.fill((0, 0, 0))
    text = "Control ready"
    text = FONT.render(text, True, WHITE)
    window.blit(text, (100, 100))
    pygame.display.flip()

    # steer, throttle
    action = np.array([-control_steering, control_throttle])

    for _ in range(frame_skip):
        obs, _, done, _ = env.step(action)
        if done:
            break
    if render:
        env.render()
    path = os.path.join(output_folder, f"{frame_num}.jpg")
    # Convert to BGR
    cv2.imwrite(path, obs[:, :, ::-1])
    if done:
        obs = env.reset()
        control_throttle, control_steering = 0, 0
