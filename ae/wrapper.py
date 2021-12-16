import os
from typing import Optional

import gym
import numpy as np

from ae.autoencoder import load_ae


class AutoencoderWrapper(gym.Wrapper):
    def __init__(self, env, ae_path: Optional[str] = os.environ.get("AAE_PATH"), add_speed: bool = False):
        super().__init__(env)
        assert ae_path is not None
        self.ae = load_ae(ae_path)
        self.add_speed = add_speed
        additional_dim = 2 if self.add_speed else 0
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.ae.z_size + additional_dim,), dtype=np.float32
        )

    def reset(self):
        # Convert to BGR
        encoded = self.ae.encode_from_raw_image(self.env.reset()[:, :, ::-1]).flatten()
        if self.add_speed:
            return np.concatenate((encoded, self.env.car.hull.linearVelocity / 100)).flatten()
        return encoded

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        encoded = self.ae.encode_from_raw_image(obs[:, :, ::-1]).flatten()
        if self.add_speed:
            encoded = np.concatenate((encoded, self.env.car.hull.linearVelocity / 100)).flatten()
        return encoded, reward, done, infos
