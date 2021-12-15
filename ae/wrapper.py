import os
from typing import Optional

import gym
import numpy as np

from ae.autoencoder import load_ae


class AutoencoderWrapper(gym.Wrapper):
    def __init__(self, env, ae_path: Optional[str] = os.environ.get("AAE_PATH")):
        super().__init__(env)
        assert ae_path is not None
        self.ae = load_ae(ae_path)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ae.z_size,), dtype=np.float32)

    def reset(self):
        return self.ae.encode_from_raw_image(self.env.reset()).flatten()

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return self.ae.encode_from_raw_image(obs).flatten(), reward, done, infos
