import os
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import torch as th

from ae.self_supervised.byol import BYOL, get_test_transform


class SelfSupervisedWrapper(gym.Wrapper):
    """
    Gym wrapper to encode image and reduce input dimension
    using pre-trained auto-encoder
    (only the encoder part is used here, decoder part can be used for debug)

    :param env: Gym environment
    :param byol_path: Path to the pretrained model
    :param add_speed: Add speed to input
    """

    def __init__(
        self,
        env: gym.Env,
        byol_path: Optional[str] = os.environ.get("BYOL_PATH"),  # noqa: B008
        add_speed: bool = True,
    ):
        super().__init__(env)
        assert byol_path is not None, "No path to BYOL model was provided"
        self.model = BYOL.load_from_checkpoint(byol_path)
        self.model.eval()
        self.test_transform = get_test_transform()
        self.embedding_dim = self.encode_image(env.observation_space.sample()).shape[0]
        # Update observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim + 1,), dtype=np.float32)
        self.add_speed = add_speed

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        with th.no_grad():
            # Reorder channels
            # H x W x C -> C x H x W
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            img = self.test_transform(th.as_tensor(image[None]))
            img = img.to(self.model.device)
            return self.model.backbone(img).flatten().cpu().numpy()

    def reset(self) -> np.ndarray:
        return np.concatenate([self.encode_image(self.env.reset()), [0.0]])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self.env.step(action)
        speed = infos["speed"] if self.add_speed else 0.0
        new_obs = np.concatenate([self.encode_image(obs), [speed]])
        return new_obs, reward, done, infos
