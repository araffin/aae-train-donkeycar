import os
from typing import Any, Dict, Tuple

import cv2
import gym
import numpy as np

from ae.autoencoder import load_ae, preprocess_image


class AutoencoderWrapper(gym.Wrapper):
    """
    Wrapper to encode input image using pre-trained AutoEncoder

    :param env: Gym environment
    :param ae_path: absolute path to the pretrained AutoEncoder
    """

    def __init__(self, env: gym.Env, ae_path: str = os.environ["AE_PATH"]):
        super().__init__(env)
        self.autoencoder = load_ae(ae_path)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.autoencoder.z_size + 1,),
            dtype=np.float32,
        )
        self.frame_num = 0
        if os.environ.get("AE_RETRAIN_PATH"):
            # Create folder if needed
            os.makedirs(os.environ.get("AE_RETRAIN_PATH"), exist_ok=True)

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        # Convert to BGR
        encoded_image = self.autoencoder.encode_from_raw_image(obs[:, :, ::-1])
        new_obs = np.concatenate([encoded_image.flatten(), [0.0]])
        return new_obs.flatten()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self.env.step(action)
        # Encode with the pre-trained AutoEncoder
        encoded_image = self.autoencoder.encode_from_raw_image(obs[:, :, ::-1])

        if os.environ.get("AE_RETRAIN_PATH") is not None:
            cropped_image = preprocess_image(obs[:, :, ::-1], convert_to_rgb=False, normalize=False)
            reconstructed_image = self.autoencoder.decode(encoded_image)[0]

            error = np.mean((cropped_image - reconstructed_image) ** 2)
            # Threshold above which the image will be saved for retrain
            max_ae_error = 80

            if error > max_ae_error:
                # save image with high error for retrain
                path = os.path.join(os.environ.get("AE_RETRAIN_PATH"), f"{self.frame_num}.jpg")
                # Convert to BGR
                cv2.imwrite(path, obs[:, :, ::-1])
                self.frame_num += 1

        # cv2.imshow("Original", obs[:, :, ::-1])
        # cv2.imshow("Reconstruction", reconstructed_image)
        # # stop if escape is pressed
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:
        #     pass
        speed = infos["speed"]
        new_obs = np.concatenate([encoded_image.flatten(), [speed]])

        return new_obs.flatten(), reward, done, infos
