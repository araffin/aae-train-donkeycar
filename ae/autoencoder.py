from typing import Optional, Tuple

import cv2
import numpy as np
import torch as th
from torch import nn

from config import INPUT_DIM, RAW_IMAGE_SHAPE, ROI


class Autoencoder(nn.Module):
    """
    Wrapper to manipulate an autoencoder.

    :param z_size: latent space dimension
    :param input_dimension: input dimension
    :param learning_rate:
    :param normalization_mode: Type of normalization to use, one of "rl" or "tf"
        By default, divides by 255.0 to scale to [0, 1]
    """

    def __init__(
        self,
        z_size: int,
        input_dimension: Tuple[int, int, int] = INPUT_DIM,
        learning_rate: float = 0.0001,
        normalization_mode: str = "rl",
    ):
        super().__init__()
        # AE input and output shapes
        self.z_size = z_size
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate

        # Training params
        self.normalization_mode = normalization_mode

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.encoder = None
        self.decoder = None
        self.shape_before_flatten = None

        # Re-order
        h, w, c = input_dimension
        self._build((c, h, w))
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode_from_raw_image(self, raw_image: np.ndarray) -> np.ndarray:
        """
        Crop and encode a BGR image.
        It returns the corresponding latent vector.

        :param raw_image: BGR image
        :return:
        """
        return self.encode(preprocess_image(raw_image, convert_to_rgb=False, normalize=False))

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize and encode a cropped image.

        :param observation: Cropped image
        :return: corresponding latent vector
        """
        assert observation.shape == self.input_dimension, f"{observation.shape} != {self.input_dimension}"
        # Normalize
        observation = preprocess_input(observation.astype(np.float32), mode=self.normalization_mode)[None]
        with th.no_grad():
            observation = th.as_tensor(observation).to(self.device)
            return self.encode_forward(observation).cpu().numpy()

    def decode(self, arr: np.ndarray) -> np.ndarray:
        """
        :param arr: latent vector
        :return: BGR image
        """
        assert arr.shape == (1, self.z_size), f"{arr.shape} != {(1, self.z_size)}"
        # Decode
        with th.no_grad():
            arr = th.as_tensor(arr).float().to(self.device)
            arr = self.decode_forward(arr).cpu().numpy()
        # Denormalize
        arr = denormalize(arr, mode=self.normalization_mode)
        return arr

    def _build(self, input_shape: Tuple[int, int, int]) -> None:
        """
        Build encoder/decoder networks.

        :param input_shape:
        """
        # n_channels, kernel_size, strides, activation, padding=0
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        # Compute the shape doing a forward pass
        self.shape_before_flatten = self.encoder(th.ones((1,) + input_shape)).shape[1:]
        flatten_size = int(np.prod(self.shape_before_flatten))

        self.encode_linear = nn.Linear(flatten_size, self.z_size)
        self.decode_linear = nn.Linear(self.z_size, flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_shape[0], kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def encode_forward(self, input_tensor: th.Tensor) -> th.Tensor:
        """
        :param input_tensor: Input image (as pytorch Tensor)
        :return: Latent vector
        """
        h = self.encoder(input_tensor).reshape(input_tensor.size(0), -1)
        return self.encode_linear(h)

    def decode_forward(self, z: th.Tensor) -> th.Tensor:
        """
        :param z: Latent vector
        :return: Reconstructed image
        """
        h = self.decode_linear(z).reshape((z.size(0),) + self.shape_before_flatten)
        return self.decoder(h)

    def forward(self, input_image: th.Tensor) -> th.Tensor:
        return self.decode_forward(self.encode_forward(input_image))

    def save(self, save_path: str) -> None:
        """
        Save to a pickle file.

        :param save_path:
        """
        data = {
            "z_size": self.z_size,
            "learning_rate": self.learning_rate,
            "input_dimension": self.input_dimension,
            "normalization_mode": self.normalization_mode,
        }

        th.save({"state_dict": self.state_dict(), "data": data}, save_path)

    @classmethod
    def load(cls, load_path: str) -> "Autoencoder":
        device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        saved_variables = th.load(load_path, map_location=device)
        model = cls(**saved_variables["data"])
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model


def preprocess_input(x: np.ndarray, mode: str = "rl") -> np.ndarray:
    """
    Normalize input.

    :param x: (RGB image with values between [0, 255])
    :param mode: One of "tf" or "rl".
        - rl: divide by 255 only (rescale to [0, 1])
        - tf: will scale pixels between -1 and 1,
            sample-wise.
    :return: Scaled input
    """
    assert x.shape[-1] == 3, f"Color channel must be at the end of the tensor {x.shape}"
    # RL mode: divide only by 255
    x /= 255.0

    if mode == "tf":
        x -= 0.5
        x *= 2.0
    elif mode == "rl":
        pass
    else:
        raise ValueError("Unknown mode for preprocessing")
    # Reorder channels
    # B x H x W x C -> B x C x H x W
    # if len(x.shape) == 4:
    #     x = np.transpose(x, (0, 2, 3, 1))
    x = np.transpose(x, (2, 0, 1))

    return x


def denormalize(x: np.ndarray, mode: str = "rl") -> np.ndarray:
    """
    De normalize data (transform input to [0, 1])

    :param x:
    :param mode: One of "tf" or "rl".
    :return:
    """

    if mode == "tf":
        x /= 2.0
        x += 0.5
    elif mode == "rl":
        pass
    else:
        raise ValueError("Unknown mode for denormalize")

    # Reorder channels
    # B x C x H x W -> B x H x W x C
    if len(x.shape) == 4:
        x = np.transpose(x, (0, 2, 3, 1))
    else:
        x = np.transpose(x, (2, 0, 1))

    # Clip to fix numeric imprecision (1e-09 = 0)
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def preprocess_image(image: np.ndarray, convert_to_rgb: bool = False, normalize: bool = True) -> np.ndarray:
    """
    Crop, resize and normalize image.
    Optionnally it also converts the image from BGR to RGB.

    :param image: image (BGR or RGB)
    :param convert_to_rgb: whether the conversion to rgb is needed or not
    :param normalize: Whether to normalize or not
    :return:
    """
    assert image.shape == RAW_IMAGE_SHAPE, f"{image.shape} != {RAW_IMAGE_SHAPE}"
    # Crop
    # Region of interest
    r = ROI
    image = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
    im = image
    # Hack: resize if needed, better to change conv2d  kernel size / padding
    if ROI[2] != INPUT_DIM[1] or ROI[3] != INPUT_DIM[0]:
        im = cv2.resize(im, (INPUT_DIM[1], INPUT_DIM[0]), interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    if convert_to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Normalize
    if normalize:
        im = preprocess_input(im.astype(np.float32), mode="rl")

    return im


def load_ae(
    path: Optional[str] = None,
    z_size: Optional[int] = None,
    quantize: bool = False,
) -> Autoencoder:
    """
    :param path:
    :param z_size:
    :param quantize: Whether to quantize the model or not
    :return:
    """
    # z_size will be recovered from saved model
    if z_size is None:
        assert path is not None

    # Hack to make everything work without trained AE
    if path == "dummy":
        autoencoder = Autoencoder(z_size=1)
    else:
        autoencoder = Autoencoder.load(path)
    print(f"Dim AE = {autoencoder.z_size}")
    print("PyTorch", th.__version__)
    return autoencoder
