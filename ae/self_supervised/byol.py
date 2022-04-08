import argparse
import copy
import time
from typing import Tuple, Union

import imgaug
import numpy as np
import pytorch_lightning as pl
import torch as th
import torchvision
import torchvision.transforms as T
from imgaug import augmenters as iaa
from imgaug.augmenters import Sometimes
from lightly.data import BaseCollateFunction, LightlyDataset
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms import RandomRotate  # GaussianBlur
from torch import nn

from ae.data_loader import RandomShadows
from config import INPUT_DIM, ROI

imagenet_normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def get_test_transform() -> th.ScriptModule:
    return th.jit.script(
        th.nn.Sequential(
            CropAndResize(*ROI),
            torchvision.transforms.Normalize(
                mean=imagenet_normalize["mean"],
                std=imagenet_normalize["std"],
            ),
        )
    )


class CropAndResize(nn.Module):
    def __init__(self, margin_left: int, margin_top: int, width: int, height: int):
        self.margin_left = margin_left
        self.margin_top = margin_top
        self.width = width
        self.height = height
        super().__init__()

    def forward(self, image: th.Tensor) -> th.Tensor:
        image = T.functional.crop(image, self.margin_top, self.margin_left, self.height, self.width)
        # Hack: resize if needed, better to change conv2d  kernel size / padding
        # if ROI[2] != INPUT_DIM[1] or ROI[3] != INPUT_DIM[0]:
        #     image = T.Resize((INPUT_DIM[1], INPUT_DIM[0]))(T.functional.to_pil_image(image))
        return image


class CustomImageCollateFunction(BaseCollateFunction):
    """Implementation of a collate function for images.
    This is an implementation of the BaseCollateFunction with a concrete
    set of transforms.
    The set of transforms is inspired by the SimCLR paper as it has shown
    to produce powerful embeddings.
    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random (+90 degree) rotation is applied.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int]] = 64,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        # min_scale: float = 1.0,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: float = 0.1,
        # vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        normalize: dict = imagenet_normalize,
    ):

        # if isinstance(input_size, tuple):
        #     input_size_ = max(input_size)
        # else:
        #     input_size_ = input_size

        cj_bright = cj_strength * 0.8
        cj_contrast = cj_strength * 0.8
        cj_sat = cj_strength * 0.8
        cj_hue = cj_strength * 0.2

        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)

        # See https://github.com/aleju/imgaug/issues/406
        img_aug_transform = iaa.Sequential(
            [
                # Add shadows (from https://github.com/OsamaMazhar/Random-Shadows-Highlights)
                Sometimes(0.3, RandomShadows(1.0)),
                Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),
                Sometimes(0.5, iaa.MotionBlur(k=(3, 11), angle=(0, 360))),
                Sometimes(0.4, iaa.Add((-25, 25), per_channel=0.5)),
                # 20% of the corresponding size of the height and width
                Sometimes(0.3, iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
            ],
            random_order=True,
        ).augment_image

        transform = [
            CropAndResize(*ROI),
            # For compat with imgaug
            np.asarray,
            img_aug_transform,
            RandomRotate(prob=rr_prob),
            # Note: the target should also be flipped horizontally
            # T.RandomHorizontalFlip(p=hf_prob),
            # T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            # T.RandomGrayscale(p=random_gray_scale),
            # GaussianBlur(kernel_size=kernel_size * input_size_, prob=gaussian_blur),
            T.ToTensor(),
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        transform = T.Compose(transform)

        super().__init__(transform)


class SmallEncoder(nn.Module):
    def __init__(
        self,
        input_dimension: Tuple[int, int, int] = INPUT_DIM,
        encoder_dim: int = 256,
    ):
        super().__init__()
        # Re-order
        h, w, c = input_dimension
        input_shape = (c, h, w)
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
        self.encode_linear = nn.Linear(flatten_size, encoder_dim)

    def forward(self, input_tensor: th.Tensor) -> th.Tensor:
        """
        :param input_tensor: Input image (as pytorch Tensor)
        :return:
        """
        h = self.encoder(input_tensor).reshape(input_tensor.size(0), -1)
        return self.encode_linear(h)


class BYOL(pl.LightningModule):
    def __init__(self, use_resnet: bool = False, encoder_dim: int = 256, projector_dim: int = 256):
        super().__init__()
        if use_resnet:
            resnet = torchvision.models.resnet18()
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.projection_head = BYOLProjectionHead(512, 1024, 256)
            self.prediction_head = BYOLProjectionHead(256, 1024, 256)
        else:
            self.backbone = SmallEncoder(encoder_dim=encoder_dim)
            self.projection_head = BYOLProjectionHead(encoder_dim, 256, projector_dim)
            self.prediction_head = BYOLProjectionHead(encoder_dim, 256, projector_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x: th.Tensor) -> th.Tensor:
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx: int) -> th.Tensor:
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        (x0, x1), _, _ = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        return loss

    def configure_optimizers(self) -> th.optim.Optimizer:
        # TODO: try with Adam
        return th.optim.SGD(self.parameters(), lr=0.06)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="Path to folder where images are saved", type=str, required=True)
    parser.add_argument("-n", "--max-epochs", help="Max number of epochs", type=int, default=10)
    parser.add_argument("-bs", "--batch-size", help="Minibatch size", type=int, default=256)
    parser.add_argument("-size", "--encoder-dim", help="Embedding dimension", type=int, default=256)
    parser.add_argument("-i", "--model", help="Path to saved model", type=str)
    args = parser.parse_args()

    if args.model is not None:
        model = BYOL.load_from_checkpoint(args.model)
    else:
        model = BYOL(encoder_dim=args.encoder_dim, projector_dim=args.encoder_dim)

    # Create a dataset from a folder containing images or videos:
    dataset = LightlyDataset(args.folder)

    # collate_fn = SimCLRCollateFunction(input_size=INPUT_DIM[0])
    # create a collate function which performs the random augmentations
    # collate_fn = lightly.data.BaseCollateFunction(transform)
    collate_fn = CustomImageCollateFunction(input_size=INPUT_DIM[:1], cj_prob=0.0)

    # See https://github.com/aleju/imgaug/issues/406
    def worker_init_fn(worker_id):
        imgaug.seed(np.random.get_state()[1][0] + worker_id)

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        worker_init_fn=worker_init_fn,
    )

    gpus = 1 if th.cuda.is_available() else 0

    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=gpus, log_every_n_steps=10)
    model_id = int(time.time())
    try:
        trainer.fit(model=model, train_dataloaders=dataloader)
    except KeyboardInterrupt:
        pass
    trainer.save_checkpoint(f"logs/byol_{model_id}.ckpt")
    print(f"Saving to logs/byol_{model_id}.ckpt")
