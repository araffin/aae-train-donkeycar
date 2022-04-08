import argparse
import os
from typing import List, Tuple

import lightly
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch as th
import torchvision
from lightly.data import LightlyDataset
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from ae.self_supervised.byol import BYOL, CropAndResize
from config import ROI


def generate_embeddings(
    model: pl.LightningModule,
    dataloader: th.utils.data.DataLoader,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generates representations for all images in the dataloader with
    the given model.

    :param model: Pre-trained model.
    :param dataloader: PyTorch dataloader
    :return: Embedding and filenames
    """
    embeddings = []
    filenames = []
    with th.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = th.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


def get_image_as_np_array(filename: str) -> np.ndarray:
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(
    embeddings: List[np.ndarray],
    filenames: List[str],
    folder: str,
    n_neighbors: int = 3,
    num_examples: int = 6,
) -> None:
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = knn.kneighbors(embeddings)

    # get some random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            fname = os.path.join(folder, filenames[neighbor_idx])
            # plot the image
            plt.imshow(get_image_as_np_array(fname))
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            plt.axis("off")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="Path to folder where images are saved", type=str, required=True)
    parser.add_argument("-i", "--model", help="Path to saved model", type=str, required=True)
    # parser.add_argument("-n", "--n-samples", help="Number of samples", type=int, default=256)
    parser.add_argument("--seed", help="Random seed", type=int)
    parser.add_argument("-k", "--n-neighbors", help="Number of neighbors", type=int, default=3)
    args = parser.parse_args()

    margin_left, margin_top, width, height = ROI

    test_transforms = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize((input_size, input_size)),
            CropAndResize(*ROI),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize["mean"],
                std=lightly.data.collate.imagenet_normalize["std"],
            ),
        ]
    )

    # Create a dataset from a folder containing images or videos:
    dataset = LightlyDataset(args.folder, transform=test_transforms)

    dataloader_test = th.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )

    model = BYOL.load_from_checkpoint(args.model)
    model.eval()
    embeddings, filenames = generate_embeddings(model, dataloader_test)

    if args.seed is not None:
        np.random.seed(args.seed)

    plot_knn_examples(embeddings, filenames, args.folder, n_neighbors=args.n_neighbors)
    plt.show()
