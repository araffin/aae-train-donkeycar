import argparse
import os

import lightly
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision
from lightly.data import LightlyDataset
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from byol import BYOL, CropAndResize  # pytype: disable=import-error


def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
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


def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(embeddings, filenames, folder, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
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


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Path to folder where images are saved", type=str, required=True)
# parser.add_argument("-n", "--n-samples", help="Number of samples", type=int, default=256)
args = parser.parse_args()

test_transforms = torchvision.transforms.Compose(
    [
        # torchvision.transforms.Resize((input_size, input_size)),
        CropAndResize(),
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

model = BYOL.load_from_checkpoint("logs/byol.ckpt")
model.eval()
embeddings, filenames = generate_embeddings(model, dataloader_test)

plot_knn_examples(embeddings, filenames, args.folder)
plt.show()
