import argparse
import pickle
from pathlib import Path

import cv2  # pytype: disable=import-error
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ae.autoencoder import load_ae

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folders", type=str, nargs="+", required=True)
parser.add_argument("-ae", "--ae-path", help="Path to saved AE", type=str)
parser.add_argument("-n", "--n-samples", help="Max number of samples", type=int, default=20)
parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)

autoencoder = None
if args.ae_path is not None:
    autoencoder = load_ae(args.ae_path)

datasets = []
info_dicts = []
names = []
for folder in args.folder:
    with open(Path(folder) / "infos.pkl", "rb") as f:
        infos = pickle.load(f)

    # Preprocess and create features
    if autoencoder is not None:
        dataset = np.zeros((len(infos), autoencoder.z_size + 1))
        for i, (name, info) in enumerate(infos.items()):
            input_image = cv2.imread(str(Path(folder) / f"{name}.jpg"))
            encoded_image = autoencoder.encode_from_raw_image(input_image).flatten()
            dataset[i][0] = info["cte"]
            dataset[i][1:] = encoded_image
    else:
        dataset = [[frame["cte"], 0.0] for frame in infos.values()]

    datasets.append(np.array(dataset))
    info_dicts.append(infos)
    names.append(list(infos.keys()))

# Normalize according to first dataset
normalizer = StandardScaler().fit(datasets[0])
for i, dataset in enumerate(datasets):
    datasets[i] = normalizer.transform(dataset)
    # More weight for CTE
    datasets[i][:, 0] *= 5

# Create KNN with first dataset
knn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(datasets[0])

other_dataset = datasets[1]
random_samples = np.random.permutation(len(other_dataset))[: args.n_samples]
image_grid = []
for idx, sample in enumerate(random_samples):

    _, neighbor_indices = knn.kneighbors([other_dataset[sample]])
    neighbor_indices = neighbor_indices.flatten()

    image1_path = str(Path(args.folder[1]) / f"{names[1][sample]}.jpg")
    image1 = cv2.imread(image1_path)

    neighbors = []
    for neighbor_idx in neighbor_indices:
        image_path = str(Path(args.folder[0]) / f"{names[0][neighbor_idx]}.jpg")
        neighbor_image = cv2.imread(image_path)
        neighbors.append(neighbor_image)
    image_grid.append(np.hstack([image1] + neighbors))

    if (idx + 1) % 5 == 0:
        grid = np.array(image_grid)
        cv2.imshow("Image grid", grid.reshape((-1,) + grid.shape[2:]))
        # stop if escape is pressed
        k = cv2.waitKey(0) & 0xFF

        if k == 27:
            break
        image_grid = []
