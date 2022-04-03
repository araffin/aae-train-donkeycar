"""
Train an autoencoder model using saved images in a folder
"""
import argparse
import os
import random
import time

import cv2  # pytype: disable=import-error
import numpy as np
import torch as th
from torch.nn import functional as F
from tqdm import tqdm

from ae.autoencoder import Autoencoder
from ae.data_loader import DataLoader
from config import INPUT_DIM, ROI

parser = argparse.ArgumentParser()
parser.add_argument("-ae", "--ae-path", help="Path to saved autoencoder (otherwise start from scratch)", type=str)
parser.add_argument(
    "-f", "--folders", help="Path to folders containing images for training", type=str, nargs="+", required=True
)
parser.add_argument("--z-size", help="Latent space", type=int, default=32)
parser.add_argument("--seed", help="Random generator seed", type=int)
parser.add_argument("--n-samples", help="Max number of samples", type=int, default=-1)
parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=64)
parser.add_argument("--learning-rate", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--n-epochs", help="Number of epochs", type=int, default=10)
parser.add_argument("--verbose", help="Verbosity", type=int, default=1)
parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
args = parser.parse_args()

if args.num_threads > 0:
    th.set_num_threads(args.num_threads)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


folders, images = [], []
for folder in args.folders:
    if not folder.endswith("/"):
        folder += "/"
    folders.append(folder)
    images_ = [folder + im for im in os.listdir(folder) if im.endswith(".jpg")]
    print(f"{folder}: {len(images_)} images")
    images.append(images_)


images = np.concatenate(images)
n_samples = len(images)

if args.n_samples > 0:
    n_samples = min(n_samples, args.n_samples)

print(f"{n_samples} images")

# indices for all time steps where the episode continues
indices = np.arange(n_samples, dtype="int64")
np.random.shuffle(indices)

# split indices into minibatches. minibatchlist is a list of lists; each
# list is the id of the observation preserved through the training
minibatchlist = [
    np.array(sorted(indices[start_idx : start_idx + args.batch_size]))
    for start_idx in range(0, len(indices) - args.batch_size + 1, args.batch_size)
]

data_loader = DataLoader(minibatchlist, images, n_workers=2)

if args.ae_path is not None:
    print("Continuing training")
    # z_size and learning rate will be overwritten
    autoencoder = Autoencoder.load(args.ae_path)
    args.z_size = autoencoder.z_size
else:
    autoencoder = Autoencoder(z_size=args.z_size, learning_rate=args.learning_rate)
autoencoder.to(autoencoder.device)

best_loss = np.inf
ae_id = int(time.time())
save_path = f"logs/ae-{args.z_size}_{ae_id}.pkl"
best_model_path = f"logs/ae-{args.z_size}_{ae_id}_best.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

try:
    for epoch in range(args.n_epochs):
        pbar = tqdm(total=len(minibatchlist))
        train_loss = 0
        for obs, target_obs in data_loader:
            obs = th.as_tensor(obs).to(autoencoder.device)
            target_obs = th.as_tensor(target_obs).to(autoencoder.device)

            autoencoder.optimizer.zero_grad()

            predicted_obs = autoencoder.forward(obs)
            loss = F.mse_loss(predicted_obs, target_obs)

            loss.backward()
            train_loss += loss.item()
            autoencoder.optimizer.step()

            pbar.update(1)
        pbar.close()
        print(f"Epoch {epoch + 1:3}/{args.n_epochs}")
        print("Loss:", train_loss)

        # TODO: use validation set
        if train_loss < best_loss:
            best_loss = train_loss
            print(f"Saving best model to {best_model_path}")
            autoencoder.save(best_model_path)

        # Load test image
        if args.verbose >= 1:
            image_idx = np.random.randint(n_samples)
            image = cv2.imread(images[image_idx])
            r = ROI
            im = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
            # Resize if needed
            if ROI[2] != INPUT_DIM[1] or ROI[3] != INPUT_DIM[0]:
                im = cv2.resize(im, (INPUT_DIM[1], INPUT_DIM[0]), interpolation=cv2.INTER_AREA)
            encoded = autoencoder.encode(im)
            reconstructed_image = autoencoder.decode(encoded)[0]
            # Plot reconstruction
            cv2.imshow("Original", image)
            cv2.imshow("Cropped", im)
            cv2.imshow("Reconstruction", reconstructed_image)
            cv2.waitKey(1)
except KeyboardInterrupt:
    pass

print(f"Saving to {save_path}")
autoencoder.save(save_path)
