"""
Train an autoencoder model using saved images in a folder
"""
import argparse
import os
import random
import time
from typing import Optional

import cv2  # pytype: disable=import-error
import numpy as np
import torch as th
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# from tqdm.rich import tqdm
from tqdm import tqdm

from ae.autoencoder import Autoencoder, preprocess_image, preprocess_input
from ae.data_loader import get_image_augmenter
from ae.match_datasets import prepare_datasets


class RacingDataset(Dataset):
    def __init__(
        self,
        teacher: Autoencoder,
        teacher_folder: str,
        student_folder: str,
        augment: bool = True,
        ae_mask: Optional[Autoencoder] = None,
    ):
        # Do not use autoencoder for matching
        # FIXME: weight seems to be wrong
        self.weight_autoencoder = 0.5 if ae_mask is not None else 1
        datasets, names = prepare_datasets(
            teacher,
            [teacher_folder, student_folder],
            normalize=False,
            ae_mask=ae_mask,
            weight_autoencoder=self.weight_autoencoder,
        )
        self.teacher = teacher
        self.teacher_folder = teacher_folder
        self.student_folder = student_folder
        self.teacher_dataset = datasets[0]
        self.student_dataset = datasets[1]
        self.names = names[1]

        self.augmenter = None
        if augment:
            self.augmenter = get_image_augmenter()
        # Create KNN with first dataset
        self.knn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(self.teacher_dataset)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        _, neighbor_indices = self.knn.kneighbors([self.student_dataset[idx]])
        neighbor_indices = neighbor_indices.flatten()
        # Remove CTE from target
        target = self.teacher_dataset[neighbor_indices[0]][: self.teacher.z_size]
        # Rescale
        # FIXME: this seems to be buggy
        target /= self.weight_autoencoder
        # TODO: try keep the old embedding too (so the model still work on the initial track)
        img_name = os.path.join(self.student_folder, f"{self.names[idx]}.jpg")

        image = preprocess_image(cv2.imread(img_name), convert_to_rgb=False, normalize=False)
        if self.augmenter is not None:
            image = self.augmenter.augment_image(image)
        # Normalize and channel first
        image = preprocess_input(image.astype(np.float32), mode="rl")
        return {"image": th.as_tensor(image), "target": th.as_tensor(target).float()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ae", "--ae-path", help="Path to saved autoencoder", type=str, required=True)
    parser.add_argument("-ae-mask", "--ae-mask-path", help="Path to saved AE for segmentation", type=str)
    parser.add_argument(
        "-s", "--student-folder", help="Path to folder containing images for training", type=str, required=True
    )
    parser.add_argument(
        "-t", "--teacher-folder", help="Path to folder containing images for matching", type=str, required=True
    )
    parser.add_argument("--seed", help="Random generator seed", type=int)
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

    teacher = Autoencoder.load(args.ae_path)
    student = Autoencoder.load(args.ae_path)
    args.z_size = teacher.z_size
    teacher.to(teacher.device)
    student.to(student.device)

    ae_mask = None
    if args.ae_mask_path is not None:
        ae_mask = Autoencoder.load(args.ae_mask_path)
        ae_mask.to(student.device)

    best_loss = np.inf
    ae_id = int(time.time())
    save_path = f"logs/match_ae-{args.z_size}_{ae_id}.pkl"
    best_model_path = f"logs/match_ae-{args.z_size}_{ae_id}_best.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = RacingDataset(teacher, args.teacher_folder, args.student_folder, ae_mask=ae_mask)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    try:
        for epoch in range(args.n_epochs):
            pbar = tqdm(total=len(dataloader))
            train_loss = 0
            for sample in dataloader:
                obs = th.as_tensor(sample["image"]).to(student.device)
                target = th.as_tensor(sample["target"]).to(student.device)

                student.optimizer.zero_grad()

                predicted_latent = student.encode_forward(obs)
                loss = F.mse_loss(predicted_latent, target)

                loss.backward()
                train_loss += loss.item()
                student.optimizer.step()

                pbar.update(1)
            pbar.close()
            print(f"Epoch {epoch + 1:3}/{args.n_epochs}")
            print("Loss:", train_loss)

            # TODO: use validation set
            if train_loss < best_loss:
                best_loss = train_loss
                print(f"Saving best model to {best_model_path}")
                student.save(best_model_path)

    except KeyboardInterrupt:
        pass

    print(f"Saving to {save_path}")
    student.save(save_path)
