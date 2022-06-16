import argparse
import functools
import glob
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision.transforms.functional as F
from torch import nn
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation

# feature_extractor = AutoFeatureExtractor.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")
# model = SegformerForSemanticSegmentation.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")

feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

print(model.config.id2label)
# import ipdb; ipdb.set_trace()


def compute_mask(image: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    with th.no_grad():
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.shape[1:],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    # flat-road, flat-cyclinglane
    # road_seg = upsampled_logits[0, model.config.label2id["road"]]
    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    # Convert to bool
    pred_masks = []
    # Note: include "fence" for generated dataset
    for label in ["road", "runway"]:
        pred_masks.append(pred_seg == model.config.label2id[label])
    pred_mask = functools.reduce(np.logical_or, pred_masks)
    return pred_mask, pred_seg


def compute_iou(mask1: th.Tensor, mask2: th.Tensor) -> float:
    # IoU calculation
    mask1 = mask1.cpu().numpy()
    mask2 = mask2.cpu().numpy()
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def plot(image: th.Tensor, mask: th.Tensor, name: str = "Image", alpha: float = 0.7) -> None:
    img_with_mask = draw_segmentation_masks(image, masks=mask.bool(), alpha=alpha)
    plt.figure(name)
    plt.imshow(F.to_pil_image(img_with_mask))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment images and compute IOU")
    parser.add_argument("-i", "--input-images", help="Input image", type=str, nargs="+")
    parser.add_argument("-f", "--folder", help="Input image", type=str)
    args = parser.parse_args()

    if args.folder is not None:
        for image_path in glob.glob(f"{args.folder}/*.jpg"):
            image = read_image(image_path)
            pred_mask, pred_seg = compute_mask(image)

            try:
                plot(image, pred_mask, "Image")

                plt.figure("Segmentation")
                plt.imshow(pred_seg)
                plt.show()
            except KeyboardInterrupt:
                break
        exit()

    image = read_image(args.input_images[0])
    pred_mask, pred_seg = compute_mask(image)

    plot(image, pred_mask, "Image1")
    plt.figure("Segmentation")
    plt.imshow(pred_seg)

    if len(args.input_images) > 1:

        image2 = read_image(args.input_images[1])
        pred_mask2, pred_seg2 = compute_mask(image2)
        plot(image2, pred_mask2, "Image2")
        iou_score = compute_iou(pred_mask, pred_mask2)
        print(f"IOU = {iou_score:.2f}")

    plt.show()