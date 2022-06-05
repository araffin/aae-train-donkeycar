import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation

# feature_extractor = AutoFeatureExtractor.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")
# model = SegformerForSemanticSegmentation.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")

feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")


def compute_mask(image_path: str):
    image = read_image(image_path)
    pil_image = Image.open(image_path)

    with th.no_grad():
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    # flat-road, flat-cyclinglane
    # road_seg = upsampled_logits[0, model.config.label2id["road"]]
    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    pred_mask = pred_seg == model.config.label2id["road"]
    return image, pred_mask, pred_seg


def compute_iou(mask1, mask2) -> float:
    # IoU calculation
    mask1 = mask1.cpu().numpy()
    mask2 = mask2.cpu().numpy()
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def plot(image, mask, name="Image", alpha=0.7):
    img_with_mask = draw_segmentation_masks(image, masks=mask.bool(), alpha=alpha)
    img = F.to_pil_image(img_with_mask)
    plt.figure(name)
    plt.imshow(img)


image_path = "/home/antonin/Images/matching/129.jpg"

image, pred_mask, pred_seg = compute_mask(image_path)

plt.figure("Segmentation")
plt.imshow(pred_seg)

plot(image, pred_mask, "Image1")


image_path = "/home/antonin/Images/matching/74.jpg"

image2, pred_mask2, pred_seg2 = compute_mask(image_path)


plot(image2, pred_mask2, "Image2")

iou_score = compute_iou(pred_mask, pred_mask2)
print(f"IOU = {iou_score:.2f}")

plt.show()
