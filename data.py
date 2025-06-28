from typing import List, Tuple
import random
import cv2
import numpy as np
import albumentations as A
import os
from PIL import Image
from util import load_dataset

# constants
IMG_ORIG_WIDTH = 500
IMG_ORIG_HEIGHT = 375

def train_test_split_dataset(
        images: np.ndarray,
        scribbles: np.ndarray,
        ground_truth: np.ndarray,
        test_size: float = 0.2,
        random_seed: int | None = None
):
    """
    Split dataset into train and test subsets.

    Args:
        images (np.ndarray): (N, H, W, 3) RGB images
        scribbles (np.ndarray): (N, H, W) scribble masks
        ground_truth (np.ndarray): (N, H, W) ground truth masks
        test_size (float): fraction of data used for testing (default 0.2)
        random_seed (int or None): seed for reproducibility

    Returns:
        (train_images, train_scribbles, train_gt, train_filenames),
        (test_images, test_scribbles, test_gt, test_filenames),
        palette
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    N = images.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    split_idx = int(N * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_images = images[train_idx]
    train_scribbles = scribbles[train_idx]
    train_gt = ground_truth[train_idx]

    test_images = images[test_idx]
    test_scribbles = scribbles[test_idx]
    test_gt = ground_truth[test_idx]

    return (
        (train_images, train_scribbles, train_gt),
        (test_images, test_scribbles, test_gt),
    )

def pad_to_512(img: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
        Pads an image or mask to 512x512 with the given pad_value.
        The padding is centered (equal on all sides as much as possible).

        :param img: The input image or mask as a NumPy array.
        :param pad_value: Value to use for padding.
                          Use 0 for images and ground truths, 255 for scribbles.
        :return: Padded image or mask.
        """
    h, w = img.shape[:2]
    pad_h = max(0, 512 - h)
    pad_w = max(0, 512 - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 2:  # grayscale or mask
        pad_val = (pad_value,)
    elif img.ndim == 3 and img.shape[2] == 3:  # RGB image
        pad_val = (pad_value, pad_value, pad_value)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    return cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_val)

def augment_triplet(
    image: np.ndarray,
    scribble: np.ndarray,
    ground_truth: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Given an (image, scribble, ground_truth) triplet,
    apply 5 augmentations (original, crop+resize, flip, rotate, color jitter)
    with exactly the same geometric transforms on all three, so masks stay aligned.

    Returns list of triplets (image_aug, scribble_aug, ground_truth_aug).
    """

    h, w = image.shape[:2]  # h=375, w=500
    # random crop
    transform_crop = A.Compose([
        A.RandomResizedCrop(size=(int(h * 0.8), int(w * 0.8)), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),
    ])

    # mirror
    transform_flip = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),  # keep original size
    ])

    # random rotate, scribbles need different filling!
    angle = random.uniform(-15, 15)
    # for img and gt
    transform_rotate_img = A.Compose([
        A.Rotate(limit=(angle, angle), border_mode=0, fill=0, p=1.0), # fill with black
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),  # keep original size
    ])
    # for scribble
    transform_rotate_scrib = A.Compose([
        A.Rotate(limit=(angle, angle), border_mode=0, fill=255, p=1.0),  # fill with white
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),  # keep original size
    ])

    # color jitter
    transform_color = A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0, p=1.0),
        # No resizing here needed for color-only transform because image is not spatially changed
    ])

    # Original (just pad to 512x512)
    orig_img = pad_to_512(image)
    orig_scribble = pad_to_512(scribble, pad_value=255)
    orig_gt = pad_to_512(ground_truth)

    # Augmented versions
    crop = transform_crop(image=image, masks=[scribble, ground_truth])
    crop_img = pad_to_512(crop['image'])
    crop_scribble = pad_to_512(crop['masks'][0], pad_value=255)
    crop_gt = pad_to_512(crop['masks'][1])

    flip = transform_flip(image=image, masks=[scribble, ground_truth])
    flip_img = pad_to_512(flip['image'])
    flip_scribble = pad_to_512(flip['masks'][0], pad_value=255)
    flip_gt = pad_to_512(flip['masks'][1])

    rotate_img = transform_rotate_img(image=image)['image']
    rotate_scribble = transform_rotate_scrib(image=scribble)['image']
    rotate_gt = transform_rotate_img(image=ground_truth)['image']
    rotate_img = pad_to_512(rotate_img)
    rotate_scribble = pad_to_512(rotate_scribble, pad_value=255)
    rotate_gt = pad_to_512(rotate_gt)

    # Color jitter applied only on image, masks resized separately to original size and padded
    color_img = transform_color(image=image)['image']
    color_img = pad_to_512(color_img)

    # Resize masks back to original size then pad (since no geometric transform for color)
    color_scribble = A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH)(image=scribble)['image']
    color_scribble = pad_to_512(color_scribble, pad_value=255)

    color_gt = A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH)(image=ground_truth)['image']
    color_gt = pad_to_512(color_gt)

    augmented_triplets = [
        (orig_img, orig_scribble, orig_gt),
        (crop_img, crop_scribble, crop_gt),
        (flip_img, flip_scribble, flip_gt),
        (rotate_img, rotate_scribble, rotate_gt),
        (color_img, color_scribble, color_gt),
    ]

    return augmented_triplets


def save_triplets(triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                  img_path: str, scrib_path: str, gt_path: str, palette: List[int], start_idx: int = 0) -> int:
    """
    Saves a list of augmentation triplets
    """
    idx = start_idx
    for triplet in triplets:
        filename_img = f"{idx:04d}.jpg"
        filename_mask = f"{idx:04d}.png"

        img_pil = Image.fromarray(triplet[0])
        scrib_pil = Image.fromarray(triplet[1].astype(np.uint8))
        gt_pil = Image.fromarray(triplet[2].astype(np.uint8), mode='P')
        gt_pil.putpalette(palette)

        img_pil.save(os.path.join(img_path, filename_img), quality=95)
        scrib_pil.save(os.path.join(scrib_path, filename_mask))
        gt_pil.save(os.path.join(gt_path, filename_mask))

        idx += 1
    return idx

def augment_and_save(source_path: str, save_path: str, save_dir: str):
    """
    Augments the training data according to the documentation and saves the resulting data
    @param source_path: path to the directory containing images, scribbles, ground truths
    """
    total_path = os.path.join(save_path, save_dir)
    img_path = os.path.join(total_path, "images")
    scrib_path = os.path.join(total_path, "scribbles")
    gt_path = os.path.join(total_path, "ground_truth")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(scrib_path):
        os.makedirs(scrib_path)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)

    images, scribbles, ground_truths, _, palette = load_dataset(
        source_path, "images", "scribbles", "ground_truth"
    )

    # ensure we have equal amounts of everything
    N_images, *_ = images.shape
    N_scribbles, *_ = scribbles.shape
    N_ground_truths, *_ = ground_truths.shape
    assert N_images == N_scribbles == N_ground_truths

    next_id = 0 # image ID for saving
    for (img, scrib, gt) in zip(images, scribbles, ground_truths):
        augmented_triplets = augment_triplet(img, scrib, gt)
        next_id = save_triplets(augmented_triplets, img_path, scrib_path, gt_path, palette, next_id)