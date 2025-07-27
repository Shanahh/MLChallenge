import os
import random
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# constants
IMG_ORIG_WIDTH = 500
IMG_ORIG_HEIGHT = 375

def train_test_split_dataset(
        images: np.ndarray,
        scribbles: np.ndarray,
        ground_truth: np.ndarray,
        test_size,
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

class SegmentationDataset(Dataset):
    def __init__(self, images, scribbles, masks, transform=None):
        """
        Parameters:
            images (np.ndarray): (N, H, W, 3) RGB images
            scribbles (np.ndarray): (N, H, W) scribble masks
            masks (np.ndarray): (N, H, W) ground truth masks
        """
        self.images = images
        self.scribbles = scribbles
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]           # shape: H x W x 3 (RGB)
        scribble = self.scribbles[idx]     # shape: H x W (single channel)
        mask = self.masks[idx]             # shape: H x W (single channel)

        # Normalize and convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # to C x H x W, float in [0,1]
        scribble = torch.from_numpy(scribble).unsqueeze(0).float() / 255.0  # add channel dim, float in [0,1]
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # add channel dim, now (1 x H x W)

        # Combine image and scribble into a 4 channel input tensor
        input_tensor = torch.cat([image, scribble], dim=0)  # shape: 4 x H x W

        if self.transform:
            input_tensor, mask = self.transform(input_tensor, mask)

        return input_tensor, mask

def _pad_to_512(img: np.ndarray, pad_value: int = 0) -> np.ndarray:
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

def _remove_padding_gt(padded_img: np.ndarray) -> np.ndarray:
    """
    Removes padding from a single 512x512 ground truth mask to recover the original image shape.

    :param padded_img: The padded gt (512x512).
    :return: Cropped image of shape original_shape.
    """
    pad_h = max(0, 512 - IMG_ORIG_HEIGHT)
    pad_w = max(0, 512 - IMG_ORIG_WIDTH)
    top = pad_h // 2
    left = pad_w // 2

    # Crop the image
    return padded_img[top:top + IMG_ORIG_HEIGHT, left:left + IMG_ORIG_WIDTH]

def remove_padding_gt(padded_images: np.ndarray) -> np.ndarray:
    """
    Removes padding from multiple 512x512 ground truth masks
    Input Dim: (B, H, W)
    Output Dim: (B, H, W)
    """
    gts_no_padding = []
    N = padded_images.shape[0]
    for i in range(N):
        gt_no_padding = _remove_padding_gt(padded_images[i])
        gts_no_padding.append(gt_no_padding)
    gts_no_padding_array = np.stack(gts_no_padding, axis=0)
    return gts_no_padding_array

def _augment_triplet(
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

    # -------- Create transformers -----------

    # random crop + resize
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
    if random.random() < 0.5:
        angle = random.uniform(-15, -5)
    else:
        angle = random.uniform(5, 15)
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

    # color jitter + perspective
    transform_color = A.ColorJitter(
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3),
        saturation=(0.7, 1.3),
        hue=(-0.5, 0.5),
        p=1.0,
    )
    transform_pers = A.Compose([
        A.Perspective(p=1.0),
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH)
    ])

    # grayscale img + flip - no transformer

    # gaussian noise
    transform_noise = A.Compose([
        A.GaussNoise(std_range=(0.2, 0.3), p=1.0),
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH)
    ])

    # -------- Augmented versions -----------

    # Original (just pad to 512x512)
    orig_img = _pad_to_512(image)
    orig_scribble = _pad_to_512(scribble, pad_value=255)
    orig_gt = _pad_to_512(ground_truth)

    # random crop + resize
    crop = transform_crop(image=image, masks=[scribble, ground_truth])
    crop_img = _pad_to_512(crop['image'])
    crop_scribble = _pad_to_512(crop['masks'][0], pad_value=255)
    crop_gt = _pad_to_512(crop['masks'][1])

    # mirror
    flip = transform_flip(image=image, masks=[scribble, ground_truth])
    flip_img = _pad_to_512(flip['image'])
    flip_scribble = _pad_to_512(flip['masks'][0], pad_value=255)
    flip_gt = _pad_to_512(flip['masks'][1])

    # rotate
    rotate_img = transform_rotate_img(image=image)['image']
    rotate_scribble = transform_rotate_scrib(image=scribble)['image']
    rotate_gt = transform_rotate_img(image=ground_truth)['image']
    rotate_img = _pad_to_512(rotate_img)
    rotate_scribble = _pad_to_512(rotate_scribble, pad_value=255)
    rotate_gt = _pad_to_512(rotate_gt)

    # color jitter + perspective
    color_pers_img = transform_color(image=image)['image']
    augmented = transform_pers(image=color_pers_img, masks=[scribble, ground_truth])
    color_pers_img = augmented['image']
    color_pers_scribble = augmented['masks'][0]
    color_pers_gt = augmented['masks'][1]
    color_pers_img = _pad_to_512(color_pers_img)
    color_pers_scribble = _pad_to_512(color_pers_scribble, pad_value=255)
    color_pers_gt = _pad_to_512(color_pers_gt)

    # grayscale img + flip
    gray_img = cv2.cvtColor(flip_img, cv2.COLOR_BGR2GRAY)  # single channel
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # convert back to 3 channels to keep shape consistent
    gray_scribble = flip_scribble
    gray_gt = flip_gt

    # gaussian noise
    noise_img = transform_noise(image=image)['image']
    noise_img = _pad_to_512(noise_img)
    noise_scribble = _pad_to_512(scribble, pad_value=255)
    noise_gt = _pad_to_512(ground_truth)

    augmented_triplets = [
        (orig_img, orig_scribble, orig_gt),
        (crop_img, crop_scribble, crop_gt),
        (flip_img, flip_scribble, flip_gt),
        (rotate_img, rotate_scribble, rotate_gt),
        (color_pers_img, color_pers_scribble, color_pers_gt),
        (gray_img, gray_scribble, gray_gt),
        (noise_img, noise_scribble, noise_gt),
    ]

    return augmented_triplets

def _save_triplets(triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                  img_path: str, scrib_path: str, gt_path: str, palette: List[int], start_idx: int = 0) -> int:
    """
    Saves a list image, scribble and ground truth triplets.
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

def _create_data_paths(save_path):
    """
    Given a path, creates and returns child paths for images, scribbles and ground truths
    """
    img_path = os.path.join(save_path, "images")
    scrib_path = os.path.join(save_path, "scribbles")
    gt_path = os.path.join(save_path, "ground_truth")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(scrib_path):
        os.makedirs(scrib_path)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    return img_path, scrib_path, gt_path

def pad_and_save_data(images, scribbles, ground_truths, palette, save_path):
    """
    Saves validation data
    """
    img_path, scrib_path, gt_path = _create_data_paths(save_path)

    images_padded = np.stack([_pad_to_512(img) for img in images], axis=0)
    scribbles_padded = np.stack([_pad_to_512(scrib, pad_value=255) for scrib in scribbles], axis=0)
    ground_truths_padded = np.stack([_pad_to_512(gt) for gt in ground_truths], axis=0)
    padded_triplets = [(img, scrib, gt) for img, scrib, gt in zip(images_padded, scribbles_padded, ground_truths_padded)]
    _save_triplets(padded_triplets, img_path, scrib_path, gt_path, palette)

def augment_and_save_data(images, scribbles, ground_truths, palette, save_path):
    """
    Augments the training data according to the documentation and saves the resulting data
    """
    img_path, scrib_path, gt_path = _create_data_paths(save_path)

    # ensure we have equal amounts of everything
    N_images, *_ = images.shape
    N_scribbles, *_ = scribbles.shape
    N_ground_truths, *_ = ground_truths.shape
    assert N_images == N_scribbles == N_ground_truths

    next_id = 0 # image ID for saving
    for (img, scrib, gt) in zip(images, scribbles, ground_truths):
        augmented_triplets = _augment_triplet(img, scrib, gt)
        next_id = _save_triplets(augmented_triplets, img_path, scrib_path, gt_path, palette, next_id)