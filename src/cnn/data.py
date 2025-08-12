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
    gc_images: np.ndarray,
    knn_images: np.ndarray,
    scribbles: np.ndarray,
    ground_truth: np.ndarray,
    validation_size: float,
    random_seed: int | None = None
):
    """
    Split dataset into train and test subsets.

    Args:
        gc_images (np.ndarray): (N, H, W, 3) gc images
        knn_images (np.ndarray): (N, H, W) knn images
        scribbles (np.ndarray): (N, H, W) scribble masks
        ground_truth (np.ndarray): (N, H, W) ground truth masks
        validation_size (float): fraction of data used for testing
        random_seed (int or None): seed for reproducibility

    Returns:
        (train_gc, train_knn, train_scribbles, train_gt),
        (test_gc, test_knn, test_scribbles, test_gt)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    N = gc_images.shape[0]
    assert knn_images.shape[0] == N
    assert scribbles.shape[0] == N
    assert ground_truth.shape[0] == N

    indices = np.arange(N)
    np.random.shuffle(indices)

    split_idx = int(N * (1 - validation_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_gc = gc_images[train_idx].copy()
    train_knn = knn_images[train_idx].copy()
    train_scribbles = scribbles[train_idx].copy()
    train_gt = ground_truth[train_idx].copy()

    test_gc = gc_images[test_idx].copy()
    test_knn = knn_images[test_idx].copy()
    test_scribbles = scribbles[test_idx].copy()
    test_gt = ground_truth[test_idx].copy()

    return (
        (train_gc, train_knn, train_scribbles, train_gt),
        (test_gc, test_knn, test_scribbles, test_gt),
    )

class SegmentationDataset(Dataset):
    def __init__(self, images, scribbles, masks, transform=None):
        """
        Parameters:
            images (np.ndarray): (N, H, W) knn images
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
        image = self.images[idx]           # shape: H x W (knn)
        scribble = self.scribbles[idx]     # shape: H x W
        mask = self.masks[idx]             # shape: H x W

        # Convert to tensors and normalize
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0       # shape: 1 x H x W
        scribble = torch.from_numpy(scribble).unsqueeze(0).float() / 255.0 # shape: 1 x H x W
        mask = torch.from_numpy(mask).unsqueeze(0).float()                 # shape: 1 x H x W

        # Combine image and scribble into 2-channel input tensor
        input_tensor = torch.cat([image, scribble], dim=0)  # shape: 2 x H x W

        if self.transform:
            input_tensor, mask = self.transform(input_tensor, mask)

        return input_tensor, mask

class SegmentationDatasetExt(Dataset):
    def __init__(self, gc_images, knn_images, scribbles, masks, transform=None):
        """
        Parameters:
            gc_images (np.ndarray): (N, H, W, 3) gc images
            knn_images (np.ndarray): (N, H, W) knn images
            scribbles (np.ndarray): (N, H, W) scribble masks
            masks (np.ndarray): (N, H, W) ground truth masks
        """
        self.gc_images = gc_images
        self.knn_images = knn_images
        self.scribbles = scribbles
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.gc_images)

    def __getitem__(self, idx):
        gc_image = self.gc_images[idx]                # shape: H x W x 3 (gc)
        knn_image = self.knn_images[idx]    # shape: H x W (knn)
        scribble = self.scribbles[idx]                  # shape: H x W
        mask = self.masks[idx]                          # shape: H x W

        # Convert to tensors and normalize
        gc_image = torch.from_numpy(gc_image).unsqueeze(0).float() / 255.0             # shape: 3 x H x W
        knn_image = torch.from_numpy(knn_image).unsqueeze(0).float() / 255.0  # shape: 1 x H x W
        scribble = torch.from_numpy(scribble).unsqueeze(0).float() / 255.0                      # shape: 1 x H x W
        mask = torch.from_numpy(mask).unsqueeze(0).float()                                      # shape: 1 x H x W

        # Combine image and scribble into 2-channel input tensor
        input_tensor = torch.cat([gc_image, knn_image, scribble], dim=0)  # shape: 5 x H x W

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

    if img.ndim == 2:  # knn or mask
        pad_val = (pad_value,)
    elif img.ndim == 3 and img.shape[2] == 3:  # gc image
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

def _augment_quadruplet_v2(
    gc_image: np.ndarray,
    knn_image: np.ndarray,
    scribble: np.ndarray,
    ground_truth: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    def _pad_all(gc, knn, scrib, gt):
        return (
            _pad_to_512(gc),
            _pad_to_512(knn),
            _pad_to_512(scrib, pad_value=255),
            _pad_to_512(gt),
        )

    results = [_pad_all(gc_image, knn_image, scribble, ground_truth)]

    def make_geometric_transform():
        return A.Compose([
            A.RandomResizedCrop(
                size=(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.4
            ),
            A.OneOf([
                A.GridDistortion(
                    num_steps=5,  # int
                    distort_limit=(-0.3, 0.3),  # ScaleFloatType
                    interpolation=1,  # <class 'int'>
                    normalized=True,  # bool
                    mask_interpolation=0,  # int
                    border_mode=cv2.BORDER_REPLICATE,
                    p=1.0,  # float
                ),
                A.Perspective(scale=(0.05, 0.1), p=1.0),
            ], p=0.8),
            A.Rotate(
                limit=(-15, 15),
                border_mode=cv2.BORDER_REPLICATE,
                p=0.4
            ),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], p=0.7),
        ])

    def make_color_transform():
        return A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=2.0, p=1.0),
            ], p=0.3),
            A.GaussNoise(std_range=(0.1, 0.3), p=0.2),
        ])

    for _ in range(3):
        geo_transform = make_geometric_transform()
        color_transform = make_color_transform()

        geo_aug = geo_transform(image=gc_image, masks=[knn_image, scribble, ground_truth])
        # Apply color transforms only to the geo-augmented gc
        color_aug = color_transform(image=geo_aug['image'])

        results.append(_pad_all(
            color_aug['image'],
            geo_aug['masks'][0],
            geo_aug['masks'][1],
            geo_aug['masks'][2],
        ))

    return results

def _augment_quadruplet_v1(
    mask_gc: np.ndarray,
    mask_knn: np.ndarray,
    scribble: np.ndarray,
    ground_truth: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Given a quadruplet (gc image, knn image, scribble, ground_truth),
    apply augmentations with identical geometric transforms on all four.

    Returns a list of quadruplets
    (gc_aug, knn_aug, scribble_aug, ground_truth_aug),
    each padded to 512×512.
    """

    h, w = mask_gc.shape[:2]

    # -------- Create transformers -----------

    # Random crop + resize
    transform_crop = A.Compose([
        A.RandomResizedCrop(
            size=(int(h * 0.8), int(w * 0.8)),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),
    ])

    # Horizontal flip
    transform_flip = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),
    ])

    # Random rotation angle
    if random.random() < 0.5:
        angle = random.uniform(-15, -5)
    else:
        angle = random.uniform(5, 15)

    transform_rotate_masks_gt = A.Compose([
        A.Rotate(limit=(angle, angle), border_mode=0, fill=0, p=1.0),
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),
    ])
    transform_rotate_scrib = A.Compose([
        A.Rotate(limit=(angle, angle), border_mode=0, fill=255, p=1.0),
        A.Resize(IMG_ORIG_HEIGHT, IMG_ORIG_WIDTH),
    ])

    # -------- Apply augmentations -----------

    # Original
    orig_gc = _pad_to_512(mask_gc)
    orig_knn = _pad_to_512(mask_knn)
    orig_scribble = _pad_to_512(scribble, pad_value=255)
    orig_gt = _pad_to_512(ground_truth)

    # Crop
    crop = transform_crop(image=mask_gc, masks=[mask_knn, scribble, ground_truth])
    crop_gc = _pad_to_512(crop['image'])
    crop_knn = _pad_to_512(crop['masks'][0])
    crop_scribble = _pad_to_512(crop['masks'][1], pad_value=255)
    crop_gt = _pad_to_512(crop['masks'][2])

    # Flip
    flip = transform_flip(image=mask_gc, masks=[mask_knn, scribble, ground_truth])
    flip_gc = _pad_to_512(flip['image'])
    flip_knn = _pad_to_512(flip['masks'][0])
    flip_scribble = _pad_to_512(flip['masks'][1], pad_value=255)
    flip_gt = _pad_to_512(flip['masks'][2])

    # Rotate
    rotate_gc = _pad_to_512(transform_rotate_masks_gt(image=mask_gc)['image'])
    rotate_knn = _pad_to_512(transform_rotate_masks_gt(image=mask_knn)['image'])
    rotate_scribble = _pad_to_512(transform_rotate_scrib(image=scribble)['image'], pad_value=255)
    rotate_gt = _pad_to_512(transform_rotate_masks_gt(image=ground_truth)['image'])

    augmented_quadruplets = [
        (orig_gc, orig_knn, orig_scribble, orig_gt),
        (crop_gc, crop_knn, crop_scribble, crop_gt),
        (flip_gc, flip_knn, flip_scribble, flip_gt),
        (rotate_gc, rotate_knn, rotate_scribble, rotate_gt)
    ]

    return augmented_quadruplets


def _save_quadruplets(quadruplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                      gc_path: str, knn_path: str, scrib_path: str, gt_path: str,
                      palette: List[int], start_idx: int = 0) -> int:
    """
    Saves a list of (gc_image, knn_image, scribble, ground_truth) quadruplets.

    Parameters:
        quadruplets: list of (H,W,3), (H,W), (H,W), (H,W) arrays
        gc_path: directory for gc .jpg images
        knn_path: directory for knn .png images
        scrib_path: directory for scribble masks
        gt_path: directory for ground truth masks
        palette: list defining the palette for the GT mask
        start_idx: starting index for file naming
    Returns:
        next index after the last saved sample
    """
    idx = start_idx
    for quad in quadruplets:
        gc_image, knn_image, scribble, gt_mask = quad

        filename_img = f"{idx:04d}.jpg"
        filename_knn = f"{idx:04d}.png"
        filename_mask = f"{idx:04d}.png"

        # Convert to PIL
        gc_pil = Image.fromarray(gc_image)
        knn_pil = Image.fromarray(knn_image.astype(np.uint8))
        scrib_pil = Image.fromarray(scribble.astype(np.uint8))
        gt_pil = Image.fromarray(gt_mask.astype(np.uint8), mode='P')
        gt_pil.putpalette(palette)

        # Save files
        gc_pil.save(os.path.join(gc_path, filename_img), quality=95)
        knn_pil.save(os.path.join(knn_path, filename_knn))
        scrib_pil.save(os.path.join(scrib_path, filename_mask))
        gt_pil.save(os.path.join(gt_path, filename_mask))

        idx += 1
    return idx

def _create_data_paths(save_path):
    """
    Given a path, creates and returns child paths for mask images, scribbles and ground truths
    """
    masks_gc_path = os.path.join(save_path, "masks_gc")
    masks_knn_path = os.path.join(save_path, "masks_knn")
    scrib_path = os.path.join(save_path, "scribbles")
    gt_path = os.path.join(save_path, "ground_truth")
    if not os.path.exists(masks_gc_path):
        os.makedirs(masks_gc_path)
    if not os.path.exists(masks_knn_path):
        os.makedirs(masks_knn_path)
    if not os.path.exists(scrib_path):
        os.makedirs(scrib_path)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    return masks_gc_path, masks_knn_path, scrib_path, gt_path

def pad_and_save_data(gc_images, knn_images, scribbles, ground_truths, palette, save_path):
    """
    Saves padded data
    """
    gc_img_path, knn_img_path, scrib_path, gt_path = _create_data_paths(save_path)

    gc_images_padded = np.stack([_pad_to_512(img) for img in gc_images], axis=0)
    knn_images_padded = np.stack([_pad_to_512(img) for img in knn_images], axis=0)
    scribbles_padded = np.stack([_pad_to_512(scrib, pad_value=255) for scrib in scribbles], axis=0)
    ground_truths_padded = np.stack([_pad_to_512(gt) for gt in ground_truths], axis=0)
    padded_quadruplets = [(gc_img, knn_img, scrib, gt) for gc_img, knn_img, scrib, gt in zip(gc_images_padded, knn_images_padded, scribbles_padded, ground_truths_padded)]
    _save_quadruplets(padded_quadruplets, gc_img_path, knn_img_path, scrib_path, gt_path, palette)

def pad_and_return_data(gc_images, knn_images, scribbles, ground_truths):
    """
    Pads gc, knn, scribble, and ground truth arrays to 512x512.
    Returns padded data arrays in the same order.
    """
    gc_padded = np.stack([_pad_to_512(img) for img in gc_images], axis=0)
    knn_padded = np.stack([_pad_to_512(img) for img in knn_images], axis=0)
    scribbles_padded = np.stack([_pad_to_512(scrib, pad_value=255) for scrib in scribbles], axis=0)
    ground_truths_padded = np.stack([_pad_to_512(gt) for gt in ground_truths], axis=0)
    return gc_padded, knn_padded, scribbles_padded, ground_truths_padded

def augment_and_save_data(masks_gc, masks_knn, scribbles, ground_truths, palette, save_path):
    """
    Augments the training data according to the documentation and saves the resulting data
    """
    masks_gc_path, masks_knn_path, scrib_path, gt_path = _create_data_paths(save_path)

    # ensure we have equal amounts of everything
    N_gc_images, *_ = masks_gc.shape
    N_knn_images, *_ = masks_knn.shape
    N_scribbles, *_ = scribbles.shape
    N_ground_truths, *_ = ground_truths.shape
    assert N_gc_images == N_knn_images == N_scribbles == N_ground_truths

    next_id = 0 # image ID for saving
    for (gc_img, knn_img, scrib, gt) in zip(masks_gc, masks_knn, scribbles, ground_truths):
        augmented_quadruplets = _augment_quadruplet_v1(gc_img, knn_img, scrib, gt)
        next_id = _save_quadruplets(augmented_quadruplets, masks_gc_path, masks_knn_path, scrib_path, gt_path, palette, next_id)