import numpy as np
from data import augment_and_save
import torch

#augment_and_save("dataset/training", "dataset", "augmentations")
print(torch.__version__)
print(torch.cuda.is_available())