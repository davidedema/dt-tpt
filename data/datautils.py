import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations
import matplotlib.pyplot as plt 
from attention import get_attention_maps
import cv2

ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix

class AttentionGuidedAugmenter(object):
    def __init__(self, base_transform, preprocess, model_dino, n_views=2, augmix=False, severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.augmix = augmix
        self.severity = severity
        self.model_dino = model_dino
        if augmix:
            self.aug_list = augmentations.augmentations  
        else:
            self.aug_list = []

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        # find attention map for the image
        attention_maps = get_attention_maps(image, self.model_dino)
        # self.save_attentions(attention_maps)
        list_of_tensors = [torch.tensor(tensor) for tensor in attention_maps]
        attention_map = torch.mean(torch.stack(list_of_tensors), dim=0)
        attention_map = attention_map / attention_map.sum() #normalize the att
        cropped = []
        cropped.append(find_bounding_box(attention_map, image, 0.1))
        cropped.append(find_bounding_box(attention_map, image, 0.25))
        cropped.append(find_bounding_box(attention_map, image, 0.3))
        cropped.append(find_bounding_box(attention_map, image, 0.5))
        max_idx = torch.argmax(attention_map)

        focal_point = np.unravel_index(max_idx.item(), attention_map.shape) #select a focal point based on attention
        attention_views = [self.augment_view(image, focal_point=focal_point) for _ in range(int(self.n_views/2))]

        normal_views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(int(self.n_views/2))]
        return [image] + attention_views + normal_views + cropped

    def augment_view(self, x, focal_point):
        
        augmented_img = self.crop_with_focal_point(x, focal_point, (random.randint(80, 140),random.randint(80, 140)), variance=0.4) #aug according to focal_point
        augmented_img = self.random_horizontal_flip(augmented_img, 0.3) # flip it
        augmented_img = self.random_horizontal_flip(augmented_img, 0.3)
        return augmented_img

    def save_attentions(self,attentions):

        for j in range(6):
            fname = os.path.join("informations", "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            plt.close()

    #crop an image tensor centered around a focal point with some variance
    def crop_with_focal_point(self, image, focal_point, crop_size, variance=0.1):
        """
        Args:
            - image (torch.Tensor): The image tensor to be cropped (C, H, W).
            - focal_point (tuple): The (x, y) coordinates of the focal point.
            - crop_size (tuple): The (crop_height, crop_width) of the crop.
            - variance (float): The variance factor to apply random offset to the focal point.

        Returns: cropped_image (torch.Tensor): The cropped image tensor.
        """
        H, W = 224, 224
        crop_height, crop_width = crop_size
        focal_x, focal_y = focal_point[0], focal_point[1]
        resize = transforms.Resize((224,224), antialias=True)
        
        #apply variance to focal point
        offset_x = int((random.random() - 0.5) * 2 * variance * crop_width)
        offset_y = int((random.random() - 0.5) * 2 * variance * crop_height)
        focal_x = min(max(focal_x + offset_x, 0), W)
        focal_y = min(max(focal_y + offset_y, 0), H)
        
        #compute crop boundaries
        x1 = max(focal_x - crop_width // 2, 0)
        y1 = max(focal_y - crop_height // 2, 0)
        x2 = min(x1 + crop_width, W)
        y2 = min(y1 + crop_height, H)
        
        #adjust x1 and y1 if the crop size exceeds image dimensions
        x1 = x2 - crop_width if x2 - crop_width < 0 else x1
        y1 = y2 - crop_height if y2 - crop_height < 0 else y1
        
        #cropping
        cropped_image = image[:, y1:y2, x1:x2]
        cropped_image = resize(cropped_image)    

        return cropped_image
    
    def random_horizontal_flip(self, image, p_flip):
        prob = random.random()

        if prob > (1-p_flip):
            image = torch.flip(image, [2])
        else:
            return image

        return image

    def random_vertical_flip(self, image, p_flip):
        prob = random.random()

        if prob > (1-p_flip):
            image = torch.flip(image, [1])
        else:
            return image
            
        return image

class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views

def find_bounding_box(attention_tensor, original_tensor, percentage):
    """
    Find the bounding box on the original image that contains the specified percentage of the total attention.

    Args:
    attention_tensor (torch.Tensor): A 2D tensor representing the attention map.
    original_tensor (torch.Tensor): A 3D tensor (C, H, W) representing the RGB image.
    percentage (float): The percentage of total attention that the bounding box must contain.

    Returns:
    np.array: The original image with a bounding box drawn.
    tuple: Bounding box coordinates (x, y, width, height).
    """

    mean=[0.48145466, 0.4578275, 0.40821073]
    std=[0.26862954, 0.26130258, 0.27577711]

    # Unnormalization function
    unnormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    original_tensor = unnormalize(original_tensor)
    # Convert tensors to numpy for processing
    attention_map = attention_tensor.numpy()
    original_image = original_tensor.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    original_image = (original_image * 255).astype(np.uint8)  # Scale to 0-255 if necessary
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # Calculate the necessary threshold
    flat_map = attention_map.flatten()
    max_value = max(flat_map)
    # Create a binary mask where the attention values are above the threshold
    mask = (attention_map >= max_value*percentage).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original_image, (0, 0, 0, 0)  # No contour found

    # Assuming the largest contour is the region of interest
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw the bounding box on the original image
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)  # Scale and convert to uint8 if not already
    original_image = np.ascontiguousarray(original_image)
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(f"informations/crop_info_{percentage}.png", original_image)
    resize = transforms.Resize((224,224), antialias=True)
    cropped_image = resize(original_tensor[:, y:y+h, x:x+w])

    return cropped_image

