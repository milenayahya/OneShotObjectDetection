import torch
from PIL import Image
import os
import json
import numpy as np
from collections import defaultdict
import random
from pycocotools.coco import COCO
import tensorflow as tf
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD  
from typing import Any, Literal, Optional, List, Tuple, Union
from config import results_dir
from torchvision.ops import masks_to_boxes


def get_preprocessed_image(pixel_values: torch.Tensor) -> Image.Image:
    pixel_values = pixel_values.squeeze().cpu().numpy()
    unnormalized_image = (
        pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]
    ) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = unnormalized_image.squeeze()
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

def convert_to_cxcywh_format(boxes, img_indices, batch, dir, per_image, normalize=True):
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h).

    Parameters:
    - boxes: list of PyTorch tensors, each tensor of shape (4,) representing a bounding box in the format (x1, y1, x2, y2).

    Returns:
    - A list of PyTorch tensors, each of shape (4,) in the format (cx, cy, w, h).

    THIS FUNCTION NORMALIZES THE COORDINATES
    """
    cxcywh_boxes = []
    
    for id, box in enumerate(boxes):
        
        if per_image:
            im = Image.open(dir)
        else:
            im_id = img_indices[id] + batch
            im = open_image_by_index(dir, im_id)
        width = im.width
        height = im.height

        x1, y1, x2, y2 = box
        # Calculate cx, cy, w, h
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        if normalize:
            # Normalize the coordinates
            cx /= width
            cy /= height
            w /= width
            h /= height

        # Ensure the output is on the same device as the input box
        cxcywh_boxes.append(torch.tensor([cx, cy, w, h], device=box.device))

    cxcywh_boxes = torch.stack(cxcywh_boxes)
    return cxcywh_boxes

def convert_from_x1y1x2y2_to_coco(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (x, y, width, height).

    Parameters:
    - boxes: list of PyTorch tensors, each tensor of shape (4,) representing a bounding box in the format (x1, y1, x2, y2).

    Returns:
    - A list of PyTorch tensors, each of shape (4,) in the format (x, y, width, height).
    """
    coco_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box

        # Calculate x, y, width, height
        x = x1
        y = y1
        width = x2 - x1
        height = y2 - y1

        # Ensure the output is on the same device as the input box
        if isinstance(box, torch.Tensor):
            device = box.device
        else:
            device = torch.device('cpu')
        coco_boxes.append(torch.tensor([x, y, width, height], device=device))

    if len(coco_boxes) > 0:
        coco_boxes = torch.stack(coco_boxes)
    return coco_boxes

def convert_boxes(cxcywh_boxes, img_indices, batch, dir, per_image):
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Parameters:
    - cxcywh_boxes: list of PyTorch tensors, each tensor of shape (4,) representing a bounding box in the format (cx, cy, w, h).

    Returns:
    - A list of PyTorch tensors, each of shape (4,) in the format (x1, y1, x2, y2).
    """


    converted_boxes = []
    
    for id, box in enumerate(cxcywh_boxes):

        if per_image:
            im = Image.open(dir)
        else:
            im_id = img_indices[id] + batch
            im = open_image_by_index(dir, im_id)

        width = im.width
        height = im.height
        cx, cy, w, h = box
        
        cx_pixel = cx * width
        cy_pixel = cy * height
        w_pixel = w * width
        h_pixel = h * height
        
        # Calculate x1, y1, x2, y2
        x1 = cx_pixel - w_pixel / 2
        y1 = cy_pixel - h_pixel / 2
        x2 = cx_pixel + w_pixel / 2
        y2 = cy_pixel + h_pixel / 2
        
       # Ensure the output is on the same device as the input box
        converted_boxes.append(torch.tensor([x1, y1, x2, y2], device=box.device))

    converted_boxes = torch.stack(converted_boxes)

    return converted_boxes

def open_image_by_index(dir, index):
    """
    Open an image from a directory by its index.

    """

    files = os.listdir(dir)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'JPEG'))]
    if index < 0 or index >= len(images):
        raise IndexError("Image index out of range")
    image_filename = images[index]
    image_path = os.path.join(dir, image_filename)
    image = Image.open(image_path)

    return image

def get_filename_by_index(dir, index):
    """
    Get the filename of an image from a directory by its index.
    
    """
    files = os.listdir(dir)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'JPEG'))]
    if index < 0 or index >= len(images):
        raise IndexError("Image index out of range")
    image_filename = images[index]

    return image_filename

def read_results(filepath, random_selection=None):
    """
    Read results from a JSON file and randomly select x% of image IDs if random_selection is True.

    Parameters:
    - filepath: Path to the JSON file.
    - random_selection: float which represents the percentage of image IDs to randomly select.

    Returns:
    - A dictionary with image data.
    """
    with open(filepath, 'r') as f:
        results = [json.loads(line) for line in f]

    image_data = defaultdict(lambda: {'bboxes': [], 'scores': [], 'categories': []})

    if random_selection is not None:
        all_image_ids = list(set(result['image_id'] for result in results))
        selected_image_ids = random.sample(all_image_ids, int(len(all_image_ids) * random_selection))
    else:
        selected_image_ids = None

    for result in results:
        image_id = result['image_id']
        if random_selection is None or image_id in selected_image_ids:
            image_data[image_id]['bboxes'].append(result['bbox'])
            image_data[image_id]['scores'].append(result['score'])
            image_data[image_id]['categories'].append(result['category_id'])

    # convert to a regular dict 
    image_data = dict(image_data)
    return image_data

def nms_tf(
        bboxes: tf.Tensor,
        scores: tf.Tensor, 
        classes: List[str], 
        threshold: float
)-> Tuple[tf.Tensor, tf.Tensor, List[str]]:
    
    """

    Perform non-maximum suppression on a set of bounding boxes.
    Parameters: 
    - bboxes: A tensor of shape (N, 4) containing the bounding boxes in (x1, y1, x2, y2) format.
    - scores: A tensor of shape (N,) containing the objectness scores for each bounding box.
    - classes: A list of strings containing the class labels for each bounding box.
    - threshold: The IoU threshold to use for NMS.
    Returns:
    - A tuple containing the filtered bounding boxes, scores, and classes.

    """
    #logger.info("Applying NMS with threshold: %f on %d boxes", threshold, len(bboxes))
    bboxes = tf.cast(bboxes, dtype=tf.float32)
    x_min, y_min, x_max, y_max = tf.unstack(bboxes, axis=1)
    bboxes = tf.stack([y_min, x_min, y_max, x_max], axis=1)
    bbox_indices = tf.image.non_max_suppression(
        bboxes, scores, 100, iou_threshold=threshold
    )
    filtered_bboxes = tf.gather(bboxes, bbox_indices)
    scores = tf.gather(scores, bbox_indices)
    y_min, x_min, y_max, x_max = tf.unstack(filtered_bboxes, axis=1)
    filtered_bboxes = tf.stack([x_min, y_min, x_max, y_max], axis=1)
    filtered_classes = np.array(classes)[bbox_indices.numpy()]
    #logger.info("NMS completed. Kept %d boxes.", len(filtered_bboxes))
    return filtered_bboxes, scores, filtered_classes



def save_results(coco_results, options, per_image, im_id):
    """
    Save the results to a JSON file.
    
    """

    if per_image:
        file = os.path.join(results_dir, f"results_{im_id}.json")
        with open(file, "w") as f:
            for result in coco_results:
                f.write(json.dumps(result) + '\n')
    else:
        file = os.path.join(results_dir, f"results_{options.comment}.json")
        with open(file, "a") as f:
            for result in coco_results:
                f.write(json.dumps(result) + '\n')
    



def create_image_id_mapping(labels_dir):
    """
    Create a mapping from image file names to COCO image IDs

    """
    coco = COCO(labels_dir)
    coco_img_ids = coco.getImgIds()
    
    coco_imgs = coco.loadImgs(coco_img_ids)
    mapping = {img['file_name']: img['id'] for img in coco_imgs}
    return mapping

def map_coco_ids(mapping, filename):
    """
    Retrieve the COCO image ID for a given image file name.

    """
    return mapping.get(filename, None)
    
def map_coco_filenames(mapping, img_id):
    """
    Retrieve the image file name for a given COCO image ID.

    """
    for key, val in mapping.items():
        if val == img_id:
            return key
    return None


def convert_masks_to_boxes(instances_semantic):
    """
    Convert instance semantic masks to bounding boxes.
    Bounding boxes are in x1, y1, x2, y2 format.
    """

    unique_values = np.unique(instances_semantic)
    masks = []
    for mask_id in unique_values:
        if mask_id == 0:
            continue
        mask = (instances_semantic == mask_id).astype(np.uint8)
        masks.append(mask)

    # Stack the binary masks to create a tensor of shape [N, H, W]
    if len(masks)>0:
        masks_tensor = torch.tensor(np.stack(masks, axis=0))
    else:
        masks_tensor = torch.tensor([])
        print("No valid masks found in instances_semantic")

    bboxes = masks_to_boxes(masks_tensor)
    return bboxes
