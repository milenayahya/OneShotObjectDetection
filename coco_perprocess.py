from pycocotools import coco
import fiftyone as fo
import fiftyone.zoo as foz
import cv2
from PIL import Image
import os
from config import PROJECT_BASE_PATH

coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]

ID2CLASS = {
    0: '__background__',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'backpack',
    26: 'umbrella',
    27: 'handbag',
    28: 'tie',
    29: 'suitcase',
    30: 'frisbee',
    31: 'skis',
    32: 'snowboard',
    33: 'sports ball',
    34: 'kite',
    35: 'baseball bat',
    36: 'baseball glove',
    37: 'skateboard',
    38: 'surfboard',
    39: 'tennis racket',
    40: 'bottle',
    41: 'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    51: 'broccoli',
    52: 'carrot',
    53: 'hot dog',
    54: 'pizza',
    55: 'donut',
    56: 'cake',
    57: 'chair',
    58: 'couch',
    59: 'potted plant',
    60: 'bed',
    61: 'dining table',
    62: 'toilet',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    69: 'microwave',
    70: 'oven',
    71: 'toaster',
    72: 'sink',
    73: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush'
}

CLASS2ID  = {
    '__background__': 0,
    'person': 1,
    'bicycle': 2,
    'car': 3,
    'motorcycle': 4,
    'airplane': 5,
    'bus': 6,
    'train': 7,
    'truck': 8,
    'boat': 9,
    'traffic light': 10,
    'fire hydrant': 11,
    'stop sign': 12,
    'parking meter': 13,
    'bench': 14,
    'bird': 15,
    'cat': 16,
    'dog': 17,
    'horse': 18,
    'sheep': 19,
    'cow': 20,
    'elephant': 21,
    'bear': 22,
    'zebra': 23,
    'giraffe': 24,
    'backpack': 25,
    'umbrella': 26,
    'handbag': 27,
    'tie': 28,
    'suitcase': 29,
    'frisbee': 30,
    'skis': 31,
    'snowboard': 32,
    'sports ball': 33,
    'kite': 34,
    'baseball bat': 35,
    'baseball glove': 36,
    'skateboard': 37,
    'surfboard': 38,
    'tennis racket': 39,
    'bottle': 40,
    'wine glass': 41,
    'cup': 42,
    'fork': 43,
    'knife': 44,
    'spoon': 45,
    'bowl': 46,
    'banana': 47,
    'apple': 48,
    'sandwich': 49,
    'orange': 50,
    'broccoli': 51,
    'carrot': 52,
    'hot dog': 53,
    'pizza': 54,
    'donut': 55,
    'cake': 56,
    'chair': 57,
    'couch': 58,
    'potted plant': 59,
    'bed': 60,
    'dining table': 61,
    'toilet': 62,
    'tv': 63,
    'laptop': 64,
    'mouse': 65,
    'remote': 66,
    'keyboard': 67,
    'cell phone': 68,
    'microwave': 69,
    'oven': 70,
    'toaster': 71,
    'sink': 72,
    'refrigerator': 73,
    'book': 74,
    'clock': 75,
    'vase': 76,
    'scissors': 77,
    'teddy bear': 78,
    'hair drier': 79,
    'toothbrush': 80
}

ID2COLOR = {
    0: (0, 0, 0),
    1: (220, 20, 60),
    2: (119, 11, 32),
    3: (0, 0, 142),
    4: (0, 0, 230),
    5: (128, 64, 128),
    6: (0, 60, 100),
    7: (0, 80, 100),
    8: (0, 0, 70),
    9: (250, 170, 30),
    10: (250, 170, 160),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (153, 153, 153),
    15: (180, 165, 180),
    16: (150, 100, 100),
    17: (150, 120, 90),
    18: (250, 170, 160),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
    34: (250, 170, 160),
    35: (230, 150, 140),
    36: (190, 153, 153),
    37: (153, 153, 153),
    38: (250, 170, 30),
    39: (220, 220, 0),
    40: (107, 142, 35),
    41: (152, 251, 152),
    42: (70, 130, 180),
    43: (220, 20, 60),
    44: (255, 0, 0),
    45: (0, 0, 142),
    46: (0, 0, 70),
    47: (0, 60, 100),
    48: (0, 0, 90),
    49: (0, 0, 110),
    50: (0, 80, 100),
    51: (0, 0, 230),
    52: (119, 11, 32),
    53: (250, 170, 160),
    54: (230, 150, 140),
    55: (190, 153, 153),
    56: (153, 153, 153),
    57: (250, 170, 30),
    58: (220, 220, 0),
    59: (107, 142, 35),
    60: (152, 251, 152),
    61: (70, 130, 180),
    62: (220, 20, 60),
    63: (255, 0, 0),
    64: (0, 0, 142),
    65: (0, 0, 70),
    66: (0, 60, 100),
    67: (0, 0, 90),
    68: (0, 0, 110),
    69: (0, 80, 100),
    70: (0, 0, 230),
    71: (119, 11, 32),
    72: (250, 170, 160),
    73: (230, 150, 140),
    74: (190, 153, 153),
    75: (153, 153, 153),
    76: (250, 170, 30),
    77: (220, 220, 0),
    78: (107, 142, 35),
    79: (152, 251, 152),
    80: (70, 130, 180)
}

def coco_to_left_upper_right_lower(coco_bbox, image_width, image_height):
    """
    Convert a COCO bounding box from normalized format to (left, upper, right, lower) format.

    Parameters:
    - coco_bbox: A list or tuple with the normalized bounding box [x, y, width, height].
    - image_width: The width of the image in pixels.
    - image_height: The height of the image in pixels.

    Returns:
    - A tuple (left, upper, right, lower) in pixel coordinates.
    """
    # Convert normalized coordinates to pixel coordinates
    x = coco_bbox[0] * image_width
    y = coco_bbox[1] * image_height
    width = coco_bbox[2] * image_width
    height = coco_bbox[3] * image_height

    # Calculate left, upper, right, and lower
    left = int(x)
    upper = int(y)
    right = int(x + width)
    lower = int(y + height)

    return (left, upper, right, lower)

def get_coco_queries(dir, nb_samples):

    # Set the directory where datasets will be downloaded
    fo.config.dataset_zoo_dir = dir

    # Load one image per class to be used as query images
    dataset_query = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections", "segmentations"],
        classes= coco_classes,
        max_samples=nb_samples, 
        shuffle=True,
        dataset_name=f"coco-2017-train-{nb_samples}"
    )

    session = fo.launch_app(dataset_query)

    class_occurences = [0]*(nb_classes+1)

    dir = os.path.join(PROJECT_BASE_PATH, "coco_query_objects")
    os.makedirs(dir, exist_ok=True)

    for sample in dataset_query:
        detections = sample["detections"].detections
        image = Image.open(sample.filepath)
        width = image.width
        height = image.height
        for detection in detections:

            bounding_box = detection.bounding_box  # This is a list of [x, y, width, height]
            label = detection.label
            id =CLASS2ID[label]
            class_occurences[id] += 1

            filename = os.path.join(dir, f"{id}_{label}_{class_occurences[id]}.JPEG")
            cropped_image = image.crop(coco_to_left_upper_right_lower(bounding_box,width,height))
            cropped_image.save(filename)

    session.wait()

def load_coco_images(dir, split):
   # Set the directory where datasets will be downloaded
    fo.config.dataset_zoo_dir = dir

    # Load one image per class to be used as query images
    dataset_query = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections", "segmentations"],
        shuffle=True,
        dataset_name= f"coco-2017-{split}"
    )

    session = fo.launch_app(dataset_query)
    session.wait()


if __name__ == "__main__":
        
    # Set the directory where datasets will be downloaded
    dir = "C:\\Users\\cm03009\\Documents\\OneShotObjectDetection"
    nb_classes= 80
    split = "test" #"validation" #"train"

    get_coco_queries(dir, nb_classes)
    #load_coco_images(dir, split)