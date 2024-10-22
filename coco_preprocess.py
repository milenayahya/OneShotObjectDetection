from pycocotools.coco import COCO
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
    0: (0.0, 0.0, 0.0),
    1: (220 / 255, 20 / 255, 60 / 255),
    2: (119 / 255, 11 / 255, 32 / 255),
    3: (0.0, 0.0, 142 / 255),
    4: (0.0, 0.0, 230 / 255),
    5: (128 / 255, 64 / 255, 128 / 255),
    6: (0.0, 60 / 255, 100 / 255),
    7: (0.0, 80 / 255, 100 / 255),
    8: (0.0, 0.0, 70 / 255),
    9: (250 / 255, 170 / 255, 30 / 255),
    10: (250 / 255, 170 / 255, 160 / 255),
    11: (70 / 255, 70 / 255, 70 / 255),
    12: (102 / 255, 102 / 255, 156 / 255),
    13: (190 / 255, 153 / 255, 153 / 255),
    14: (153 / 255, 153 / 255, 153 / 255),
    15: (180 / 255, 165 / 255, 180 / 255),
    16: (150 / 255, 100 / 255, 100 / 255),
    17: (150 / 255, 120 / 255, 90 / 255),
    18: (250 / 255, 170 / 255, 160 / 255),
    19: (250 / 255, 170 / 255, 30 / 255),
    20: (220 / 255, 220 / 255, 0.0),
    21: (107 / 255, 142 / 255, 35 / 255),
    22: (152 / 255, 251 / 255, 152 / 255),
    23: (70 / 255, 130 / 255, 180 / 255),
    24: (220 / 255, 20 / 255, 60 / 255),
    25: (255 / 255, 0.0, 0.0),
    26: (0.0, 0.0, 142 / 255),
    27: (0.0, 0.0, 70 / 255),
    28: (0.0, 60 / 255, 100 / 255),
    29: (0.0, 0.0, 90 / 255),
    30: (0.0, 0.0, 110 / 255),
    31: (0.0, 80 / 255, 100 / 255),
    32: (0.0, 0.0, 230 / 255),
    33: (119 / 255, 11 / 255, 32 / 255),
    34: (250 / 255, 170 / 255, 160 / 255),
    35: (230 / 255, 150 / 255, 140 / 255),
    36: (190 / 255, 153 / 255, 153 / 255),
    37: (153 / 255, 153 / 255, 153 / 255),
    38: (250 / 255, 170 / 255, 30 / 255),
    39: (220 / 255, 220 / 255, 0.0),
    40: (107 / 255, 142 / 255, 35 / 255),
    41: (152 / 255, 251 / 255, 152 / 255),
    42: (70 / 255, 130 / 255, 180 / 255),
    43: (220 / 255, 20 / 255, 60 / 255),
    44: (255 / 255, 0.0, 0.0),
    45: (0.0, 0.0, 142 / 255),
    46: (0.0, 0.0, 70 / 255),
    47: (0.0, 60 / 255, 100 / 255),
    48: (0.0, 0.0, 90 / 255),
    49: (0.0, 0.0, 110 / 255),
    50: (0.0, 80 / 255, 100 / 255),
    51: (0.0, 0.0, 230 / 255),
    52: (119 / 255, 11 / 255, 32 / 255),
    53: (250 / 255, 170 / 255, 160 / 255),
    54: (230 / 255, 150 / 255, 140 / 255),
    55: (190 / 255, 153 / 255, 153 / 255),
    56: (153 / 255, 153 / 255, 153 / 255),
    57: (250 / 255, 170 / 255, 30 / 255),
    58: (220 / 255, 220 / 255, 0.0),
    59: (107 / 255, 142 / 255, 35 / 255),
    60: (152 / 255, 251 / 255, 152 / 255),
    61: (70 / 255, 130 / 255, 180 / 255),
    62: (220 / 255, 20 / 255, 60 / 255),
    63: (255 / 255, 0.0, 0.0),
    64: (0.0, 0.0, 142 / 255),
    65: (0.0, 0.0, 70 / 255),
    66: (0.0, 60 / 255, 100 / 255),
    67: (0.0, 0.0, 90 / 255),
    68: (0.0, 0.0, 110 / 255),
    69: (0.0, 80 / 255, 100 / 255),
    70: (0.0, 0.0, 230 / 255),
    71: (119 / 255, 11 / 255, 32 / 255),
    72: (250 / 255, 170 / 255, 160 / 255),
    73: (230 / 255, 150 / 255, 140 / 255),
    74: (190 / 255, 153 / 255, 153 / 255),
    75: (153 / 255, 153 / 255, 153 / 255),
    76: (250 / 255, 170 / 255, 30 / 255),
    77: (220 / 255, 220 / 255, 0.0),
    78: (107 / 255, 142 / 255, 35 / 255),
    79: (152 / 255, 251 / 255, 152 / 255),
    80: (70 / 255, 130 / 255, 180 / 255)
}


def coco_to_left_upper_right_lower(coco_bbox, image_width=None, image_height=None):
    """
    Convert a COCO bounding box from normalized format to (left, upper, right, lower) format.

    Parameters:
    - coco_bbox: A list or tuple with the normalized bounding box [x, y, width, height].
    - image_width: The width of the image in pixels.
    - image_height: The height of the image in pixels.

    Returns:
    - A tuple (left, upper, right, lower) in pixel coordinates.
    """

    if image_height or image_width is not None:
    # Convert normalized coordinates to pixel coordinates
        x = coco_bbox[0] * image_width
        y = coco_bbox[1] * image_height
        width = coco_bbox[2] * image_width
        height = coco_bbox[3] * image_height
    else:
        x, y, width, height = coco_bbox 
    # Calculate left, upper, right, and lower
    left = int(x)
    upper = int(y)
    right = int(x + width)
    lower = int(y + height)

    return (left, upper, right, lower)

def crop_image(dir, image_path, ann, cat, i):
    image = Image.open(image_path)
    filename = os.path.join(dir, f"{CLASS2ID[cat['name']]}_{cat['name']}_{i+1}.JPEG")
    cropped_im = image.crop(coco_to_left_upper_right_lower(ann['bbox']))
    print(cropped_im.format)
    cropped_im.save(filename)
    image.close()

def coco_create_queries(labels_file, image_dir, min_size, num_objects_per_class):
    coco = COCO(labels_file)
    cats = coco.loadCats(coco.getCatIds())

    dir = os.path.join(PROJECT_BASE_PATH, "coco_query_objects_filtered")
    os.makedirs(dir, exist_ok=True)

    for cat in cats:
        ann_ids = coco.getAnnIds(catIds=[cat['id']], areaRng=[min_size, float('inf')])
        anns = coco.loadAnns(ann_ids)
        for i, ann in enumerate(anns):
            image_info = coco.loadImgs(ann['image_id'])[0]
            image_path = os.path.join(dir, image_dir, image_info['file_name'])
            crop_image(dir, image_path, ann, cat, i)
            if i == num_objects_per_class - 1:
                break

def load_coco_train(dir, nb_samples):

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

    
    class_occurences = [0]*(80+1)

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

def load_coco_test_val(dir, split):
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
    image_dir = "C:\\Users\\cm03009\\Documents\\OneShotObjectDetection\\coco-2017\\train\\data"
    labels_file = "C:\\Users\\cm03009\\Documents\\OneShotObjectDetection\\coco-2017\\train\\labels.json"
    nb_samples= 1000
    split = "test" #"validation" #"train"
    k = 5
    min_size = 5000

    #load_coco_train(dir, nb_samples)
    #load_coco_test_val(dir, split)

    coco_create_queries(labels_file, image_dir, min_size, k)
    
    