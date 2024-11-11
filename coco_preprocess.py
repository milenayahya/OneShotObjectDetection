from pycocotools.coco import COCO
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image 
import os
from config import PROJECT_BASE_PATH

coco_classes = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", 
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", 
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
    "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", 
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", 
    "hair drier", "toothbrush"
]


CLASS2ID = {
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
    'stop sign': 13,
    'parking meter': 14,
    'bench': 15,
    'bird': 16,
    'cat': 17,
    'dog': 18,
    'horse': 19,
    'sheep': 20,
    'cow': 21,
    'elephant': 22,
    'bear': 23,
    'zebra': 24,
    'giraffe': 25,
    'backpack': 27,
    'umbrella': 28,
    'handbag': 31,
    'tie': 32,
    'suitcase': 33,
    'frisbee': 34,
    'skis': 35,
    'snowboard': 36,
    'sports ball': 37,
    'kite': 38,
    'baseball bat': 39,
    'baseball glove': 40,
    'skateboard': 41,
    'surfboard': 42,
    'tennis racket': 43,
    'bottle': 44,
    'wine glass': 46,
    'cup': 47,
    'fork': 48,
    'knife': 49,
    'spoon': 50,
    'bowl': 51,
    'banana': 52,
    'apple': 53,
    'sandwich': 54,
    'orange': 55,
    'broccoli': 56,
    'carrot': 57,
    'hot dog': 58,
    'pizza': 59,
    'donut': 60,
    'cake': 61,
    'chair': 62,
    'couch': 63,
    'potted plant': 64,
    'bed': 65,
    'dining table': 67,
    'toilet': 70,
    'tv': 72,
    'laptop': 73,
    'mouse': 74,
    'remote': 75,
    'keyboard': 76,
    'cell phone': 77,
    'microwave': 78,
    'oven': 79,
    'toaster': 80,
    'sink': 81,
    'refrigerator': 82,
    'book': 84,
    'clock': 85,
    'vase': 86,
    'scissors': 87,
    'teddy bear': 88,
    'hair drier': 89,
    'toothbrush': 90
}

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
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

ID2COLOR = {
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
    27: (0.0, 0.0, 70 / 255),
    28: (0.0, 60 / 255, 100 / 255),
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
    67: (0.0, 0.0, 90 / 255),
    70: (0.0, 0.0, 230 / 255),
    72: (250 / 255, 170 / 255, 160 / 255),
    73: (230 / 255, 150 / 255, 140 / 255),
    74: (190 / 255, 153 / 255, 153 / 255),
    75: (153 / 255, 153 / 255, 153 / 255),
    76: (250 / 255, 170 / 255, 30 / 255),
    77: (220 / 255, 220 / 255, 0.0),
    78: (107 / 255, 142 / 255, 35 / 255),
    79: (152 / 255, 251 / 255, 152 / 255),
    80: (70 / 255, 130 / 255, 180 / 255),
    81: (220 / 255, 20 / 255, 60 / 255),
    82: (255 / 255, 0.0, 0.0),
    84: (0.0, 0.0, 142 / 255),
    85: (0.0, 0.0, 70 / 255),
    86: (0.0, 0.0, 90 / 255),
    87: (0.0, 0.0, 110 / 255),
    88: (0.0, 80 / 255, 100 / 255),
    89: (0.0, 60 / 255, 100 / 255),
    90: (0.0, 0.0, 230 / 255),
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
       # classes= coco_classes,
        max_samples=nb_samples, 
        shuffle=True,
        dataset_name=f"coco-2017-train-{nb_samples}"
    )

    session = fo.launch_app(dataset_query)

    
    class_occurences = [0]*(90+1)

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
    nb_samples= 5000
    split = "test" #"validation" #"train"
    k = 5
    min_size = 5000

    load_coco_train(dir, nb_samples)
    #load_coco_test_val(dir, split)

   # coco_create_queries(labels_file, image_dir, min_size, k)
    
