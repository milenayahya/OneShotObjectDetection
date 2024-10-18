from pycocotools import coco
import fiftyone as fo
import fiftyone.zoo as foz

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

# Set the directory where datasets will be downloaded
fo.config.dataset_zoo_dir = "C:\\Users\\cm03009\\Documents\\OneShotObjectDetection"

# Load one image per class to be used as query images
dataset_query = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections", "segmentations"],
    classes=coco_classes,
    max_samples=80, 
    shuffle=True,
)

session = fo.launch_app(dataset_query)

# Create a view of the dataset
view = dataset_query.view()

# Count the unique labels
unique_labels = view.count_values("detections.label")

# Print unique labels
print("Unique labels:", unique_labels)



for sample in dataset_query:
    print("Image path: ", sample.filepath)
    detections = sample["detections"].detections

    for detection in detections:
    
        bounding_box = detection.bounding_box  # This is a list of [x, y, width, height]
        label = detection.label
        print(label)

session.wait()
