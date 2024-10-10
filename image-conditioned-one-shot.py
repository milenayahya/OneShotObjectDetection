from transformers import Owlv2Processor, Owlv2ForObjectDetection
import requests
from PIL import Image, ImageDraw
import torch
import numpy as np
import os
import cv2 # type: ignore
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD # type: ignore
from matplotlib import rcParams
import matplotlib.pyplot as plt

def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

def load_image_batch(image_dir):
    images = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
    return images

if __name__ == "__main__":

    # Image-Conditioned Object Detection
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # Set figure size
    rcParams['figure.figsize'] = 11 ,8

    # Target image
    # target_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # target_image = Image.open(requests.get(target_url, stream=True).raw)
    # target_sizes = torch.Tensor([target_image.size[::-1]])
    image_path = 'test_images/dog2.jpg'
    target_image = cv2.imread(image_path)
    target_sizes = torch.Tensor([target_image.shape[0], target_image.shape[1]])  # height, width

    #Source image
    image_path = 'query_images/dog.jpeg'
    source_image = cv2.imread(image_path)
    #source_url = "http://images.cocodataset.org/val2017/000000058111.jpg"
    #source_image = Image.open(requests.get(source_url, stream=True).raw)

    # Process target and source image for the model
    inputs = processor(images=target_image, query_images=source_image, return_tensors="pt")
    # Print input names and shapes
    for key, val in inputs.items():
        print(f"{key}: {val.shape}")

    # Get predictions
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)

    target_pixel_values = processor(images=target_image, return_tensors="pt").pixel_values
    unnormalized_target_image = get_preprocessed_image(target_pixel_values)

    img = cv2.cvtColor(np.array(unnormalized_target_image), cv2.COLOR_BGR2RGB)
    outputs.logits = outputs.logits.cpu()
    outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()

    target_sizes = torch.Tensor([unnormalized_target_image.size[::-1]])
    # good values are 0.98, 0.95
    results = processor.post_process_image_guided_detection(outputs=outputs, threshold=0.99, nms_threshold=0.2, target_sizes=target_sizes)
    print(results)
    boxes, scores = results[0]["boxes"], results[0]["scores"]

    # Draw predicted bounding boxes
    for box, score in zip(boxes, scores):
        box = [int(i) for i in box.tolist()]

        img = cv2.rectangle(img, box[:2], box[2:], (255,0,0), 5)
        if box[3] + 25 > 768:
            y = box[3] - 10
        else:
            y = box[3] + 25

    plt.imshow(img[:,:,::-1])
    plt.show()