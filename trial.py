from transformers import Owlv2Processor, Owlv2ForObjectDetection
import requests
from PIL import Image, ImageDraw
import torch
import numpy as np
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    '''
    ## Text-Conditioned Object Detwection
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    texts = [['a cat', 'remote control']]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    for k,v in inputs.items():
        print(k,v.shape)


    with torch.no_grad():
        outputs = model(**inputs)

    unnormalized_image = get_preprocessed_image(inputs.pixel_values)
    unnormalized_image.show()
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    # visualize 
    visualized_image = unnormalized_image.copy()
    draw = ImageDraw.Draw(visualized_image)

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        x1, y1, x2, y2 = tuple(box)
        draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
        draw.text(xy=(x1, y1), text=text[label])

    visualized_image.show()
    '''
    # Test image
    image_path = 'test_images/dog1.jpg'
    target_image = cv2.imread(image_path)
    target_sizes = torch.Tensor([target_image.shape[0], target_image.shape[1]])  # height, width

    #Source image
    image_path = 'query_images/dog.jpeg'
    source_image = cv2.imread(image_path)

    # Image-Conditioned Object Detection
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # Set figure size
    rcParams['figure.figsize'] = 11 ,8

    '''
    # Target image
    target_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    target_image = Image.open(requests.get(target_url, stream=True).raw)
    target_sizes = torch.Tensor([target_image.size[::-1]])

    #Source image
    source_url = "http://images.cocodataset.org/val2017/000000058111.jpg"
    source_image = Image.open(requests.get(source_url, stream=True).raw)

    # Display input image and query image
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(target_image)
    ax[1].imshow(source_image)
    #plt.show()
    '''
    #output = processor(images=source_image, return_tensors="pt")
    #print(output)
    print(dir(model))
    print("\n")
    print(model.image_embedder)

    # Process source image
    source_pixel_values = processor(images=source_image, return_tensors="pt").pixel_values

    # For visualization, we need the preprocessed source image (i.e. padded and resized, but not yet normalized)
    unnormalized_source_image = get_preprocessed_image(source_pixel_values) #manually defined function

    # Get image features
    with torch.no_grad():
        feature_map = model.image_embedder(source_pixel_values)[0]
    print(feature_map.shape)

    # Rearrange feature map
    batch_size, height, width, hidden_size = feature_map.shape
    image_features = feature_map.reshape(batch_size, height * width, hidden_size)
    
    # Get objectness logits
    objectnesses = model.objectness_predictor(image_features)
    print(objectnesses)

    num_patches = (model.config.vision_config.image_size // model.config.vision_config.patch_size)**2
    print(num_patches)

    source_boxes = model.box_predictor(image_features, feature_map=feature_map)
    source_class_embeddings = model.class_predictor(image_features)[1]
    
    # Remove batch dimension
    objectnesses = torch.sigmoid(objectnesses[0].detach()).numpy()
    source_boxes = source_boxes[0].detach().numpy()
    source_class_embeddings = source_class_embeddings[0].detach().numpy()


    # Let's show the top 3 patches
    top_k = 3
    # objectnesses = sigmoid(objectnesses)
    objectness_threshold = np.partition(objectnesses, -top_k)[-top_k]
    print(objectness_threshold)  #0.27
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(unnormalized_source_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for i, (box, objectness) in enumerate(zip(source_boxes, objectnesses)):
        if objectness < objectness_threshold:
            continue

        cx, cy, w, h = box
        ax.plot(
            [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
            [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
            color='lime',
        )

        print("Index:", i)
        print("Objectness:", objectness)

        ax.text(
            cx - w / 2 + 0.015,
            cy + h / 2 - 0.015,
            f'Index {i}: {objectness:1.2f}',
            ha='left',
            va='bottom',
            color='black',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'lime',
                'boxstyle': 'square,pad=.3',
            },
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title(f'Top {top_k} objects by objectness')

    plt.show()

    # Get the query embedding with the index of the selected object.
    # We're using the cat:
    query_object_index = 1465  # Index of the cat box above.
    query_embedding = source_class_embeddings[query_object_index]

    # We have peformed zero shot object detection on the query image to identify the query object
    # Now we use this query object for image-conditioned one-shot object detection

    # Process target image
    target_pixel_values = processor(images=target_image, return_tensors="pt").pixel_values
    unnormalized_target_image = get_preprocessed_image(target_pixel_values)

    with torch.no_grad():
        feature_map = model.image_embedder(target_pixel_values)[0]

    # Get boxes and class embeddings (the latter conditioned on query embedding)
    b, h, w, d = feature_map.shape
    target_boxes = model.box_predictor(
        feature_map.reshape(b, h * w, d), feature_map=feature_map
    )

    target_class_predictions = model.class_predictor(
        feature_map.reshape(b, h * w, d),
        torch.tensor(query_embedding[None, None, ...]),  # [batch, queries, d]
    )[0]

    # Remove batch dimension and convert to numpy:
    target_boxes = target_boxes[0].detach().numpy()
    target_logits = target_class_predictions[0].detach().numpy()

    # Take the top 2 scoring logits
    top_indices = np.argsort(target_logits[:, 0])[-2:]  # Get the indices of the top 2 scores
    scores = sigmoid(target_logits[top_indices, 0])  # Get scores for the top 2 logits

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(unnormalized_target_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for i, top_ind in enumerate(top_indices):
        # Get the corresponding bounding box
        cx, cy, w, h = target_boxes[top_ind]
        
        # Plot the bounding box
        ax.plot(
            [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
            [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
            color='lime',
        )

        # Add text for the score
        ax.text(
            cx - w / 2 + 0.015,
            cy + h / 2 - 0.015,
            f'Score: {scores[i]:1.2f}',  # Use scores[i] for the respective score
            ha='left',
            va='bottom',
            color='black',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'lime',
                'boxstyle': 'square,pad=.3',
            },
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title(f'Top 2 Closest Matches')
    plt.show()
    plt.show()