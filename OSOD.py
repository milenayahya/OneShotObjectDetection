from transformers import Owlv2Processor, Owlv2ForObjectDetection
import requests
import os
from PIL import Image, ImageDraw
from scipy.special import expit as sigmoid
import torch
import numpy as np
import cv2
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from matplotlib import rcParams
import matplotlib.pyplot as plt

# Colors for bounding boxes for different source queries
colors = ['red', 'green', 'blue', 'purple'] 
linestyles = ['-', '-', '--', '-' ]

def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

def load_image_group(image_dir):
    images = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image is not None:
                images.append(image)
    return images

        
def visualize_objectnesses(unnormalized_source_image, source_boxes, objectnesses, topk):
    # LSow the top k patches
    top_k = topk
    objectness_threshold = np.partition(objectnesses, -top_k)[-top_k]
    # print(objectness_threshold)  
    
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

def zero_shot_detection(source_image_paths, model, processor, topk):

    source_class_embeddings = []
    images = load_image_group(source_image_paths)

    # Process each source image to extract query embeddings
    for idx, image in enumerate(images):
        # Load and preprocess source image
        source_pixel_values = processor(images=image, return_tensors="pt").pixel_values
        unnormalized_source_image = get_preprocessed_image(source_pixel_values)

        # Get image features
        with torch.no_grad():
            feature_map = model.image_embedder(source_pixel_values)[0]

        # Rearrange feature map
        batch_size, height, width, hidden_size = feature_map.shape
        image_features = feature_map.reshape(batch_size, height * width, hidden_size)

        # Get objectness logits and boxes
        objectnesses = model.objectness_predictor(image_features)
        source_boxes = model.box_predictor(image_features, feature_map=feature_map)
        source_class_embedding = model.class_predictor(image_features)[1]
        source_class_embeddings.append(source_class_embedding)

        # Remove batch dimension
        objectnesses = torch.sigmoid(objectnesses[0].detach()).numpy()
        source_boxes = source_boxes[0].detach().numpy()
        source_class_embedding = source_class_embedding[0].detach().numpy()

        visualize_objectnesses(unnormalized_source_image, source_boxes, objectnesses, topk)

       
def find_query_patches(source_image_paths, model, processor, indexes):
    
    query_embeddings = []
    images = load_image_group(source_image_paths)

    for idx, image in enumerate(images):
        
        # Preprocess source image
        source_pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
        # Get image features
        with torch.no_grad():
            feature_map = model.image_embedder(source_pixel_values)[0]

        # Rearrange feature map
        batch_size, height, width, hidden_size = feature_map.shape
        image_features = feature_map.reshape(batch_size, height * width, hidden_size)

        # Get objectness logits and boxes
        source_boxes = model.box_predictor(image_features, feature_map=feature_map)
        source_class_embedding = model.class_predictor(image_features)[1]

        # Remove batch dimension
        source_boxes = source_boxes[0].detach().numpy()
        source_class_embedding = source_class_embedding[0].detach().numpy()
        query_embedding = source_class_embedding[indexes[idx]]
        query_embeddings.append(query_embedding)

    return query_embeddings

def one_shot_detection(target_image, model, processor, query_embeddings, threshold, topk=None):
        
    # Process the target image for one-shot detection
    target_pixel_values = processor(images=target_image, return_tensors="pt").pixel_values
    unnormalized_target_image = get_preprocessed_image(target_pixel_values)

    with torch.no_grad():
        feature_map = model.image_embedder(target_pixel_values)[0]

    b, h, w, d = map(int, feature_map.shape)
    target_boxes = model.box_predictor(
        feature_map.reshape(b, h * w, d), feature_map=feature_map
    )

    # Reshape feature map using view for robust reshaping
    reshaped_feature_map = feature_map.view(b, h * w, d)   

    all_scores_and_boxes = []
    
    # Visualize results for each query embedding
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(unnormalized_target_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for idx, query_embedding in enumerate(query_embeddings):
        query_embedding_tensor = torch.tensor(query_embedding[None, None, ...], dtype=torch.float32)
        # Get boxes and class embeddings (conditioned on current query embedding)
        target_class_predictions = model.class_predictor(
        reshaped_feature_map,
        query_embedding_tensor  # Ensure query embedding is properly formatted
        )[0]

        # Remove batch dimension and convert to numpy:
        target_boxes_np = target_boxes[0].detach().numpy()
        target_logits = target_class_predictions[0].detach().numpy()

        # Take the top K scoring logits for each query
        if topk != None:   
            top_indices = np.argsort(target_logits[:, 0])[-topk[idx]:]  # Get the indices of the top 2 scores
            scores = sigmoid(target_logits[top_indices, 0]) 
        else: 
            scores = sigmoid(target_logits[:, 0])  # Get scores for the top 2 logits
            top_indices = np.where(scores>threshold)[0]
            scores = scores[top_indices]

        # Store the results in the container
        query_results = {
            'query_index': idx + 1,
            'scores': scores,
            'boxes': target_boxes_np[top_indices]
        }
        all_scores_and_boxes.append(query_results)

        # Plot bounding boxes for the current query
        for i, top_ind in enumerate(top_indices):
            cx, cy, w, h = target_boxes_np[top_ind]
            
            # Plot the bounding box with the respective color
            ax.plot(
                [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                color=colors[idx],  # Use a different color for each query
                alpha = 0.5,
                linestyle = linestyles[idx]
            )

            # Add text for the score
            ax.text(
                cx - w / 2 + 0.015,
                cy + h / 2 - 0.015,
                f'Query {idx+1} Score: {scores[i]:1.2f}',  # Indicate which query the result is from
                ha='left',
                va='bottom',
                color='black',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': colors[idx],  # Use respective color
                    'boxstyle': 'square,pad=.3',
                },
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title('One-Shot Object Detection from Multiple Queries')
    plt.show()

    return all_scores_and_boxes

if __name__ == "__main__":

    # Test image paths (for multiple sources)
    source_image_paths = 'query_images/'
    target_image_paths = 'test_images/dog2.jpg'
    many_images = False

    # Load the model and processor
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    
    # Find the objects in the query images
    top_objectness = 3
   # zero_shot_detection(source_image_paths, model, processor, top_objectness)

    # Manually select the indexes
    indexes = [1523, 1700, 1465, 1344]
    query_embeddings = find_query_patches(source_image_paths, model, processor, indexes)

    if many_images:
        images = load_image_group(target_image_paths)
    else:
        images = cv2.cvtColor(cv2.imread(target_image_paths), cv2.COLOR_BGR2RGB)
    
    # Find objects in target image(s)
    instances = [2,2,2]
    threshold = 0.96
    results = one_shot_detection(images, model, processor, query_embeddings, threshold)
    print(f" the results are: {results}")
    '''
    for idx, image in enumerate(images):
        results = one_shot_detection(image, model, processor, query_embeddings, threshold)
        print(f"for the image {idx}, the results are: {results}")
    '''
   