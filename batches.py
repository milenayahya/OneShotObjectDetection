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
from scipy.special import expit as sigmoid

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

def visualize_objectnesses_batch(image_batch, source_boxes, source_pixel_values, objectnesses, topk):
    
    unnormalized_source_images = []
    for pixel_value in source_pixel_values:
        unnormalized_image = get_preprocessed_image(pixel_value)
        unnormalized_source_images.append(unnormalized_image)
    
    for idx, image in enumerate(image_batch):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(unnormalized_source_images[idx], extent=(0, 1, 1, 0))
        ax.set_axis_off()

        # Get objectness scores and boxes for the current image
        
        current_objectnesses = torch.sigmoid(objectnesses[idx].detach()).numpy()
        current_boxes = source_boxes[idx].detach().numpy()

        top_k = topk
        objectness_threshold = np.partition(current_objectnesses, -top_k)[-top_k]
        # print(objectness_threshold)  
        

        for i, (box, objectness) in enumerate(zip(current_boxes, current_objectnesses)):
            if objectness < objectness_threshold:
                continue

            cx, cy, w, h = box
            
            # Plot bounding box
            ax.plot(
                [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                color='lime',
            )

            print("Index:", i)
            print("Objectness:", objectness)

            # Add text for objectness score
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
        ax.set_title(f'Top {topk} objects by objectness')

        plt.show()
        


def zero_shot_detection(source_image_paths, model, processor, topk, batch_size,visualize, query_selection= False):

    source_class_embeddings = []
    images = load_image_group(source_image_paths)

    for batch_start in range(0,len(images),batch_size):
        image_batch = images[batch_start:batch_start + batch_size]
        source_pixel_values = processor(images=image_batch, return_tensors="pt").pixel_values
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
        if visualize:
            visualize_objectnesses_batch(image_batch, source_boxes, source_pixel_values, objectnesses, topk)
            
        if query_selection:
            query_embeddings = []
            indexes= []
            # Remove batch dimension for each image in the batch
            for i in range(min(batch_size, len(source_boxes))): 
                current_source_boxes = source_boxes[i].detach().numpy()
                current_objectnesses = torch.sigmoid(objectnesses[i].detach()).numpy()
                current_class_embedding = source_class_embedding[i].detach().numpy()
                
                # Extract the query embedding for the current image based on the given index
                query_embedding = current_class_embedding[np.argmax(current_objectnesses)]
                indexes.append(np.argmax(current_objectnesses))
                query_embeddings.append(query_embedding)
       
    if query_selection:
        return indexes, query_embeddings


def find_query_patches_batches(source_image_paths, model, processor, indexes, batch_size):
    query_embeddings = []
    images = load_image_group(source_image_paths)

    for batch_start in range(0,len(images),batch_size):
        image_batch = images[batch_start:batch_start + batch_size]
        source_pixel_values = processor(images=image_batch, return_tensors="pt").pixel_values
        with torch.no_grad():
            feature_map = model.image_embedder(source_pixel_values)[0]
        
        # Rearrange feature map
        batch_size, height, width, hidden_size = feature_map.shape
        image_features = feature_map.reshape(batch_size, height * width, hidden_size)
        source_boxes = model.box_predictor(image_features, feature_map=feature_map)
        source_class_embedding = model.class_predictor(image_features)[1]

        # Remove batch dimension for each image in the batch
        for i in range(batch_size):
            current_source_boxes = source_boxes[i].detach().numpy()
            current_class_embedding = source_class_embedding[i].detach().numpy()
            
            # Extract the query embedding for the current image based on the given index
            query_embedding = current_class_embedding[indexes[batch_start + i]]
            query_embeddings.append(query_embedding)

    return query_embeddings

def one_shot_detection_batches(target_image_paths, model, processor, query_embeddings, threshold, batch_size, visualize, topk=None):

    images = load_image_group(target_image_paths)
    all_batch_results = [] 

    for batch_start in range(0, len(images), batch_size):
        image_batch = images[batch_start:batch_start + batch_size]
        target_pixel_values = processor(images=image_batch, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            feature_map = model.image_embedder(target_pixel_values)[0]

        b, h, w, d = map(int, feature_map.shape)
        target_boxes = model.box_predictor(
            feature_map.reshape(b, h * w, d), feature_map=feature_map
        )

        reshaped_feature_map = feature_map.view(b, h * w, d) 

        batch_results = [] 

        # Process each image in the batch
        for image_idx in range(b):  
            unnormalized_image = get_preprocessed_image(target_pixel_values[image_idx])

            if visualize:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(unnormalized_image, extent=(0, 1, 1, 0))
                ax.set_axis_off()

            for idx, query_embedding in enumerate(query_embeddings):
                query_embedding_tensor = torch.tensor(query_embedding[None, None, ...], dtype=torch.float32)

                target_class_predictions = model.class_predictor(
                    reshaped_feature_map,
                    query_embedding_tensor
                )[0]

                target_boxes_np = target_boxes[image_idx].detach().numpy()
                target_logits = target_class_predictions[image_idx].detach().numpy()

                if topk is not None:
                    top_indices = np.argsort(target_logits[:, 0])[-topk[idx]:]
                    scores = sigmoid(target_logits[top_indices, 0])
                else:
                    scores = sigmoid(target_logits[:, 0])
                    top_indices = np.where(scores > threshold)[0]
                    scores = scores[top_indices]

                batch_query_results = {
                    'batch_index': batch_start // batch_size + 1,
                    'query_index': idx + 1,
                    'scores': scores,
                    'boxes': target_boxes_np[top_indices]
                }
                batch_results.append(batch_query_results)

                # Plot bounding boxes for each query
                if visualize:
                    for i, top_ind in enumerate(top_indices):
                        cx, cy, w, h = target_boxes_np[top_ind]

                        # Plot the bounding box with the respective color
                        ax.plot(
                            [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                            [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                            color=colors[idx],  # Use a different color for each query
                            alpha=0.5,
                            linestyle=linestyles[idx]
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


            if visualize:
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)
                ax.set_title(f'One-Shot Object Detection (Batch {batch_start // batch_size + 1}, Image {image_idx + 1})')
                plt.show()

        all_batch_results.append(batch_results)

    return all_batch_results



if __name__ == "__main__":

    source_image_paths = 'query_images/'
    target_image_paths = 'test_images/'

    # Image-Conditioned Object Detection
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    batch_size = 4
    top_objectness = 3
    manual_query_selection = False
    threshold = 0.96
    visualize = True

    # Find the objects in the query images
    if manual_query_selection:
        zero_shot_detection(source_image_paths, model, processor, top_objectness, batch_size, visualize, query_selection=False)
        indexes = [1523, 1700, 1465, 1344]
        query_embeddings = find_query_patches_batches(source_image_paths, model, processor, indexes, batch_size)

    else: 
        indexes, query_embeddings = zero_shot_detection(source_image_paths, model, processor, top_objectness, batch_size, visualize, query_selection=True)
    
    results = one_shot_detection_batches(target_image_paths,model,processor,query_embeddings, threshold, batch_size, visualize)
    print(results)

    

    