from transformers import Owlv2Processor, Owlv2ForObjectDetection
import requests
from PIL import Image, ImageDraw
import torch
import numpy as np
import os
import cv2  # type: ignore
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD  # type: ignore
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import warnings
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from typing import Any, Literal, Optional, List, Tuple, Union
from pprint import pformat
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
import logging
from logs.tglog import RequestsHandler, LogstashFormatter, TGFilter

PROJECT_BASE_PATH = os.path.abspath(
    "C:\\Users\\cm03009\\Documents\\OneShotObjectDetection"
)
tg_handler = RequestsHandler()
formatter = LogstashFormatter()
filter = TGFilter()
tg_handler.setFormatter(formatter)
tg_handler.addFilter(filter)
logging.basicConfig(
    format="%(asctime)s [%(filename)s@%(funcName)s] [%(levelname)s]:> %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_BASE_PATH, "logs/debug.log")),
        logging.StreamHandler(),
        tg_handler,
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Colors for bounding boxes for different source queries
colors = ["red", "green", "blue", "purple"]
linestyles = ["-", "-", "--", "-"]
ID2CLASS = {1: "apple", 2: "cat", 3: "dog", 4: "usb"}

CLASS2ID = {v: k for k, v in ID2CLASS.items()}


def nms_tf(bboxes, scores, classes, threshold):
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

    return filtered_bboxes, scores, filtered_classes


def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (
        pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]
    ) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image


def load_image_group(image_dir):
    images = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image is not None:
                images.append(image)
    return images


def load_query_image_group(image_dir):
    images = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            category = ID2CLASS[int(image_name.split("_")[0])]
            image_path = os.path.join(image_dir, image_name)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image is not None:
                images.append((image, category))
    return images


def visualize_objectnesses_batch(
    image_batch, source_boxes, source_pixel_values, objectnesses, topk
):

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
                color="lime",
            )

            print("Index:", i)
            print("Objectness:", objectness)

            # Add text for objectness score
            ax.text(
                cx - w / 2 + 0.015,
                cy + h / 2 - 0.015,
                f"Index {i}: {objectness:1.2f}",
                ha="left",
                va="bottom",
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "lime",
                    "boxstyle": "square,pad=.3",
                },
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_title(f"Top {topk} objects by objectness")

        plt.show()


def zero_shot_detection(
    source_image_paths,
    model,
    processor,
    topk,
    batch_size,
    visualize,
    query_selection=False,
):

    source_class_embeddings = []
    images, classes = zip(*load_query_image_group(source_image_paths))
    images = list(images)
    classes = list(classes)
    query_embeddings = []
    indexes = []

    for batch_start in range(0, len(images), batch_size):
        image_batch = images[batch_start : batch_start + batch_size]
        source_pixel_values = processor(
            images=image_batch, return_tensors="pt"
        ).pixel_values
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
            visualize_objectnesses_batch(
                image_batch, source_boxes, source_pixel_values, objectnesses, topk
            )

        if query_selection:
            # Remove batch dimension for each image in the batch
            for i, image in enumerate(image_batch):
                current_source_boxes = source_boxes[i].detach().numpy()
                current_objectnesses = torch.sigmoid(objectnesses[i].detach()).numpy()
                current_class_embedding = source_class_embedding[i].detach().numpy()

                # Extract the query embedding for the current image based on the given index
                query_embedding = current_class_embedding[
                    np.argmax(current_objectnesses)
                ]
                indexes.append(np.argmax(current_objectnesses))
                query_embeddings.append(query_embedding)

    if query_selection:
        return indexes, query_embeddings, classes


def find_query_patches_batches(
    source_image_paths, model, processor, indexes, batch_size
):
    query_embeddings = []
    images, classes = zip(*load_query_image_group(source_image_paths))
    images = list(images)
    classes = list(classes)

    for batch_start in range(0, len(images), batch_size):
        image_batch = images[batch_start : batch_start + batch_size]
        source_pixel_values = processor(
            images=image_batch, return_tensors="pt"
        ).pixel_values
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

    return query_embeddings, classes


def one_shot_detection_batches(
    target_image_paths,
    model,
    processor,
    query_embeddings,
    classes,
    threshold,
    batch_size,
    nms_interclass=False,
    visualize=True,
    topk=None,
):

    images = load_image_group(target_image_paths)
    all_batch_results = []

    for batch_start in range(0, len(images), batch_size):
        image_batch = images[batch_start : batch_start + batch_size]
        target_pixel_values = processor(
            images=image_batch, return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            feature_map = model.image_embedder(target_pixel_values)[0]

        b, h, w, d = map(int, feature_map.shape)
        target_boxes = model.box_predictor(
            feature_map.reshape(b, h * w, d), feature_map=feature_map
        )
        # Contains the predicted boxes for each image in the batch.
        # It is a list of lists
        reshaped_feature_map = feature_map.view(b, h * w, d)

        batch_results = []

        # Process each image in the batch
        for image_idx in range(b):
            unnormalized_image = get_preprocessed_image(target_pixel_values[image_idx])
            class_wise_boxes = {cls: [] for cls in classes}  # dictionary of lists
            class_wise_scores = {cls: [] for cls in classes}
            all_boxes = []
            all_scores = []
            class_identifiers = []

            if visualize:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(unnormalized_image, extent=(0, 1, 1, 0))
                ax.set_axis_off()

            for idx, query_embedding in enumerate(query_embeddings):
                query_embedding_tensor = torch.tensor(
                    query_embedding[None, None, ...], dtype=torch.float32
                )

                target_class_predictions = model.class_predictor(
                    reshaped_feature_map, query_embedding_tensor
                )[0]
                target_boxes_np = target_boxes[image_idx].detach().numpy()
                target_logits = target_class_predictions[image_idx].detach().numpy()

                if topk is not None:
                    top_indices = np.argsort(target_logits[:, 0])[-topk[idx] :]
                    scores = sigmoid(target_logits[top_indices, 0])
                else:
                    scores = sigmoid(target_logits[:, 0])
                    top_indices = np.where(scores > threshold)[0]
                    scores = scores[top_indices]

                if not nms_interclass:
                    # Accumulate boxes and scores for each class
                    class_wise_boxes[classes[idx]].extend(target_boxes_np[top_indices])
                    class_wise_scores[classes[idx]].extend(scores)
                else:
                    all_boxes.append(target_boxes_np[top_indices])
                    all_scores.append(scores)
                    class_identifiers.extend([classes[idx]] * len(top_indices))

            final_boxes = []
            final_scores = []
            final_classes = []

            if nms_interclass == False:
                for cls in classes:
                    if class_wise_boxes[cls]:
                        # Apply NMS on the bounding boxes of the same class
                        bboxes_tensor = tf.convert_to_tensor(
                            class_wise_boxes[cls], dtype=tf.float32
                        )
                        pscores_tensor = tf.convert_to_tensor(
                            class_wise_scores[cls], dtype=tf.float32
                        )
                        nms_boxes, nms_scores, _ = nms_tf(
                            bboxes_tensor, pscores_tensor, class_identifiers, 0.3
                        )
                        nms_boxes = nms_boxes.numpy()
                        nms_scores = nms_scores.numpy()

                        final_boxes.extend(nms_boxes)
                        final_scores.extend(nms_scores)
                        final_classes.extend([cls] * len(nms_boxes))

            else:
                # Concatenate all boxes and scores for NMS
                all_boxes = np.concatenate(all_boxes, axis=0)
                all_scores = np.concatenate(all_scores, axis=0)

                bboxes_tensor = tf.convert_to_tensor(all_boxes, dtype=tf.float32)
                pscores_tensor = tf.convert_to_tensor(all_scores, dtype=tf.float32)

                # nms_indices = tf.image.non_max_suppression(bboxes_tensor, pscores_tensor, max_output_size=100, iou_threshold=0.3)
                nms_boxes, nms_scores, nms_classes = nms_tf(
                    bboxes_tensor, pscores_tensor, class_identifiers, 0.3
                )
                nms_boxes = nms_boxes.numpy()
                nms_scores = nms_scores.numpy()

                final_boxes.extend(nms_boxes)
                final_scores.extend(nms_scores)
                final_classes.extend(nms_classes)

            # Plot bounding boxes for each image
            if visualize:
                for i, (box, score, cls) in enumerate(
                    zip(final_boxes, final_scores, final_classes)
                ):
                    cx, cy, w, h = box

                    # Plot the bounding box with the respective color
                    ax.plot(
                        [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                        [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                        color=colors[
                            CLASS2ID[cls] - 1
                        ],  # Use a different color for each query
                        alpha=0.5,
                        linestyle=linestyles[CLASS2ID[cls] - 1],
                    )

                    # Add text for the score
                    ax.text(
                        cx - w / 2 + 0.015,
                        cy + h / 2 - 0.015,
                        f"Class: {cls} Score: {score:1.2f}",  # Indicate which query the result is from
                        ha="left",
                        va="bottom",
                        color="black",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": colors[
                                CLASS2ID[cls] - 1
                            ],  # Use respective color
                            "boxstyle": "square,pad=.3",
                        },
                    )

            if visualize:
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)
                ax.set_title(
                    f"One-Shot Object Detection (Batch {batch_start // batch_size + 1}, Image {image_idx + 1})"
                )
                plt.show()

            batch_query_results = {
                "batch_index": batch_start // batch_size + 1,
                "image_idx": image_idx + batch_start,
                "boxes": final_boxes,
                "scores": final_scores,
                "classes": final_classes,
            }
            batch_results.append(batch_query_results)
            print(batch_query_results)

        all_batch_results.append(batch_results)

    return all_batch_results


if __name__ == "__main__":

    source_image_paths = "fewshot_query_images/"
    target_image_paths = "test_images/"

    # Image-Conditioned Object Detection
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble"
    )

    batch_size = 4
    top_objectness = 3
    manual_query_selection = True
    threshold = 0.96
    visualize = True
    nms_interclass = True

    # Find the objects in the query images
    if manual_query_selection:
        # zero_shot_detection(source_image_paths, model, processor, top_objectness, batch_size, visualize, query_selection=False)
        # indexes = [1523, 1700, 1465, 1344]
        indexes = [1523, 1641, 1750, 1700, 1700, 747, 1465, 1704, 1214, 1344, 876, 2071]
        query_embeddings, classes = find_query_patches_batches(
            source_image_paths, model, processor, indexes, batch_size
        )

    else:
        indexes, query_embeddings, classes = zero_shot_detection(
            source_image_paths,
            model,
            processor,
            top_objectness,
            batch_size,
            visualize,
            query_selection=True,
        )

    print(indexes, classes)

    # Detect query objects in test images
    results = one_shot_detection_batches(
        target_image_paths,
        model,
        processor,
        query_embeddings,
        classes,
        threshold,
        batch_size,
        nms_interclass,
        visualize,
    )

    print(results)
