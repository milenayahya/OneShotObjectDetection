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
from datetime import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from typing import Any, Literal, Optional, List, Tuple, Union
from pprint import pformat
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
import logging
from RunOptions import RunOptions

#from logs.tglog import RequestsHandler, LogstashFormatter, TGFilter

PROJECT_BASE_PATH = os.path.abspath(
    "C:\\Users\\cm03009\\Documents\\OneShotObjectDetection"
)

'''
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
'''
# Directory and file path
log_dir = os.path.join(PROJECT_BASE_PATH, "logs")
# Create a timestamp for the log file name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"debug_{timestamp}.log")

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
        ],
    )
logger = logging.getLogger(__name__)



# Colors for bounding boxes for different source queries
'''
colors = ["red", "green", "blue", "purple"]
linestyles = ["-", "-", "--", "-"]
ID2CLASS = {1: "apple", 2: "cat", 3: "dog", 4: "usb"}
'''

colors = ["red", "green", "blue"]
linestyles = ["-", "-", "-"]
ID2CLASS = {1: "squirrel", 2: "nail", 3: "pin"}

CLASS2ID = {v: k for k, v in ID2CLASS.items()}


def nms_tf(
        bboxes: tf.Tensor,
        scores: tf.Tensor, 
        classes: List[str], 
        threshold: float
)-> Tuple[tf.Tensor, tf.Tensor, List[str]]:
    logger.info("Applying NMS with threshold: %f on %d boxes", threshold, len(bboxes))
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
    logger.info("NMS completed. Kept %d boxes.", len(filtered_bboxes))
    return filtered_bboxes, scores, filtered_classes


def get_preprocessed_image(pixel_values: torch.Tensor) -> Image.Image:
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (
        pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]
    ) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image


def load_image_group(image_dir: str) -> List[np.ndarray]:
    logger.info("Loading images from directory: %s", image_dir)
    images = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith((".png", ".jpg", ".jpeg", ".bmp", "JPEG")):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image is not None:
                images.append(image)
    if images is not None:
        logger.info("Loaded %d images.", len(images))
    else:
        logger.warning("Failed to load test image batch")
    return images


def load_query_image_group(image_dir: str) -> List[Tuple[np.ndarray, str]]:
    logger.info("Loading query images and categories from directory: %s", image_dir)
    images = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith((".png", ".jpg", ".jpeg", ".bmp", "JPEG")):
            category = ID2CLASS[int(image_name.split("_")[0])]
            image_path = os.path.join(image_dir, image_name)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image is not None:
                images.append((image, category))
    if images is not None:
        logger.info("Loaded %d query images.", len(images))
    else:
        logger.warning("Failed to load query images")
    return images


def visualize_objectnesses_batch(
    image_batch, source_boxes, source_pixel_values, objectnesses, topk, batch_index, writer = Optional[SummaryWriter]
):
    logger.info("Visualizing the objectnesses for objects detected in query images")
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
        # Add image with bounding boxes to the writer
        writer.add_figure(f"Query_Images_with_boxes/image_{idx}_batch{batch_index}", fig, global_step= batch_index+idx+1)
        writer.flush()

def zero_shot_detection(
    model: Owlv2ForObjectDetection,
    processor: Owlv2Processor,
    args: "RunOptions",
    writer: Optional[SummaryWriter] = None,
)-> Union[Tuple[List[int], List[np.ndarray], List[str]], None]:

    source_class_embeddings = []
    images, classes = zip(*load_query_image_group(args.source_image_paths))
    images = list(images)
    classes = list(classes)
    query_embeddings = []
    indexes = []

    for batch_start in range(0, len(images), args.query_batch_size):
        logger.info(f"Performing zero-shot detection on batch {batch_start // args.query_batch_size + 1} of the source images.")
        image_batch = images[batch_start : batch_start + args.query_batch_size]
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
        batch_index = batch_start//batch_size+1
        if args.visualize_query_images:
            visualize_objectnesses_batch(
                image_batch, source_boxes, source_pixel_values, objectnesses, args.topk_query, batch_index, writer
            )

        if not args.manual_query_selection:
            # Remove batch dimension for each image in the batch
            for i, image in enumerate(image_batch):
                current_objectnesses = torch.sigmoid(objectnesses[i].detach()).numpy()
                current_class_embedding = source_class_embedding[i].detach().numpy()
                # Extract the query embedding for the current image based on the given index
                query_embedding = current_class_embedding[
                    np.argmax(current_objectnesses)
                ]
                indexes.append(np.argmax(current_objectnesses))
                query_embeddings.append(query_embedding)

    if not args.manual_query_selection:
        writer.add_text("indexes of query objects", str(indexes))
        writer.add_text("classes of query objects", str(classes) )
        logger.info("Finished extracting the query embeddings")
        writer.flush()
        return indexes, query_embeddings, classes

    logger.info("Zero-shot detection completed")
    return None

def find_query_patches_batches( 
    model: Owlv2ForObjectDetection, 
    processor: Owlv2Processor, 
    args: "RunOptions",
    indexes: List [int], 
    writer: Optional["SummaryWriter"] = None
):
    query_embeddings = []
    images, classes = zip(*load_query_image_group(args.source_image_paths))
    images = list(images)
    classes = list(classes)

    for batch_start in range(0, len(images), args.query_batch_size):
        image_batch = images[batch_start : batch_start + args.query_batch_size]
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

        writer.add_text("indexes of query objects", str(indexes))
        writer.add_text("classes of query objects", str(classes))
        writer.flush()
        np.save("query_embeddings.npy", query_embeddings)
        np.save("classes.npy", classes)

    return query_embeddings, classes


def one_shot_detection_batches(
    model,
    processor,
    query_embeddings,
    classes,
    args: "RunOptions",
    writer: Optional["SummaryWriter"] = None    
):

    images = load_image_group(args.target_image_paths)
    all_batch_results = []
    logger.info("Performing Prediction on test images")
    for batch_start in range(0, len(images), args.test_batch_size):
        logger.info(f"Processing bacth {batch_start // args.test_batch_size + 1}")
        image_batch = images[batch_start : batch_start + args.test_batch_size]
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

            if args.visualize_test_images:
                logger.info("Visualizing the images and predicted results")
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

                if args.topk_test is not None:
                    top_indices = np.argsort(target_logits[:, 0])[-args.topk_test[idx] :]
                    scores = sigmoid(target_logits[top_indices, 0])
                else:
                    scores = sigmoid(target_logits[:, 0])
                    top_indices = np.where(scores > args.confidence_threshold)[0]
                    scores = scores[top_indices]

                if not args.nms_between_classes:
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

            if not args.nms_between_classes:
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
                            bboxes_tensor, pscores_tensor, class_identifiers, args.nms_threshold
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
                    bboxes_tensor, pscores_tensor, class_identifiers, args.nms_threshold
                )
                nms_boxes = nms_boxes.numpy()
                nms_scores = nms_scores.numpy()

                final_boxes.extend(nms_boxes)
                final_scores.extend(nms_scores)
                final_classes.extend(nms_classes)

            # Plot bounding boxes for each image
            if args.visualize_test_images:
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

            if args.visualize_test_images:
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)
                ax.set_title(
                    f"One-Shot Object Detection (Batch {batch_start // args.test_batch_size + 1}, Image {image_idx + 1})"
                )
                writer.add_figure(f"Test_Image_with_prediction_boxes/image_{image_idx}_batch_{batch_start//args.test_batch_size +1}", fig, global_step=image_idx + batch_start//args.test_batch_size +1)

            batch_query_results = {
                "batch_index": batch_start // args.test_batch_size + 1,
                "image_idx": image_idx + batch_start,
                "boxes": final_boxes,
                "scores": final_scores,
                "classes": final_classes,
            }
            batch_results.append(batch_query_results)
            print(batch_query_results)

        all_batch_results.append(batch_results)

    writer.flush()
    logger.info(f"Finished prediction of all images, the results are: \n {pformat(all_batch_results, indent=2, underscore_numbers=True)}")

    return all_batch_results


if __name__ == "__main__":

    
    options = RunOptions()
    print(options.__dict__)

    # Image-Conditioned Object Detection
    writer = SummaryWriter(comment=options.comment)
    model = options.model.from_pretrained(options.backbone)
    processor = options.processor.from_pretrained(options.backbone)

    # Find the objects in the query images
    if options.manual_query_selection:
        zero_shot_detection(model, processor, options, writer)
        #indexes = [1523, 1700, 1465, 1344]
        indexes = [1523, 1641, 1750, 1700, 1700, 747, 1465, 1704, 1214, 1344, 876, 2071]
        query_embeddings, classes = find_query_patches_batches(
            model, processor, options, indexes, writer
        )

    else:
        indexes, query_embeddings, classes = zero_shot_detection(
            model,
            processor,
            options,
            writer
        )

    # Detect query objects in test images
    results = one_shot_detection_batches(
        model,
        processor,
        query_embeddings,
        classes,
        options,
        writer
    )

    