from transformers import Owlv2Processor, Owlv2ForObjectDetection
import json
import torch
import numpy as np
import os
import cv2 
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import tensorflow as tf
from datetime import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from typing import Any, Literal, Optional, List, Tuple, Union
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
import logging
from tqdm import tqdm
from RunOptions import RunOptions
from config import PROJECT_BASE_PATH, query_dir, results_dir, test_dir
from torchvision.ops import batched_nms
import tensorflow as tf
from utils import  get_preprocessed_image, convert_from_x1y1x2y2_to_coco, create_image_id_mapping, save_results, get_filename_by_index, map_coco_ids, read_results

class ModelOutputs:
    def __init__(self, logits, pred_boxes):
        self.logits = logits
        self.pred_boxes = pred_boxes

# Directories
log_dir = os.path.join(PROJECT_BASE_PATH, "logs")

# Create a timestamp for the log file name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"debug_{timestamp}.log")

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

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
#Use for my data
ID2COLOR = {
    1: "red",
    2: "green",
    3: "blue",
    4: "purple"
}
linestyles = ["-", "-", "--", "-"]
ID2CLASS = {1: "apple", 2: "cat", 3: "dog", 4: "usb"}
CLASS2ID = {v: k for k, v in ID2CLASS.items()}

#Use for ImageNet

colors = ["red", "green", "blue"]
linestyles = ["-", "-", "-"]
ID2CLASS = {1: "squirrel", 2: "nail", 3: "pin"}

CLASS2ID = {v: k for k, v in ID2CLASS.items()}
'''

def nms_batched(boxes, scores, classes, im_indices, args):
    
    '''
    Perform non-maximum suppression on a batch of bounding boxes.
    Takes boxes in (x1, y1, x2, y2) format for NMS.
    Returns the boxes in COCO format (x, y, width, height) for evaluation.

    '''
    if args.nms_between_classes:
        classes_nms = torch.zeros_like(classes)
    else:
        classes_nms = classes

    indices = batched_nms(boxes, scores, classes_nms, args.nms_threshold)
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
    filtered_classes = classes[indices]
    filtered_indices = im_indices[indices]
    filtered_boxes_coco = convert_from_x1y1x2y2_to_coco(filtered_boxes)

    return  filtered_boxes_coco, filtered_scores, filtered_classes, filtered_indices


def load_image_group(image_dir: str) -> List[np.ndarray]:
    '''
    Load images from a directory.
    
    '''
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

def load_query_image_group(image_dir: str, k=None) -> List[Tuple[np.ndarray, str]]:
    
    '''

    Load query images and their categories from a directory.
    Parameters: directory, k (number of query images to load per category)
    Returns: a list of tuples, each containing an image and its category

    '''
    logger.info("Loading query images and categories from directory: %s", image_dir)
    images = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith((".png", ".jpg", ".jpeg", ".bmp", "JPEG")):
           # category = ID2CLASS[int(image_name.split("_")[0])]
            category = ID2CLASS[float(image_name.split("_")[0])]
            #Taking only k query images to perfrom k shot detection
            k_hat = image_name.split("_")[-1].split(".")[0]
            if k is not None and int(k_hat) > k:         
                continue

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
    image_batch, source_boxes, source_pixel_values, objectnesses, topk, batch_start, batch_size, classes, all_data, writer = Optional[SummaryWriter]
):
    """

    Visualize the objectness scores and bounding boxes for a batch of query images. (Zero-Shot Detection Visualization)

    """
    unnormalized_source_images = []
    for pixel_value in source_pixel_values:
        unnormalized_image = get_preprocessed_image(pixel_value)
        unnormalized_source_images.append(unnormalized_image)   

    for idx, image in enumerate(image_batch):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(unnormalized_source_images[idx], extent=(0, 1, 1, 0))
        ax.set_axis_off()

        image_idx = batch_start + idx 
        category = classes[image_idx]

        data = {
            "category": ID2CLASS[category],
            "boxes": [],
        }

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

            data["boxes"].append({
                "index": i,
                "objectness": round(float(objectness), 2)
            })

            # Plot bounding box
            ax.plot(
                [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                color="lime",
            )

           # print("Index:", i)
           # print("Objectness:", objectness)

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
        all_data.append(data)
        ax.set_ylim(1, 0)
        ax.set_title(f"Zero-Shot on {ID2CLASS[category]}: Top {topk} objects by objectness,")
        # Add image with bounding boxes to the writer
        writer.add_figure(f"Query_Images_with_boxes/image_{idx}_batch{batch_start//batch_size}", fig, global_step= batch_start+idx+1)
        writer.flush()


def zero_shot_detection(
    model: Owlv2ForObjectDetection,
    processor: Owlv2Processor,
    args: "RunOptions",
    writer: Optional[SummaryWriter] = None,
)-> Union[Tuple[List[int], List[np.ndarray], List[str]], None]:

    """
    Perform zero-shot detection on a batch of query images.

    """
    source_class_embeddings = []
    images, classes = zip(*load_query_image_group(args.source_image_paths, args.k_shot))
    images = list(images)
    classes = list(classes)
    classes = [CLASS2ID[class_name] for class_name in classes]
    query_embeddings = []
    indexes = []
    all_data = []

    start_GPUtoCPU = torch.cuda.Event(enable_timing=True)
    end_GPUtoCPU = torch.cuda.Event(enable_timing=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    logger.info(f"Performing zero-shot detection on the query images.")
    total_batches = (len(images) + args.query_batch_size - 1) // args.query_batch_size
  
    with tqdm(total=total_batches, desc="Processing query batches") as pbar:
        for batch_start in range(0, len(images), args.query_batch_size):
    
            torch.cuda.empty_cache()
            image_batch = images[batch_start : batch_start + args.query_batch_size]
            source_pixel_values = processor(
                images=image_batch, return_tensors="pt"
            ).pixel_values.to(device)
            with torch.no_grad():
                feature_map = model.image_embedder(source_pixel_values)[0]

                # Rearrange feature map
                batch_size, height, width, hidden_size = feature_map.shape
                image_features = feature_map.reshape(batch_size, height * width, hidden_size).to(device)

                # Get objectness logits and boxes
                objectnesses = model.objectness_predictor(image_features)
                source_boxes = model.box_predictor(image_features, feature_map=feature_map)
                source_class_embedding = model.class_predictor(image_features)[1]
                source_class_embeddings.append(source_class_embedding)
   
            if args.visualize_query_images:
                visualize_objectnesses_batch(
                    image_batch, source_boxes.cpu(), source_pixel_values.cpu(), objectnesses.cpu(), args.topk_query, batch_start, args.query_batch_size, classes, all_data, writer
                )

            if not args.manual_query_selection:
            
                #start_GPUtoCPU.record()
                current_objectnesses = torch.sigmoid(objectnesses.detach())
                current_class_embeddings = source_class_embedding.detach()
                #end_GPUtoCPU.record()
                #torch.cuda.synchronize()
                #time_total = start_GPUtoCPU.elapsed_time(end_GPUtoCPU)

                #print(f"Time taken to transfer embeddings of image {i} in batch {batch_start//batch_size} from GPU to CPU is: {time_total} ms")
               
                # Extract the query embedding for the current images based on the provided index
                max_indices = torch.argmax(current_objectnesses, dim=1) # has the shape of batch_size
                indexes.extend(max_indices.cpu().tolist()) # convert to list

                batch_query_embeddings = current_class_embeddings[torch.arange(batch_size), max_indices]
                
                query_embeddings.extend(batch_query_embeddings) #embeddings are stored as GPU tensors

            pbar.update(1)

    file = os.path.join(query_dir, f"objectness_indexes_{args.comment}.json")
    with open(file, 'w') as f:
        json.dump(all_data, f, indent=4)

    if not args.manual_query_selection:
        writer.add_text("indexes of query objects", str(indexes))
        writer.add_text("classes of query objects", str(classes) )
        logger.info("Finished extracting the query embeddings")
        writer.flush()
        logger.info("Zero-shot detection completed")

        if args.topk_query > 1:
            class_embeddings_dict = {}

            # Group queries of same class together
            for embedding, class_label in zip(query_embeddings, classes):
                if class_label not in class_embeddings_dict:
                    class_embeddings_dict[class_label] = []
                class_embeddings_dict[class_label].append(embedding)
            
            query_embeddings = []
            classes = []
            for class_label, embeddings in class_embeddings_dict.items():
                average_embedding = torch.mean(torch.stack(embeddings), dim=0)
                query_embeddings.append(average_embedding)
                classes.append(class_label)

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
    """
    Find the query embeddings for a batch of query images based on the provided indexes.
    Allows manual selection of query object in query image based on index.

    """
    query_embeddings = []
    images, classes = zip(*load_query_image_group(args.source_image_paths))
    images = list(images)
    classes = list(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    for batch_start in range(0, len(images), args.query_batch_size):
        torch.cuda.empty_cache()
        image_batch = images[batch_start : batch_start + args.query_batch_size]
        source_pixel_values = processor(
            images=image_batch, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            feature_map = model.image_embedder(source_pixel_values)[0]

        # Rearrange feature map
        batch_size, height, width, hidden_size = feature_map.shape
        image_features = feature_map.reshape(batch_size, height * width, hidden_size).to(device)
        source_boxes = model.box_predictor(image_features, feature_map=feature_map)
        source_class_embedding = model.class_predictor(image_features)[1]

        # Remove batch dimension for each image in the batch
        for i in range(batch_size):
            current_source_boxes = source_boxes[i].detach().cpu().numpy()
            current_class_embedding = source_class_embedding[i].detach().cpu().numpy()

            # Extract the query embedding for the current image based on the given index
            query_embedding = current_class_embedding[indexes[batch_start + i]]
            query_embeddings.append(query_embedding)

        writer.add_text("indexes of query objects", str(indexes))
        writer.add_text("classes of query objects", str(classes))
        writer.flush()
        np.save("query_embeddings.npy", query_embeddings)
        np.save("classes.npy", classes)

    return query_embeddings, classes

def visualize_results(filepath, writer, per_image, args, random_selection=None):
    """
    Visualize the test images with the predicted bounding boxes and confidence scores.
    Boxes have to be in x, y, w, h format (not normalized).
    Random_selection is used to randomly select x% of the images for visualization.
    
    """ 
    if args.data == "COCO":
        from coco_preprocess import ID2CLASS, CLASS2ID, ID2COLOR
    elif args.data == "MGN":
        from mgn_preprocess import ID2CLASS, CLASS2ID, ID2COLOR
    image_data = read_results(filepath, random_selection)
    dir = args.target_image_paths
    if per_image:   
        dir = dir.split("/")[0]
    for image_id, data in image_data.items():
        if str(image_id).endswith((".png", ".jpg", ".jpeg", ".bmp", "JPEG")):
            image_id = image_id.split(".")[0]
        for filename in os.listdir(dir):
            if args.data == "MGN" and not per_image:
                filenamee = filename.split("_")[0]
            else:
                filenamee = filename.split(".")[0]
            file = filename.split(".")[0]
            if filenamee.endswith(str(image_id)):
                if args.data == "COCO":
                    image_path = os.path.join(dir, file + ".jpg")
                if args.data == "MGN":
                    image_path = os.path.join(dir, file + ".png")
                if args.data == "TestData":
                    image_path = os.path.join(dir, file + ".jpg")
                if args.data == "ImageNet":
                    image_path = os.path.join(dir, file + ".JPEG")
                if per_image:
                    image_path = os.path.join(dir, file + ".jpg")
                    
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) 
                fig, ax = plt.subplots()
                plt.imshow(image)  
                ax = plt.gca()
                for box, cat, score in zip(data['bboxes'], data['categories'], data['scores']):
                    x,y,w,h =box
                    ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor=ID2COLOR[cat], facecolor='none', linewidth=2))
                    ax.text(
                        x + 0.015, 
                        y+ 3, 
                        f"{ID2CLASS[cat]}: {score:.2f}", fontsize=6, color=ID2COLOR[cat],
                        ha="left",
                        va="bottom",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": ID2COLOR[
                                cat
                            ],  # Use respective color
                            #"edgecolor": colors[
                            #    CLASS2ID[cat] - 1
                            #],  # Use respective color
                            "boxstyle": "square",
                        }
                    )
                
                writer.add_figure(f"Test_Image_with_prediction_boxes/image_{image_id}", fig)
                

def one_shot_detection_batches(
    model,
    processor,
    query_embeddings,
    classes,
    args: "RunOptions",
    writer: Optional["SummaryWriter"] = None,   
    per_image: bool = False 
):
    """
    Perform One-Shot Detection on a batch of images.
    'test' mode allows specifying "topk" and/or confidence threshold for filtering the results.
    'validation' mode does not apply any filtering on the predicted boxes.
    'per_image' flag is used to indicate whether args.target_image_paths is a single image or a directory.
    'per_image' is set to True when server is waiting for an image to perform real-time detection.

    """
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    if per_image:
        logger.info("Loading image")
        if args.target_image_paths.endswith((".png", ".jpg", ".jpeg", ".bmp", "JPEG")):
            images = [cv2.cvtColor(cv2.imread(args.target_image_paths), cv2.COLOR_BGR2RGB)]
    
    else:
        images = load_image_group(args.target_image_paths)

    coco_results = []

    mapping = create_image_id_mapping('coco-2017/validation/labels.json')
    logger.info("Performing Prediction on test images")
    total_batches = (len(images) + args.test_batch_size - 1) // args.test_batch_size
    pbar = tqdm(total=total_batches, desc="Processing test batches")
    for batch_start in range(0, len(images), args.test_batch_size):

        torch.cuda.empty_cache()
        image_batch = images[batch_start : batch_start + args.test_batch_size]
        target_sizes = torch.tensor([img.shape[:2] for img in image_batch], device=device)
        target_pixel_values = processor(
            images=image_batch, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            feature_map = model.image_embedder(target_pixel_values)[0]
            b, h, w, d = map(int, feature_map.shape)
            target_boxes = model.box_predictor(
                feature_map.reshape(b, h * w, d), feature_map=feature_map
            ) # dimension = [batch_size, nb_of_boxes, 4]

            reshaped_feature_map = feature_map.view(b, h * w, d)

            query_embeddings_tensor = torch.stack(query_embeddings) # Shape: (num_batches, batch_size, hidden_size)
            target_class_predictions, _ = model.class_predictor(reshaped_feature_map, query_embeddings_tensor)  # Shape: [batch_size, num_queries, num_classes]

            outputs = ModelOutputs(logits=target_class_predictions, pred_boxes=target_boxes)
            results =processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0)
            
            extracted_boxes = [result['boxes'] for result in results]
            extracted_boxes_tensor = torch.stack([torch.tensor(boxes) for boxes in extracted_boxes])

            # Set target_boxes to the extracted boxes tensor
            target_boxes = extracted_boxes_tensor.to(device) # Boxes are in x1, y1, x2, y2 format
            target_boxes = target_boxes.detach()  # Keep in GPU
            scores = torch.sigmoid(target_class_predictions)

            if args.topk_test is not None:
                top_indices = torch.argsort(scores[:, :, 0], descending=True)[:, :args.topk_test]
                scores = scores[torch.arange(b)[:, None], top_indices]
                target_boxes = target_boxes[torch.arange(b)[:, None], top_indices]  

            if args.mode == "test":
                if isinstance(args.confidence_threshold, (int, float)):
                    top_indices = (scores > args.confidence_threshold).any(dim=-1)
                else:
                    idxs = torch.argmax(scores, dim=-1)
                    # Compare the scores with the corresponding class-specific threshold
                    predicted_classes = [[classes[idx] for idx in image_idxs] for image_idxs in idxs.tolist()]
                    thresholds = [[args.confidence_threshold[class_id] for class_id in image_classes] for image_classes in predicted_classes]
                    thresholds_tensor = torch.tensor(thresholds, device=scores.device)
                    max_scores = scores[torch.arange(scores.size(0)).unsqueeze(1), torch.arange(scores.size(1)).unsqueeze(0), idxs]
                    top_indices = (max_scores > thresholds_tensor).to(device) 
                    
                filtered_boxes = [target_boxes[i][top_indices[i]] for i in range(b)]
                filtered_scores = [scores[i][top_indices[i]] for i in range(b)]    

                # Padding filtered boxes and scores to have the same number of boxes across the batch
                max_boxes_per_image = max(len(boxes) for boxes in filtered_boxes)
                padded_boxes = torch.zeros(b, max_boxes_per_image, 4, device=device)
                padded_scores = torch.full((b, max_boxes_per_image), -float('inf'), device=device)
                padded_classes = torch.zeros(b, max_boxes_per_image, device=device, dtype=torch.long)
                image_indices = torch.zeros(b, max_boxes_per_image, device=device, dtype=torch.long)

                for i in range(b):
                    num_boxes = filtered_boxes[i].shape[0]
                    padded_boxes[i, :num_boxes] = filtered_boxes[i]
                    padded_scores[i, :num_boxes] = filtered_scores[i].max(dim=1)[0]
                    
                    max_scores, max_indexes = torch.max(filtered_scores[i], dim=1)
                    class_ids = torch.tensor([classes[idx] for idx in max_indexes.tolist()]).to(device)
                    padded_classes[i, :num_boxes] = class_ids  # Update padded_classes correctly
                    image_indices[i, :num_boxes] = i  # Assign the current batch index to each box

            else:
                # Validation mode
                padded_boxes = target_boxes
                padded_scores = scores
                class_indexes = torch.argmax(scores, dim=-1).flatten().tolist()
                padded_classes = torch.tensor([classes[idx] for idx in class_indexes], device=device) 
                image_indices = torch.arange(b, device=device).unsqueeze(1).expand(b, padded_boxes.shape[1]).contiguous()

            # Reshape the padded data for batched NMS
            # Flatten the padded tensors for NMS
            flattened_boxes = padded_boxes.view(-1, 4)
            flattened_scores = padded_scores.view(-1)
            flattened_classes = padded_classes.view(-1)
            flattened_image_indices = image_indices.view(-1)

            if args.mode == "test":
                # Filter out padded values (-inf scores)
                valid_indices = flattened_scores > -float('inf')
                flattened_boxes = flattened_boxes[valid_indices]
                flattened_scores = flattened_scores[valid_indices]
                flattened_classes = flattened_classes[valid_indices]
                flattened_image_indices = flattened_image_indices[valid_indices]
            

            #NMS only on same class
            nms_boxes_coco, nms_scores, nms_classes, nms_image_indices = nms_batched(
                flattened_boxes, flattened_scores, flattened_classes, flattened_image_indices, args
                )  

            # Collect results in COCO format
            for idx, (box, score, cls, img_idx) in enumerate(zip(nms_boxes_coco, nms_scores, nms_classes, nms_image_indices)):
                
                rounded_box = [round(coord, 2) for coord in box.tolist()]
                if per_image:
                    img_id = args.target_image_paths.split("/")[-1].split(".")[0]
                elif args.data == "COCO":
                    img_id = map_coco_ids(mapping, get_filename_by_index(args.target_image_paths, img_idx.item() + batch_start))
                elif args.data == "MGN":
                    img_id = get_filename_by_index(args.target_image_paths, img_idx.item() + batch_start)
                    img_id = int(img_id.split(".")[0].split("_")[0].split("scene")[1])
                else:
                    img_id = img_idx.item() + batch_start

                coco_results.append({
                    "image_id": img_id,
                    "category_id": cls.item(),
                    "bbox": rounded_box,
                    "score": round(score.item(), 2)
                })
      
        pbar.update(1)

        # Periodically save results to file every 30 batches
        # TO DO: make the number of batches to save results to file a parameter
        batch_index = batch_start // args.test_batch_size + 1
        if batch_index % args.write_to_file_freq == 0:
            save_results(coco_results, args, per_image, img_id)
            #logger.info(f"Saved results of 30 batches to file starting from batch {batch_index - 30} to batch {batch_index}")    
            coco_results.clear()

        torch.cuda.empty_cache()

    logger.info(f"Finished prediction of all images")
    pbar.close()

    # Save the remaining results to file
    save_results(coco_results, args, per_image, img_id)

    if per_image:        
        return img_id, coco_results
    else:
        return coco_results

if __name__ == "__main__":
    

    options = RunOptions(
        mode = "test",
        source_image_paths= os.path.join(query_dir, "MGN_query_set"),
        target_image_paths= os.path.join(test_dir, "MGN/MGN_subset"), 
        data="MGN",
        comment="vis_test", 
        query_batch_size=8, 
        manual_query_selection=False,
        confidence_threshold=0.1,
        test_batch_size=8, 
        k_shot=1,
        topk_test= 170,
        visualize_query_images=True,
        nms_between_classes=False,
        nms_threshold=0.3,
        write_to_file_freq=5,
    )


    # Image-Conditioned Object Detection
    writer = SummaryWriter(comment=options.comment)
    model = options.model.from_pretrained(options.backbone)
    processor = options.processor.from_pretrained(options.backbone)
    
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

    '''

    # Find the objects in the query images
    if options.manual_query_selection:
        zero_shot_detection(model, processor, options, writer)
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

    
    file = os.path.join(query_dir, f"classes_{options.comment}.json")
    with open(file, 'w') as f:
        json.dump(classes, f)

    # Save the list of GPU tensors to a file
    torch.save(query_embeddings, os.path.join(query_dir, f'query_embeddings_{options.comment}_gpu.pth'))
    
    
    file = os.path.join(query_dir, f"classes_{options.data}.json")
    with open(file, 'r') as f:
        classes = json.load(f)

    # Load the list of tensors onto the GPU
    query_embeddings = torch.load(f'Queries/query_embeddings_{options.data}_gpu.pth', map_location='cuda')
    
    # Detect query objects in test images
    coco_results = one_shot_detection_batches(
        model,
        processor,
        query_embeddings,
        classes,
        options,
        writer,
        per_image=False
    )
    '''
    filepath = os.path.join(results_dir, f"results_MGN_subset_test_nms_sameClass.json")
    visualize_results(filepath, writer, per_image=False, args=options, random_selection=0.1)


