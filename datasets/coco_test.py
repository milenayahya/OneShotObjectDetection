from osod import zero_shot_detection, one_shot_detection_batches,visualize_results
from RunOptions import RunOptions 
from tensorboardX import SummaryWriter
from typing import Optional, Literal
import json
import torch
import os
import random
import shutil
from osod import logger
from config import query_dir, test_dir, results_dir

Tasks = Literal["1_shot_multi_nms", "1_shot_single_nms", "5_shot_multi_nms", "5_shot_single_nms"]

def create_task_options() -> dict[Tasks, RunOptions]:
    return {
        "1_shot_multi_nms": RunOptions(
            source_image_paths="coco_query_objects_filtered",
            target_image_paths="coco-2017/validation/data", 
            comment="coco_1shot_multi_nms", 
            query_batch_size=8, 
            test_batch_size=8, 
            k_shot=1,
            visualize_test_images=False,
            nms_between_classes=True,
            nms_threshold=0.5
        ),
        "1_shot_single_nms": RunOptions(
            source_image_paths="coco_query_objects_filtered",
            target_image_paths="coco-2017/validation/data", 
            comment="coco_1shot_single_nms", 
            query_batch_size=8, 
            test_batch_size=8, 
            k_shot=1,
            visualize_test_images=False,
            nms_between_classes=False,
            nms_threshold=0.5
        ),
        "5_shot_multi_nms": RunOptions(
            source_image_paths="coco_query_objects_filtered",
            target_image_paths="coco-2017/validation/data", 
            comment="coco_5shot_multi_nms", 
            query_batch_size=8, 
            test_batch_size=8, 
            k_shot=5,
            visualize_test_images=False,
            nms_between_classes=True,
            nms_threshold=0.5
        ),
        "5_shot_single_nms": RunOptions(
            source_image_paths="coco_query_objects_filtered",
            target_image_paths="coco-2017/validation/data", 
            comment="coco_5shot_single_nms", 
            query_batch_size=8, 
            test_batch_size=8, 
            k_shot=5,
            visualize_test_images=False,
            nms_between_classes=False,
            nms_threshold=0.5
        )
    }
def create_val_subset(source_dir, target_dir, num_images=100):
    """
    Randomly select a specified number of images from the source directory
    and store them in the target directory.

    Parameters:
    - source_dir: Path to the source directory containing the images.
    - target_dir: Path to the target directory where the selected images will be stored.
    - num_images: Number of images to select and copy.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get a list of all image files in the source directory
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Randomly select the specified number of images
    selected_images = random.sample(image_files, num_images)

    # Copy the selected images to the target directory
    for image in selected_images:
        source_path = os.path.join(source_dir, image)
        target_path = os.path.join(target_dir, image)
        shutil.copyfile(source_path, target_path)

    print(f"Copied {num_images} images to {target_dir}")
def run_all_tasks(tasks: Optional[dict[Tasks, RunOptions]] = None):
    tasks = tasks or create_task_options()

    for task_key, options in tasks.items():
        logger.info(f"Running task {task_key}")


        writer = SummaryWriter(comment=options.comment)
        model = options.model.from_pretrained(options.backbone)
        processor = options.processor.from_pretrained(options.backbone)

        indexes, query_embeddings, classes = zero_shot_detection(
        model,
        processor,
        options,
        writer
        )

        coco_results, target_pixel_values = one_shot_detection_batches(
            model,
            processor,
            query_embeddings,
            classes,
            options,
            writer
        )
        
        logger.info("Writing Results")
        with open(f"results_cocoFormat_{task_key}.json", "w") as f:
            json.dump(coco_results, f)

        writer.close() 

if __name__ == "__main__":

    # create_val_subset("coco-2017/validation/data", os.path.join(test_dir,"coco_val_subset"), num_images=500)

    thresholds = {
        1: 0.8,
        2: 0.6,
        3: 0.8,
        4: 0.4,
        5: 0.15,
        6: 0.95,
        7: 0.95,
        8: 0.95,
        9: 0.1,
        10: 0.1,
        11: 0.9,
        13: 0.9,
        14: 0.7,
        15: 0.8,
        16: 0.85,
        17: 0.9,
        18: 0.7,
        19: 0.9,
        20: 0.75,
        21: 0.65,
        22: 0.1,
        23: 0.3,
        24: 0.1,
        25: 0.1,
        27: 0.95,
        28: 0.1,
        31: 0.85,
        32: 0.7,
        33: 0.35,
        34: 0.85,
        35: 0.1,
        36: 0.95,
        37: 0.85,
        38: 0.1,
        39: 0.7,
        40: 0.1,
        41: 0.1,
        42: 0.95,
        43: 0.7,
        44: 0.75,
        46: 0.85,
        47: 0.45,
        48: 0.9,
        49: 0.95,
        50: 0.85,
        51: 0.95,
        52: 0.1,
        53: 0.1,
        54: 0.65,
        55: 0.2,
        56: 0.1,
        57: 0.1,
        58: 0.95,
        59: 0.9,
        60: 0.95,
        61: 0.9,
        62: 0.75,
        63: 0.85,
        64: 0.95,
        65: 0.75,
        67: 0.95,
        70: 0.1,
        72: 0.85,
        73: 0.85,
        74: 0.95,
        75: 0.6,
        76: 0.95,
        77: 0.8,
        78: 0.9,
        79: 0.8,
        80: 0,
        81: 0.95,
        82: 0.1,
        84: 0.95,
        85: 0.2,
        86: 0.9,
        87: 0.75,
        88: 0.65,
        89: 0,
        90: 0.95,
    }
    '''
    options= RunOptions(
        mode="test",
        source_image_paths= os.path.join(query_dir, "coco_query_objects_filtered/1_shot"),
        target_image_paths= os.path.join(test_dir, "coco_val_subset"),
        data="COCO",
        comment="imgnet_coco_subset", 
        query_batch_size=8, 
        test_batch_size=8, 
        confidence_threshold=0.96,
        topk_test=50,
        k_shot=1,
        visualize_test_images=True,
        visualize_query_images=False,
        nms_between_classes=False,
        write_to_file_freq=40,
        nms_threshold=0.3
    )

    writer = SummaryWriter(comment=options.comment)
    model = options.model.from_pretrained(options.backbone)
    processor = options.processor.from_pretrained(options.backbone)
    
    
    indexes, query_embeddings, classes = zero_shot_detection(
        model,
        processor,
        options,
        writer
    )
     
    file = os.path.join(query_dir, f"classes_{options.comment}.json")    
    with open(file, 'w') as f:
        json.dump(classes, f)

    torch.save(query_embeddings, f"Queries/query_embeddings_{options.comment}_gpu.pth")

    

    file = os.path.join(query_dir, f"classes_ImageNet.json")    
    query_embeddings = torch.load(f"Queries/query_embeddings_ImageNet_gpu.pth")
    classes = json.load(open(file, 'r'))

    coco_results = one_shot_detection_batches(
            model,
            processor,
            query_embeddings,
            classes,
            options,
            writer,
            per_image= False
        ) 
    
    result_file = os.path.join(results_dir, f"results_{options.comment}.json")
    visualize_results(result_file, writer, per_image=False, args=options, random_selection=0.3)
    writer.close()  

   # run_all_tasks()

   '''
    