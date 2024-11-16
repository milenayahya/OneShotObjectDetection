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
    
   # create_val_subset("coco-2017/validation/data", "coco_val_subset", num_images=100)

    options_1s = RunOptions(
        mode="test",
        source_image_paths= os.path.join(query_dir, "coco_query_objects_filtered/1_shot/"),
        target_image_paths= "coco-2017/validation/data/",
        comment="coco_validation", 
        query_batch_size=8, 
        test_batch_size=8, 
        confidence_threshold=0.09,
        topk_test=170,
        k_shot=1,
        visualize_test_images=False,
        nms_threshold=0.3
    )

    writer = SummaryWriter(comment=options_1s.comment)
    model = options_1s.model.from_pretrained(options_1s.backbone)
    processor = options_1s.processor.from_pretrained(options_1s.backbone)
    
    '''
    indexes, query_embeddings, classes = zero_shot_detection(
        model,
        processor,
        options_1s,
        writer
    )
     
    file = os.path.join(query_dir, f"classes_{options_1s.comment}.json")    
    with open(file, 'w') as f:
        json.dump(classes, f)

    torch.save(query_embeddings, f"Queries/query_embeddings_{options_1s.comment}_gpu.pth")

    '''
    file = os.path.join(query_dir, f"classes_coco_queries.json")    
    query_embeddings = torch.load(f"Queries/query_embeddings_coco_queries_gpu.pth")
    classes = json.load(open(file, 'r'))

    coco_results = one_shot_detection_batches(
            model,
            processor,
            query_embeddings,
            classes,
            options_1s,
            writer,
            per_image= False
        ) 
    
    result_file = os.path.join(results_dir, f"results_{options_1s.comment}.json")
    #visualize_results(result_file, writer, per_image=False, args=options_1s, random_selection=True)

    writer.close()  

   # run_all_tasks()