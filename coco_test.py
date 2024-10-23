from osod import zero_shot_detection, one_shot_detection_batches
from RunOptions import RunOptions 
from tensorboardX import SummaryWriter
from typing import Optional, Literal
import json
from osod import logger

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

        results, coco_results = one_shot_detection_batches(
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

        with open(f"results_{task_key}.json", "w") as f:
            json.dump(results, f)
        
        writer.close() 

if __name__ == "__main__":
    '''
    options = RunOptions(
        source_image_paths="coco_query_objects_filtered",
        target_image_paths="coco-2017/validation/data", 
        comment="coco_0shot_test", 
        query_batch_size=8, 
        test_batch_size=8, 
        k_shot=1,
        visualize_test_images=False,
        nms_threshold=0.5
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

    writer.close()  
    '''

    run_all_tasks()