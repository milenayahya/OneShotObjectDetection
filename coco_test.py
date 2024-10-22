from osod import zero_shot_detection, one_shot_detection_batches
from RunOptions import RunOptions 
from tensorboardX import SummaryWriter
from pycocotools import coco
from pycocotools  import cocoeval
import pylab
import json

if __name__=="__main__":

    options = RunOptions(
        source_image_paths="coco_query_objects_filtered",
        target_image_paths="coco-2017/validation/data", 
        comment="coco_0shot", 
        query_batch_size=8, 
        test_batch_size=8, 
        k_shot=1,
        visualize_test_images=False,
        nms_threshold=0.5
    )
    

    # Image-Conditioned Object Detection
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

    with open("results_cocoFormat_1shot.json", "w") as f:
        json.dump(coco_results,f)

    with open("results.json", "w") as f:
        json.dump(results,f)




