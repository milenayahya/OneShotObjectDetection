from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json
from torch.utils.tensorboard import SummaryWriter   
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def load_json_lines(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    
def pr_curve(writer, precisions, recalls):
    for i, (precision, recall) in enumerate(zip(precisions, recalls)):
        writer.add_pr_curve(f"Precision/Recall Curve/{i}", precision, recall)

if __name__ == '__main__':
    annType = 'bbox'      #specify type here
    #initialize COCO ground truth api
    annFile = 'coco-2017/validation/labels.json'
    cocoGt=COCO(annFile)

    #initialize COCO detections api
    resFile = 'results_coco_1shot_val.json'
    # Load results
    annotations = load_json_lines(resFile)

    # Print image IDs from results
    results_img_ids = {ann['image_id'] for ann in annotations}
    print("Image IDs in results:", len(results_img_ids))

    # Print image IDs from COCO ground truth
    gt_img_ids = set(cocoGt.getImgIds())
    print("Number of images in COCO ground truth:", len(gt_img_ids))
    # Check if image IDs match
    if not results_img_ids.issubset(gt_img_ids):
        print("Error: Results contain image IDs not present in COCO ground truth.")
    else:
        # Initialize COCO detections api
        cocoDt = cocoGt.loadRes(annotations)
    
    

        imgIds=sorted(cocoGt.getImgIds())
       

        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = [0]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        '''

         # Evaluate results per category
        cat_ids = cocoGt.getCatIds()
        for cat_id in cat_ids:
            cocoEval = COCOeval(cocoGt, cocoDt, annType)
            cocoEval.params.catIds = [cat_id]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            cat_name = cocoGt.loadCats(cat_id)[0]['name']
            print(f"Evaluation results for category: {cat_name}")

        '''