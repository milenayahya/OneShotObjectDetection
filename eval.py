from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def load_json_lines(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

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
    print("Image IDs in results:", results_img_ids)

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
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()