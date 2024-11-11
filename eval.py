from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json
import math
from torchvision.ops import nms
import torch
from torch.utils.tensorboard import SummaryWriter  
import matplotlib.pyplot as plt 
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def load_json_lines(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    
def calculate_iou(pred_bbox, gt_bbox):
    
    x1_pred, y1_pred, w_pred, h_pred = pred_bbox
    x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred

    x1_gt, y1_gt, w_gt, h_gt = gt_bbox
    x2_gt, y2_gt = x1_gt + w_gt, y1_gt + h_gt

    inter_x1 = max(x1_pred, x1_gt)
    inter_y1 = max(y1_pred, y1_gt)
    inter_x2 = min(x2_pred, x2_gt)
    inter_y2 = min(y2_pred, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area = w_pred * h_pred
    gt_area = w_gt * h_gt
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou       
 
def plot_pr_curve(results):
    thresholds = [result[0] for result in results]
    precisions = [result[1][0] for result in results]  # AP @ IoU=0.50:0.95
    recalls = [result[1][8] for result in results]  # AR @ IoU=0.50:0.95

    plt.figure()
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    
    annType = 'bbox'      
    annFile = 'Results/instances_val2017.json'
    cocoGt=COCO(annFile)

    #initialize COCO detections api
    resFile = 'Results/results_coco_queries.json'

    # Load results
    annotations = load_json_lines(resFile)

    results_img_ids = {ann['image_id'] for ann in annotations}
    results_cat_ids = {ann['category_id'] for ann in annotations}
    print("results img ids",results_img_ids)
    gt_img_ids = set(cocoGt.getImgIds())
   
    # Check if image IDs match
    if not results_img_ids.issubset(gt_img_ids):
        print("Error: Results contain image IDs not present in COCO ground truth.")
    else:
        # Initialize COCO detections api
        cocoDt = cocoGt.loadRes(annotations)
        imgIds= list(results_img_ids)
        single_imgId = imgIds[0]
        catIds = list(results_cat_ids)
    
        # Print predictions for the selected image
        predictions = [ann for ann in annotations if ann['image_id'] == single_imgId]
        print("Length of predictions: ", len(predictions))

        print(f"Predictions for Image ID {single_imgId}:")
        for pred in predictions:
            print(pred)

        # Print ground truth annotations for the selected image
        gt_annotations = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[single_imgId]))
        print(f"Ground Truth for Image ID {single_imgId}:")
        for gt in gt_annotations:
            print(gt)
        
       
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.params.catIds = catIds  # Set to a single category ID

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
    
    