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
from coco_preprocess import ID2CLASS
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
      
def plot_pr_curve(cocoEval, precisions, recall, category_id, cat_id_to_index):
    plt.figure()
    cat_idx = cat_id_to_index[category_id]
    for iou_idx in range(precisions.shape[0]):
        precision = precisions[iou_idx,  :, cat_idx, 0, 2] # all areas and max detections
        # Filter out invalid precision and recall values
        valid_indices = precision != -1
        precision = precision[valid_indices]

        if precision.shape[0] == 0:
            print(f"Warning: No valid precision values for IoU={cocoEval.params.iouThrs[iou_idx]:.2f}")
            continue  # Skip if precision array is empty

        if precision.shape[0] != recall.shape[0]:
            print(f"Warning: Precision and Recall shapes do not match. Precision shape: {precision.shape}, Recall shape: {recall.shape}")
            continue  # Skip if shapes don't match

        plt.plot(recall, precision, label=f'IoU={cocoEval.params.iouThrs[iou_idx]:.2f}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Category ID {category_id} ({ID2CLASS[category_id]})')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate(annFile, resFile, plot_pr = False):
    cocoGt=COCO(annFile)
    annotations = load_json_lines(resFile)


    results_img_ids = {ann['image_id'] for ann in annotations}
    results_cat_ids = {ann['category_id'] for ann in annotations}
    gt_img_ids = set(cocoGt.getImgIds())
   
    # Check if image IDs match
    if not results_img_ids.issubset(gt_img_ids):
        print("Error: Results contain image IDs not present in COCO ground truth.")
    else:
        # Initialize COCO detections api
        cocoDt = cocoGt.loadRes(annotations)
        imgIds= list(results_img_ids)
        catIds = list(results_cat_ids)

        #mappping to index the categories in the evaluation results
        cat_id_to_index = {cat_id: i for i, cat_id in enumerate(catIds)}

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.params.catIds = catIds  # Set to a single category ID

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        precisions = cocoEval.eval['precision'] 
        recalls = cocoEval.params.recThrs

        if plot_pr: 
            for cat_id in catIds:
                plot_pr_curve(cocoEval, precisions, recalls, cat_id, cat_id_to_index)

def tune_confidence_threshold(annFile, resFile, threshold_range):
    cocoGt = COCO(annFile)
    annotations = load_json_lines(resFile)
    cat_ids = {ann['category_id'] for ann in annotations}

    optimal_thresholds = {}
    for cat_id in cat_ids:
        best_f1 = 0
        best_threshold = 0

        for threshold in threshold_range:
            filtered_annotations = [ann for ann in annotations if ann['score'] > threshold and ann['category_id'] == cat_id]
            if not filtered_annotations:
                continue

            results_img_ids = {ann['image_id'] for ann in filtered_annotations}
            gt_img_ids = set(cocoGt.getImgIds())

            cats = {ann['category_id'] for ann in filtered_annotations}
            mapping = {cat: i for i, cat in enumerate(cats)}

            # Check if image IDs match
            if not results_img_ids.issubset(gt_img_ids):
                print(f"Error: Results contain image IDs not present in COCO ground truth for category {cat_id}.")
                continue

            # Initialize COCO detections api
            cocoDt = cocoGt.loadRes(filtered_annotations)
            imgIds = list(results_img_ids)


            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds
            cocoEval.params.catIds = [cat_id]
            cocoEval.params.iouThrs = [0.5]

            cocoEval.evaluate()
            cocoEval.accumulate()
         #   cocoEval.summarize()

            precision = cocoEval.eval['precision']  # [TxRxKxAxM]
            recall = cocoEval.eval['recall']  # [TxKxAxM]

            ### extract relevant precision and recall for category
            precision = precision[:, :, mapping[cat_id], 0, 2]  # all areas and max detections
            recall = recall[:, mapping[cat_id], 0, 2]

            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall)
            f1_score = np.nan_to_num(f1_score, nan=0.0)  # Replace NaN with 0
        
            # Find the maximum F1 score
            max_f1 = np.max(f1_score)
            if max_f1 > best_f1:
                best_f1 = max_f1
                best_threshold = threshold
        
        optimal_thresholds[cat_id] = (best_threshold, best_f1)
    return cat_ids, optimal_thresholds


if __name__ == '__main__':

    annFile = 'coco-2017/raw/instances_val2017.json'
    resFile = 'Results/results_coco_queries.json'

    evaluate(annFile, resFile, plot_pr = False)

    '''
    # Tune confidence threshold
    thresholds = np.arange(0.1, 1.0, 0.05)
    cat_ids, optimal_thresholds = tune_confidence_threshold(annFile, resFile, thresholds)

    print("Categories evaluated: ", cat_ids)
    print("Optimal thresholds and F1 scores: ", optimal_thresholds)

    '''


    # MANAGE IOU THRESHOLD in F1 SCORE CALCULATION





    '''
    single_imgId = imgIds[0]
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

    '''

    
