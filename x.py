import fiftyone as fo
import os
import fiftyone.zoo as foz
from utils import read_results, create_image_id_mapping, map_coco_filenames
from coco_preprocess import ID2CLASS
import fiftyone.utils.coco as fouc
    

if __name__ == "__main__":
    
    data_dir = "C:/Users/cm03009/Documents/OneShotObjectDetection/coco-2017/validation/data"  
    labels_path = "C:/Users/cm03009/Documents/OneShotObjectDetection/coco-2017/raw/instances_val2017.json"  
    resFile = 'Results/results_coco_queries.json'
    dataset_name = "coco-2017-validation-project"

    # Load results
    predictions = read_results(resFile, random_selection=False)
    mapping = create_image_id_mapping('Results/instances_val2017.json')
    
    # Check if the dataset already exists
    if dataset_name in fo.list_datasets():
        print(f"Loading existing dataset: {dataset_name}")
        dataset_val_coco = fo.load_dataset(dataset_name)
       
    else:
        # Load the dataset from your project directory
        print(f"Loading dataset from {data_dir}")
        dataset_val_coco = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=data_dir,
            label_field="ground_truth",
            labels_path=labels_path,
            label_types = "detections",
            name=dataset_name,
        )

    print("Dataset: ", dataset_val_coco)

    fouc.add_coco_labels(
        dataset_val_coco,
        label_field="predictions",
        label_type="detections",
        labels_or_path=labels_path,
        categories=ID2CLASS,
    )

    print("Dataset: ", dataset_val_coco)

    predicted_sample_ids = [sample.id for sample in dataset_val_coco if sample["predictions"]]
    filtered_dataset = dataset_val_coco.select(predicted_sample_ids)

    print("Filtered Dataset: ", filtered_dataset)

    sample = filtered_dataset.first()  # Get the first sample (replace with specific sample if needed)
    print("sample: ", sample)
    print("Ground Truth for Image ID:", sample.filepath)
    print(sample["ground_truth"])  # Print the ground truth annotations
    print("Predictions for Image ID:", sample.filepath)   
    print(sample["predictions"])  # Print the predictions

    results = filtered_dataset.evaluate_detections(
        "predictions",
        eval_key="coco_2017_eval",
        method="coco",
        iou=0.75,
        classwise=True,
    )

    dataset_val_coco.save()
    results.print_report()
    print("mAP: ", results.mAP())
    session = fo.launch_app(dataset_val_coco)
    session.wait()