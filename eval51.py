import fiftyone as fo
import os
import fiftyone.zoo as foz
from utils import read_results, create_image_id_mapping, map_coco_filenames
from coco_preprocess import ID2CLASS
    

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

    for img_id, data in predictions.items():
        img_filename = map_coco_filenames(mapping, img_id)
        path = os.path.abspath(os.path.join(data_dir, img_filename))
        sample_view = dataset_val_coco.match({"filepath": path})

        if sample_view is not None:
            print("Sample found for ", img_filename)
            sample = sample_view.first()
            detections = []
            for bbox, score, category in zip(data['bboxes'], data['scores'], data['categories']):
                detection = fo.Detection(
                    label=ID2CLASS[category],  # Convert category to string if needed
                    bounding_box=bbox,
                    confidence=score,
                )
                detections.append(detection)

        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

    else:
        print(f"Sample not found for {img_filename}")    

    # Filter the dataset to include only samples with predictions
    predicted_sample_ids = [sample.id for sample in dataset_val_coco if sample["predictions"]]
    filtered_dataset = dataset_val_coco.select(predicted_sample_ids)

     # Ensure ground truth annotations are preserved
    for sample in filtered_dataset:
        if "ground_truth" not in sample:
            sample["ground_truth"] = dataset_val_coco[sample.id]["ground_truth"]
            sample.save()

    print("Filtered dataset: ", filtered_dataset)

    sample = filtered_dataset.first()  # Get the first sample (replace with specific sample if needed)
    print("sample: ", sample)
    print("Ground Truth for Image ID:", sample.coco_id)
    print(sample["ground_truth"])  # Print the ground truth annotations
    print("Predictions for Image ID:", sample.coco_id)   
    print(sample["predictions"])  # Print the predictions


    results = dataset_val_coco.evaluate_detections(
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



