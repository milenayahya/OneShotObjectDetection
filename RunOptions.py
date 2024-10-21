from pathlib import Path
from typing import Literal
import argparse
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torch.utils.tensorboard import SummaryWriter

class RunOptions:
    def __init__(self, 
                 backbone: str = "google/owlv2-base-patch16-ensemble",
                 source_image_paths: str = "ImageNet_query_fewshot/",
                 target_image_paths: str = "ImageNet_test/",
                 comment: str = "_3shot_on_ImageNet",
                 query_batch_size: int = 4,
                 test_batch_size: int = 4,
                 topk_query: int = 3,
                 topk_test: int = None,
                 k_shot = None,
                 manual_query_selection: bool = False,
                 confidence_threshold: float = 0.96,
                 visualize_query_images: bool = True,
                 visualize_test_images: bool = True,
                 nms_between_classes: bool = True,
                 nms_threshold: float = 0.3):
        self.model = Owlv2ForObjectDetection  
        self.processor = Owlv2Processor 
        self.backbone = backbone
        self.source_image_paths = source_image_paths
        self.target_image_paths = target_image_paths
        self.comment = comment
        self.query_batch_size = query_batch_size
        self.test_batch_size = test_batch_size
        self.topk_query = topk_query
        self.topk_test = topk_test
        self.k_shot = k_shot
        self.manual_query_selection = manual_query_selection
        self.confidence_threshold = confidence_threshold
        self.visualize_query_images = visualize_query_images
        self.visualize_test_images = visualize_test_images
        self.nms_between_classes = nms_between_classes
        self.nms_threshold = nms_threshold

    @classmethod
    def from_args(cls, args):
        """Creates an instance of RunOptions from parsed command-line arguments."""
        return cls(
            model = args.model,
            processor = args.processor,
            backbone=args.backbone,
            source_image_paths=args.source_image_paths,
            target_image_paths=args.target_image_paths,
            comment=args.comment,
            query_batch_size=args.query_batch_size,
            test_batch_size=args.test_batch_size,
            topk_query=args.topk_query,
            topk_test=args.topk_test,
            k_shot = None,
            manual_query_selection=args.manual_query_selection,
            confidence_threshold=args.confidence_threshold,
            visualize_query_images=args.visualize_query_images,
            visualize_test_images=args.visualize_test_images,
            nms_between_classes=args.nms_between_classes,
            nms_threshold=args.nms_threshold,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Run options for one-shot object detection.")
    parser.add_argument("--backbone", type=str, default="google/owlv2-base-patch16-ensemble",
                        help="The backbone model to use.")
    parser.add_argument("--source_image_paths", type=str, default="query_images/",
                        help="Path to source images for querying.")
    parser.add_argument("--target_image_paths", type=str, default="test_images/",
                        help="Path to target images for testing.")
    parser.add_argument("--comment", type=str, default="",
                        help="Optional comment for this run.")
    parser.add_argument("--query_batch_size", type=int, default=4,
                        help="Batch size for query images.")
    parser.add_argument("--test_batch_size", type=int, default=4,
                        help="Batch size for test images.")
    parser.add_argument("--topk_query", type=int, default=3,
                        help="Top k objectnesses in query images.")  
    parser.add_argument("--topk_test", type=int, default=3,
                        help="Top k predictions kept in test images.")  
    parser.add_argument("--k_shot", type=int, default=3,
                        help="K-Shot Object Detection")                   
    parser.add_argument("--manual_query_selection", action="store_true",
                        help="Manually select query images.")
    parser.add_argument("--confidence_threshold", type=float, default=0.96,
                        help="Confidence threshold for detections.")
    parser.add_argument("--visualize_query_images", action="store_true",
                        help="Visualize query images during processing.")
    parser.add_argument("--visualize_test_images", action="store_true",
                        help="Visualize test images during processing.")
    parser.add_argument("--nms_between_classes", action="store_true",
                        help="Apply non-maximum suppression between classes.")
    parser.add_argument("--nms_threshold", type=float, default=0.3,
                        help="Threshold for non-maximum suppression.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    options = RunOptions.from_args(args)

    writer = SummaryWriter(comment=options.comment)
    model = options.model.from_pretrained(options.backbone)
    processor = options.processor.from_pretrained(options.backbone)

    print(options.__dict__)  