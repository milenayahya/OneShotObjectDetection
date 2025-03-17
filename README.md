#One-Shot Object Detection
This project implements state-of-the-art zero, one, and few-shot image-guided object detection using the OWLv2 model. It features a highly flexible, end-to-end pipeline, designed with a variety of user-configurable parameters to cater to diverse use cases.

Key features include:

- Evaluation Code: Assess model performance on labeled datasets by calculating and visualizing mAP, precision, recall, and F1 score—both globally and per-category.
- Per-Category Threshold Tuning: Fine-tune detection thresholds for each category, enabling precise control over performance.
- Preprocessing and Postprocessing Techniques: Including supercategory mapping either pre- or post-inference, and both class-agnostic and class-specific Non-Maximum Suppression.
- Visualization Tools: Display detection results via TensorBoard for insightful analysis and debugging.
- Real-time Server Architecture: A scalable, TCP/IP-based server for seamless real-time deployment, introduction of new classes in real-time, and client interaction, ensuring low-latency and efficient operation.
  
This project aims to offer an adaptable solution for quick and efficient object detection, eliminating the need for data collection and labelling, and model training and finetuning.
