import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from osod import find_embeddings, load_query_image_group
import os
from datetime import datetime
from config import PROJECT_BASE_PATH, query_dir, test_dir   
import logging
from tqdm import tqdm
from RunOptions import RunOptions
import json
import numpy as np


log_dir = os.path.join(PROJECT_BASE_PATH, "logs")

# Create a timestamp for the log file name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"debug_{timestamp}.log")
# Setup logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
        ],
    )
logger = logging.getLogger(__name__)

class ClassTokenTuner:
    def __init__(self, model, lr= 1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.visual_projection = model.visual_projection
        self.optimizer = AdamW(self.visual_projection.parameters(), lr=lr, weight_decay=0.0)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        #self.criterion = nn.CosineEmbeddingLoss()
        #self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()
        self.device = device

    def compute_loss(self, query_embeddings, cls_token, labels):
        #query_embeddings = query_embeddings.to(self.device)
        #cls_token = torch.squeeze(cls_token[0], dim=0).to(self.device).requires_grad_()
        
        query_embeddings = torch.stack(query_embeddings).to(self.device)
        cls_token = torch.cat(cls_token, dim=0).to(self.device).requires_grad_()
        
        labels = torch.tensor(labels).to(self.device)
        loss = self.criterion(query_embeddings, cls_token)
        return loss
    
    def train_step(self, query_embeddings, cls_token, labels):
        self.optimizer.zero_grad()
        loss = self.compute_loss(query_embeddings, cls_token, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(model, processor, options, device, logger, lr = 1e-4, num_epochs = 100, min_delta=1e-4, patience=20):

        # Freeze all parameters except the visual projection layer
        for param in model.parameters():
            param.requires_grad = False
        model.visual_projection.requires_grad_(True)
        
        model.to(device)
        images, classes = zip(*load_query_image_group(options))
        images = list(images)
        classes = list(classes)

        file = os.path.join(query_dir, f"classes_{options.data}.json")
        with open(file, 'r') as f:
            emb_classes = json.load(f)

        query_embeddings = []
        query_embeddings_all = torch.load(os.path.join(query_dir,f'query_embeddings_{options.data}_gpu.pth'))
        idx = [i for i, class_value in enumerate(emb_classes) if class_value in {3.0, 7.0, 8.0}]
        print("idx", idx)

        for i in idx:
            query_embeddings.append(query_embeddings_all[i])

        print("len query embeddings", len(query_embeddings))


        labels = torch.ones(len(images))

        class_token_tuner = ClassTokenTuner(model, lr=lr, device=device)
        best_loss = float('inf')
        epochs_no_improve = 0
        total_batches = (len(images) + options.query_batch_size - 1) // options.query_batch_size

        with tqdm(total=num_epochs, desc="Training") as pbar:
            for i in range(num_epochs):
                total_loss = 0.0

                for batch_idx in range(0, len(images), options.query_batch_size):                        
                    cls_token, first_token = find_embeddings(images[batch_idx:batch_idx+options.query_batch_size], model, processor, options)
                    if i == 0:
                        cls_token = first_token

                    #batch_query_embeddings = query_embeddings[0]
                    #batch_labels = labels[0]

                    batch_query_embeddings = query_embeddings[batch_idx:batch_idx + options.query_batch_size]
                    batch_labels = labels[batch_idx:batch_idx + options.query_batch_size]
                   
                    loss = class_token_tuner.train_step(batch_query_embeddings, cls_token, batch_labels)
                    total_loss += loss
                
                avg_loss = total_loss / total_batches
                logger.info(f"Epoch {i+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                pbar.update(1)
                pbar.set_postfix({"Epoch loss":avg_loss})

                class_token_tuner.scheduler.step()

                if avg_loss < best_loss - min_delta:
                  #  torch.save(model.state_dict(), f"best_model_epoch_{i+1}.pth")
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {i + 1} epochs.")
                    break
            


            print("Training complete.")
            return model    
            

if __name__ == "__main__":

    options = RunOptions(
        mode = "test",
        source_image_paths= os.path.join(query_dir, "FT"),
        target_image_paths= os.path.join(test_dir, "Comau/3D"), 
        data="MGN",
        comment="get_feats", 
        query_batch_size=8, 
        manual_query_selection=False,
        confidence_threshold= 0.95,
        test_batch_size=8, 
        k_shot=1,
        topk_test= 10,
        visualize_query_images=True,
        nms_between_classes=False,
        nms_threshold=0.3,
        write_to_file_freq=5,
    )
   
    model = options.model.from_pretrained(options.backbone)
    processor = options.processor.from_pretrained(options.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ClassTokenTuner.train(model, processor, options, device, logger, lr = 1e-4, num_epochs = 100, min_delta=1e-4, patience=20)
    torch.save(model.state_dict(), "model-FT-3_2.pth")