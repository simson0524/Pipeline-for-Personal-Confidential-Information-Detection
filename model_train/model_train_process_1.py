# model_train/model_train_process_1.py

from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from classifier.model import Classifier
from dataset.pipeline_dataset import PipelineDataset, load_all_json
from database.create_dbs import *
from database.db_logics import *
from datetime import datetime
import torch



def model_train_process_1(model, epoch, dataloader, optimizer, device, loss_fn=None):
    # 1. 모델학습 시작시간
    start_time = datetime.now()

    model.train()
    total_loss = 0
    if not loss_fn:
        loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, total=len(dataloader), desc=f"1. 모델학습(epoch{epoch}) 프로세스"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_start = batch["token_start"].to(device)
        token_end = batch["token_end"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_start=token_start,
                        token_end=token_end)
        
        loss = loss_fn(outputs["logits"], labels) # Logits : [0.87, 0.13] -> [1, 0]  /  GT : [0, 1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 1. 모델학습 종료시간
    end_time = datetime.now()

    # 1. 모델학습 소요시간
    duration = end_time - start_time

    # return total_loss / len(dataloader), start_time, end_time, duration
    return {
        'avg_loss': total_loss/len(dataloader),
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration
    }