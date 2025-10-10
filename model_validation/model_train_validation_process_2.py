# model_validation/model_train_validation_process_2.py

from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from database.db_logics import insert_many_rows
from datetime import datetime
import torch


def model_train_validation_process_2(conn, experiment_name, model, epoch, dataloader, device, label_2_id, id_2_label):
    # 2. 모델학습검증 시작시간
    start_time = datetime.now()
    
    model.eval()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    preds, targets = [], []

    # 불일치 샘플 목록
    model_train_sent_dataset_log = []

    with torch.no_grad():        
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'2. 모델학습검증(epoch{epoch}) 프로세스'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_start = batch["token_start"].to(device)
            token_end = batch["token_end"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_start=token_start,
                            token_end=token_end,
                            labels=labels)
            
            loss = outputs['loss']
            total_loss += loss.item()

            pred_labels = torch.argmax(outputs['logits'], dim=-1)
            preds.extend(pred_labels.cpu().tolist())
            targets.extend(labels.cpu().tolist())

            batch_size = labels.size(0)
            for i in range(batch_size):
                # model_train_sent_dataset_log 테이블에 들어가는 scheme
                model_train_sent_dataset_log_scheme = (
                    experiment_name,
                    batch['sentence_id'][i],
                    epoch,
                    batch['sentence'][i],
                    batch['span_token'][i],
                    batch['idx'][i].item(),
                    id_2_label[labels[i].item()],
                    id_2_label[pred_labels[i].item()],
                    batch['file_name'][i],
                    batch['sentence_seq'][i]
                )
                model_train_sent_dataset_log.append( model_train_sent_dataset_log_scheme )

    # DB(model_train_sent_dataset_log)에 정보 추가하기
    insert_many_rows(conn, "model_train_sent_dataset_log", model_train_sent_dataset_log)

    metric = [ [0 for i in range(len(label_2_id))] for _ in range(len(label_2_id)) ]

    for i, (pred, gt) in enumerate(zip(preds, targets)):
        metric[pred][gt] += 1
                            
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    # 2. 모델학습검증 종료시간
    end_time = datetime.now()

    # 2. 모델학습검증 소요시간
    duration = end_time - start_time

    return {
        'avg_loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'metric': metric,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration
    }