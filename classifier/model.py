# classifier/model.py

import torch.nn.functional as F
import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, pretrained_bert, num_labels, use_focal=False, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()

        # Classifier basic config
        self.pretrained_bert = pretrained_bert
        self.hidden_size     = pretrained_bert.config.hidden_size
        self.num_labels      = num_labels

        # Focal loss config
        self.use_focal = use_focal
        self.alpha     = alpha     # 희귀 클래스에 주는 가중치
        self.gamma     = gamma     # 정답 확률이 높은 샘플은 무시 <=> 정답 확률이 낮은 샘플은 강조
        self.reduction = reduction

        # Classifier head structure
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_labels)
        )

    def focal_loss(self, logits, labels):
        # 기본 CE Loss
        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        # 확률값
        pt = torch.exp(-ce_loss)

        # Focal Loss
        focal_loss = self.alpha * (1-pt) ** self.gamma * ce_loss

        """
        [질문] 이거 왜 reduction 그냥 F.cross_entropy 파라미터로 안쓰고 이렇게 한건지?
        """
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
        
    def forward(self, input_ids, attention_mask, token_start, token_end, labels=None):
        outputs = self.pretrained_bert(
            input_ids=input_ids,          # shape: (batch_size, token_len)
            attention_mask=attention_mask # shape: (batch_size, token_len)
        )

        # 각 입력 토큰에 대한 마지막 hidden state
        last_hidden = outputs.last_hidden_state # shape: (batch_size, token_len, hidden_size)

        # Batch-wise 인덱싱
        batch_size   = input_ids.size(0)
        start_embeds = last_hidden[torch.arange(batch_size), token_start] # shape: (batch_size, hidden_size)
        end_embeds   = last_hidden[torch.arange(batch_size), token_end]   # shape: (batch_size, hidden_size)

        # 임베딩 Concat
        span_representation = torch.cat([start_embeds, end_embeds], dim=-1) # shape: (batch_size, hidden_size*2)

        # 분류 결과(logits)
        logits = self.classifier( span_representation )

        if labels is not None:
            if self.use_focal:
                loss = self.focal_loss(logits=logits, labels=labels)
            else:
                loss = F.cross_entropy(input=logits, target=labels)

            return {
                "logits": logits,
                "loss": loss
                }