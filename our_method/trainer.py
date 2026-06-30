# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from lrp import calculate_lrp
from utils import get_roc_metrics

def evaluate_model_ETD(model, data, DEVICE):
    model.to(DEVICE)
    model.eval()
    eval_loader = DataLoader(data, batch_size=1, shuffle=False)
    epoch_crit_train_original, epoch_crit_train_sampled = [], []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            text = batch
            output = model(text)
            # 计算 LRP 差值 (用于构建 AUROC)
            original_lrp = calculate_lrp(model.scoring_model, model.scoring_tokenizer, text[0], DEVICE) - \
                           calculate_lrp(model.reference_model, model.reference_tokenizer, text[0], DEVICE)
            sampled_lrp = calculate_lrp(model.scoring_model, model.scoring_tokenizer, text[1], DEVICE) - \
                          calculate_lrp(model.reference_model, model.reference_tokenizer, text[1], DEVICE)
            epoch_crit_train_original.append(original_lrp)
            epoch_crit_train_sampled.append(sampled_lrp)

        _, _, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)

    print(f"Validation ROC AUC: {roc_auc:.4f}")
    return {"val_ROC_AUC": roc_auc}

def fine_tune_ours(model, data, DEVICE, ckpt_dir='./ckpt', args=None):
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # 日志目录只保留基础命名
    writer = SummaryWriter(log_dir=f"logs/{args.task_name}_etd_{current_time}/train_ai_detection")

    train_loader = DataLoader(data[0], batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    model.to(DEVICE)

    for epoch in range(args.epochs):
        epoch_crit_train_original, epoch_crit_train_sampled = [], []
        
        for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            text = batch
            # 前向传播 (仅为获取 crit 以计算 AUROC)
            with autocast(device_type='cuda'):
                outputs = model(text)
                epoch_crit_train_original.extend(outputs['crit'][1].tolist())
                epoch_crit_train_sampled.extend(outputs['crit'][3].tolist())
                loss = outputs['loss'] / args.a
                
            scaler.scale(loss).backward()
            if ((epoch * len(train_loader) + _) + 1) % args.a == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()

        # 仅记录 AUROC
        _, _, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        writer.add_scalar('ROC_AUC/epoch', roc_auc, epoch)
        print(f"\nEpoch {epoch} | ROC AUC: {roc_auc:.4f}")

        # 验证时也仅记录 AUROC
        if ((epoch + 1) % args.val_freq) == 0:
            val_result = evaluate_model_ETD(model, data[1], DEVICE)
            writer.add_scalar('val_ROC_AUC/epoch', val_result['val_ROC_AUC'], epoch)

    model.save_pretrained(ckpt_dir)
    writer.close()
    return model