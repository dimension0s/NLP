# 主循环

from transformers import AdamW, get_scheduler
import torch
import random
import os
import numpy as np
from collate_fn import train_dataloader,valid_dataloader
from train import train_loop
from test import test_loop
from model import model

learning_rate = 1e-5
epoch_num = 50  # 已经训练了58轮，还可以再尝试几轮，比如一共可以尝试70轮


def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(42)

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader))

total = 0
best_acc = 0.
for epoch in range(epoch_num):
    print(f'Epoch{epoch + 1}/{epoch_num}...........')
    train_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch + 1)
    intent_acc, slot_acc, intent_report, slot_report = test_loop(valid_dataloader, model, 'Valid')
    if intent_acc > best_acc:
        best_acc = intent_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(),
                   f'epoch_{epoch + 1}_intent_acc_{intent_acc:.4f}_slot_acc_{slot_acc:.4f}_model_weights.bin')

print("Done!")