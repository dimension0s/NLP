from tqdm.auto import tqdm
import os
import numpy as np
from device import device
# 训练函数

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch):
    total_loss = 0.
    total = 0
    model.train()

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch_data in progress_bar:
        input_ids, intent_labels, slot_labels = batch_data
        # 注意：需要分别.to(device)，不能合起来，因为batch_data是一个元组，元组不支持to(device)操作
        # 应该分别对每个张量调用to(device)
        input_ids = input_ids.to(device)
        intent_labels = intent_labels.to(device)
        slot_labels = slot_labels.to(device)

        outputs = model(
            input_ids=input_ids,
            intent_labels=intent_labels,
            slot_labels=slot_labels,
        )

        loss = outputs['intent_loss'] + outputs['slot_loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

        avg_loss = total_loss / (step + 1)
        progress_bar.set_description(f'epoch:{epoch},avg_loss:{avg_loss:.4f}')
    return avg_loss