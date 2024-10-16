# 分批处理，分词，编码
import torch
from labels_dict import get_slot_labels,tokenizer,intent2id,slot2id
from data import train_data,valid_data
from torch.utils.data import DataLoader

def collate_fn(batch_samples):
    batch_input_ids,batch_slot_ids,batch_intent_ids = [],[],[]
    max_seq_len = 0
    for sample in batch_samples:
        slot_labels = get_slot_labels(sample['text'],sample['slots'],tokenizer)
        slot_ids = [slot2id.get(slot_label,slot2id['[UNK]']) for slot_label in slot_labels]
        # 获取意图标签
        # 注意这里使用列表，因为每个样本通常只有一个意图
        intent_ids = [intent2id.get(sample['intent'],intent2id['[UNK]'])]
        # 对文本编码
        input_ids = tokenizer.encode(sample['text'], add_special_tokens=True)

        # 更新最大序列长度
        max_seq_len = max(max_seq_len, len(input_ids))

        batch_input_ids.append(input_ids)
        batch_slot_ids.append(slot_ids)
        batch_intent_ids.append(intent_ids)

        # 对输入序列和槽位标签进行填充
        # 使用 tokenizer 的填充标记填充 input_ids
    pad_token_id = tokenizer.pad_token_id
    batch_inputs_ids = [input_ids + [pad_token_id] * (max_seq_len - len(input_ids))
                        for input_ids in batch_input_ids]

    # 使用槽位标签的填充标记进行填充
    pad_slot_id = slot2id['[PAD]']
    batch_slots_ids = [slot_ids + [pad_slot_id] * (max_seq_len - len(slot_ids))
                       for slot_ids in batch_slot_ids]

    batch_inputs_ids = torch.tensor(batch_inputs_ids, dtype=torch.long)
    batch_intent_ids = torch.tensor(batch_intent_ids, dtype=torch.long)
    batch_slots_ids = torch.tensor(batch_slots_ids, dtype=torch.long)

    return batch_inputs_ids, batch_intent_ids, batch_slots_ids

train_dataloader = DataLoader(train_data,batch_size=8,shuffle=True,collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data,batch_size=8,shuffle=False,collate_fn=collate_fn)


# 打印数据集
input_ids,intent_ids,slot_ids = next(iter(train_dataloader))
print(input_ids)
print(intent_ids)
print(slot_ids)