from transformers import BertPreTrainedModel, BertModel, BertConfig
from device import device
from labels_dict import slot2id,intent2id,checkpoint
import torch.nn as nn


class BertForIntention(BertPreTrainedModel):
    def __init__(self, config, intent_label_num, slot_label_num):
        super().__init__(config)
        self.intent_label_num = intent_label_num  # 意图分类类别数量
        self.slot_label_num = slot_label_num  # 槽位分类类别数量
        self.bert = BertModel(config)
        # 意图识别分类头
        self.intent_cls = nn.Linear(config.hidden_size, intent_label_num)
        # 槽位标签分类头
        self.slot_cls_ = nn.Linear(config.hidden_size, 3968)
        self.slot_cls = nn.Linear(3968, slot_label_num)

        # 权重衰减
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.post_init()  # 后处理

    def forward(self, input_ids, attention_mask=None, intent_labels=None, slot_labels=None):
        # 输出维度：[batch_size,seq_len,hidden_size]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs.pooler_output  # 池化输出，用于意图分类,也可以写成outputs[1]
        sequence_output = outputs.last_hidden_state  # 序列输出，用于槽位标签分类,也可以写成outputs[0]

        # 对输出进行dropout处理
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)

        # 意图识别分类输出,[batch_size,num_intent_labels]
        intent_logits = self.intent_cls(pooled_output)

        # 槽位标签分类输出,[batch_size,seq_len,num_slot_labels]
        slot_logits_ = self.slot_cls_(sequence_output)
        slot_logits = self.slot_cls(slot_logits_)

        # 如果提供了意图标签和槽位标签，则计算损失
        intent_loss = None
        slot_loss = None

        if intent_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.intent_label_num), intent_labels.view(-1))

        if slot_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充的标签（如 [PAD] 标签）
            slot_loss = loss_fct(slot_logits.view(-1, self.slot_label_num), slot_labels.view(-1))

        return {
            'intent_loss': intent_loss,
            'slot_loss': slot_loss,
            'intent_logits': intent_logits,
            'slot_logits': slot_logits,
        }

intent_label_num = len(intent2id) # 24
slot_label_num = len(slot2id) # 124
# 注：虽然构建了多头模型，但是传入的参数：意图头和槽位头都只有1个，每个头包含若干个类别
# 在本任务中，每个意图头包含24个类别，每个槽位头包含124个类别

config = BertConfig.from_pretrained(checkpoint)
model = BertForIntention(config,intent_label_num=intent_label_num,slot_label_num=slot_label_num)
model = model.to(device)

print(model)