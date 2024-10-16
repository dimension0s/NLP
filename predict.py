# 预测函数
import torch
from model import model,BertForIntention
from transformers import BertConfig
from labels_dict import intent2id,slot2id,id2slot,id2intent,tokenizer
from device import device

class IntentSlotPredictor:
    def __init__(self, model_path, tokenizer, id2intent, id2slot, device):
        self.model = model
        self.tokenizer = tokenizer
        self.id2intent = id2intent
        self.id2slot = id2slot
        self.device = device

        # 加载模型
        checkpoint = 'bert-base-chinese'
        config = BertConfig.from_pretrained(checkpoint)
        intent_label_num = len(intent2id)
        slot_label_num = len(slot2id)

        self.model = BertForIntention(config, intent_label_num=intent_label_num, slot_label_num=slot_label_num)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

    def predict(self, text):
        # 单个文本的预测方法
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensor='pt', max_length=25)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 2.模型推断
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        # 3.获取意图预测结果
        intent_logits = outputs['intent_logits']
        intent_preds = torch.argmax(intent_logits, dim=-1).cpu().numpy()
        intent_label = self.id2intent[intent_preds[0]]

        # 4.获取槽位预测结果
        slot_logits = outputs['slot_logits']
        slot_preds = torch.argmax(slot_logits, dim=-1).cpu().numpy()[0]  # 取第一个样本
        tokenized_words = self.tokenizer.tokenize(text)
        slot_labels = [self.id2slot[idx] for idx in slot_preds[:len(tokenized_words)]]

        return {
            'intent': intent_label,
            'slots': slot_labels
        }

    def batch_predict(self, texts):
        # 批量预测文本
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=25, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # 1.获取意图预测结果
        intent_logits = outputs['intent_logits']
        intent_preds = torch.argmax(intent_logits, dim=-1).cpu().numpy()  # 批量预测意图

        # 2.获取槽位预测结果
        slot_logits = outputs['slot_logits']

        slot_preds = torch.argmax(slot_logits, dim=-1).cpu().numpy()  # 批量预测槽位

        results = []
        for i, text in enumerate(texts):
            intent_label = self.id2intent[intent_preds[i]]  # 获取意图标签
            tokenized_words = self.tokenizer.tokenize(text)
            slot_labels = [self.id2slot[idx] for idx in slot_preds[i][:len(tokenized_words)]]  # 获取槽位标签

            results.append({
                'intent': intent_label,
                'slots': slot_labels,
            })

        return results

# 比如：
texts = ["今天天气怎么样？", "播放一些轻音乐", "预订一本小说","给朋友发信息","下载QQ音乐APP","下载B站网页","打开谷歌浏览器"]

predictor = IntentSlotPredictor(
    model_path = "epoch_48_intent_acc_0.9131_slot_acc_0.9595_model_weights.bin",
    tokenizer=tokenizer,
    id2intent=id2intent,
    id2slot=id2slot,
    device=device)

batch_results = predictor.batch_predict(texts)
for idx,res in enumerate(batch_results):
    print(f'Text{idx+1}:{res}')