# 构建意图和槽位标签字典

from transformers import AutoTokenizer,BertTokenizer
from data import *

checkpoint = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(checkpoint)

data = MyData("E:\\NLPProject\\意图识别\\data\\data.json")

intent_labels = ['[UNK]']
slot_labels = ['[PAD]','[UNK]','[O]']

# 提取意图和槽位标签，这里暂时没有将其存入单独的文件中：
for item in data:
    if item['intent'] not in intent_labels:
        intent_labels.append(item['intent'])

    assert len(intent_labels) == len(set(intent_labels))

    for slot_name,slot_value in item['slots'].items():
        if 'B_'+slot_name not in slot_labels:
            slot_labels.extend(['I_'+slot_name,'B_'+slot_name])

# 存放到当前目录中
with open('slot_labels.txt','w') as f:
    f.write('\n'.join(slot_labels))

with open('intent_labels.txt','w') as f:
    f.write('\n'.join(intent_labels))

# 配上索引，构成映射字典
intent2id = {intent:idx for idx,intent in enumerate(intent_labels)}
slot2id = {slot:idx+1 for idx,slot in enumerate(slot_labels)}
slot2id['O'] = 0

id2intent = {idx:intent for intent,idx in intent2id.items()}
id2slot = {idx:slot for slot,idx in slot2id.items()}

# 打印测试
print("intent2id mapping:\n",intent2id)
print("id2intent mapping:\n",id2intent)
print("slot2id mapping:\n",slot2id)
print("id2slot mapping:\n",id2slot)


# 设计每个文本分词对应的槽位标签
def get_slot_labels(text,slots,tokenizer):
    text_tokens = tokenizer.tokenize(text)
    slot_labels = []
    i = 0
    while i < len(text_tokens):
        slot_matched = False
        for slot_label,slot_values in slots.items():
            if slot_matched:
                break
            if isinstance(slot_values,str):
                slot_values = [slot_values]  # 如果槽位值是字符串，转换为列表
            for text_pattern in slot_values:
                # 将槽位值和文本一样进行分词，以便后续进行匹配
                pattern_tokens = tokenizer.tokenize(text_pattern)
                # 通过比较分词后的文本片段是否与槽位的模式相同
                if "".join(text_tokens[i:i + len(pattern_tokens)]) == "".join(pattern_tokens):
                    slot_matched = True
                    # B_ 表示槽位的开始（Beginning），即第一个匹配的 token。
                    # I_ 表示槽位的内部（Inside），即后续匹配的 token。
                    # 比如，'name': 'uc' 会被标注为 ['B_name', 'I_name']
                    slot_labels.extend(['B_' + slot_label] + ['I_' + slot_label] * (len(pattern_tokens) - 1))
                    # 将索引 i 移动到匹配的槽位之后，以继续处理剩余的文本。
                    i += len(pattern_tokens)
                    break

        if not slot_matched:
            # 没有匹配任何槽位模式，则将其标记为 [O]，
            # 通常代表不属于任何槽位的 token），并继续下一个 token
            slot_labels.append('[O]')
            i += 1

        return slot_labels
