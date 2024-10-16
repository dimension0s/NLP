# 验证函数
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import precision_score, f1_score, classification_report
import torch
from device import device


def test_loop(dataloader, model, mode='Valid'):
    assert mode in ['Valid', 'Test']
    all_intent_preds, all_intent_labels = [], []
    all_slot_preds, all_slot_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in dataloader:
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

            # 1.意图分类预测
            intent_preds = torch.argmax(outputs['intent_logits'], dim=-1)
            all_intent_preds.extend(intent_preds.cpu().numpy())
            all_intent_labels.extend(intent_labels.cpu().numpy())

            # 2.槽位分类预测
            slot_preds = torch.argmax(outputs['slot_logits'], dim=-1)
            all_slot_preds.extend(slot_preds.view(-1).cpu().numpy())  # 展平预测的槽位标签
            all_slot_labels.extend(slot_labels.view(-1).cpu().numpy())  # 展平真实的槽位标签



    # 3.1 意图分类的准确率
    intent_acc = sk_accuracy_score(all_intent_labels, all_intent_preds)
    # 3.2 槽位填充的准确率
    slot_acc = sk_accuracy_score(all_slot_labels, all_slot_preds)

    # 3.计算P,R,F1值,先出F1值
    # macro avg的f1:宏观平均，对于平衡的数据集更合适
    # weighted avg的f1：加权平均，考虑了每个类别在数据集中出现的频率，适合数据不平衡的情况
    intent_report = classification_report(all_intent_labels, all_intent_preds, output_dict=True, zero_division=1)
    in_macro_f1, in_micro_f1 = intent_report['macro avg']['f1-score'], intent_report['weighted avg']['f1-score']
    slot_report = classification_report(all_slot_labels, all_slot_preds, output_dict=True, zero_division=1)
    sl_macro_f1, sl_micro_f1 = slot_report['macro avg']['f1-score'], slot_report['weighted avg']['f1-score']

    print(f"intent_acc:{intent_acc:.4f}/ intent_macro_f1:{in_macro_f1:.4f}/ intent_micro_f1:{in_micro_f1:.4f}/ \
           slot_acc:{slot_acc:.4f}/slot_macro_f1:{sl_macro_f1:.4f}/slot_micro_f1:{sl_micro_f1:.4f}")


    return intent_acc, slot_acc, intent_report, slot_report