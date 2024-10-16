import json
from torch.utils.data import DataLoader,Dataset,random_split

# 构建数据集
# 加载数据
class MyData(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        with open(data_file,'r',encoding='utf-8') as f:
            Data = json.load(f)
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = MyData("E:\\NLPProject\\意图识别\\data\\data.json")
train_data,valid_data = random_split(data,[1850,794])


# 打印数据集测试
print(data[0])
print(train_data[0])
print(valid_data[0])
print(len(data))