from TextRCNN.model import Config, Model
import numpy as np
import torch
import time
from utils import build_dataset, build_iterator, get_time_dif
from train_eval import train, evaluate
from utils import build_onedata

# config = Config("./")
config = Config("./", embedding = 'random')
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
vocab, train_data, dev_data, test_data = build_dataset(config, False)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

config.n_vocab = len(vocab)
model = Model(config).to(config.device)
print(model.parameters)
train(config, model, train_iter, dev_iter, test_iter)


text = "请问去哪里能借到低息贷款？"
print("待预测分类的短文本：",text)
textid = build_onedata(config,text)
print('文本转换成词id：',textid)
inpt = torch.LongTensor([textid]).to(config.device)
print("输入模型的数据：",inpt)

# model.eval()
with torch.no_grad():
    outputs = model(inpt)
    print("预测结果：",outputs.data)
    predic = torch.max(outputs.data, 1)[1].cpu().numpy()
    print("预测分类标号：",predic)