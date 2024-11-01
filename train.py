import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  #判断GPU是否可用
print(device)

# 数据预处理 数据增强
transform = transforms.Compose([
    # 对图像进行随机的裁剪crop以后再resize成固定大小（224*224）
    transforms.RandomResizedCrop(224),
    # 随机旋转20度（顺时针和逆时针）
    transforms.RandomRotation(20),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),
    # 将数据转换为tensor
    transforms.ToTensor()
])

# 读取数据
root = './data'   #root是数据集目录
# 获取数据的路径，使用transform增强变化
train_dataset = datasets.ImageFolder(root + '/train', transform)
test_dataset = datasets.ImageFolder(root + '/test', transform)
# 导入数据
# 每个批次8个数据，打乱
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
# 类别名称
classes = train_dataset.classes
# 类别编号
classes_index = train_dataset.class_to_idx
print("类别名称",classes)
print("类别编号",classes_index)
# models.下有很多pytorch提供的训练好的模型
model = models.vgg16(pretrained=True)
# 我们主要是想调用vgg16的卷积层，全连接层自己定义，覆盖掉原来的
# 如果想只训练模型的全连接层（不想则注释掉这个for）
for param in model.parameters():
    param.requires_grad = False
# 构建新的全连接层
# 25088：卷阶层输入的是25088个神经元，中间100是自己定义的，输出类别数量2
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 100),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(100, 2)
                                       # 这里可以加softmax也可以不加
                                       )
model=model.to(device)      #将模型发送到GPU上
print("使用GPU:",next(model.parameters()).device)  # 输出：cuda:0
LR = 0.0001
# 定义代价函数
entropy_loss = nn.CrossEntropyLoss()   #损失函数
# 定义优化器
optimizer = optim.SGD(model.parameters(), LR, momentum=0.9)
print("开始训练~")
train_losses = []
train_accs = []
test_losses = []
test_accs = []

def train():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader):
        # 获得数据和对应的标签
        inputs, labels = data
        inputs,labels=inputs.to(device),labels.to(device)  #将数据发送到GPU上
        # 获得模型预测结果，（64，10）
        out = model(inputs)
        # 交叉熵代价函数out(batch,C),labels(batch)
        loss = entropy_loss(out, labels).to(device)  #别忘了损失函数也要发到GPU
        # 梯度清0
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 修改权值
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(out, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

y_true = []
y_pred = []
def test():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # 获得数据和对应的标签
            inputs, labels = data
            inputs,labels=inputs.to(device),labels.to(device)
            out = model(inputs)
            loss = entropy_loss(out, labels)

            running_loss += loss.item()
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    test_losses.append(epoch_loss)
    test_accs.append(epoch_acc)
    print(f'Test Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}')

    correct = 0
    for i, data in enumerate(train_loader):
        # 获得数据和对应的标签
        inputs, labels = data
        inputs,labels=inputs.to(device),labels.to(device)
        # 获得模型预测结果
        out = model(inputs)
        # 获得最大值，以及最大值所在的位置
        _, predicted = torch.max(out, 1)
        # 预测正确的数量
        correct += (predicted == labels).sum()

epochs = 2;
for epoch in range(0,epochs):
    print('epoch:', epoch)
    train()
    test()
torch.save(model.state_dict(), 'model.pth')
print("~结束训练")

# 绘制损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), test_losses, label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accs, label='Train Accuracy')
plt.plot(range(epochs), test_accs, label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()