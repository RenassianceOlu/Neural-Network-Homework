import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# 定义 ResNet 的基础残差块 (Residual Block)
class BasicBlock(nn.Module):
    expansion = 1 

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
     
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        out = F.relu(out)
        return out

# 设计 ResNet 网络架构
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
   
        self.linear = nn.Sequential(
        nn.Dropout(0.5),  
        nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)    # 32x32 -> 32x32
        out = self.layer2(out)    # 32x32 -> 16x16
        out = self.layer3(out)    # 16x16 -> 8x8
        out = self.layer4(out)    # 8x8 -> 4x4
       
        out = F.avg_pool2d(out, 4) 
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet-18 (4层，每层2个block)
def CustomResNet():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def main():
    torch.manual_seed(42)

    # 图像预处理与数据增强 
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3)], p=0.7),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    print("正在准备数据集...")
    train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
    test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)

    batch_size = 128 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化全局变量 
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    model = CustomResNet().to(device)
    print("ResNet模型构建完成！")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)  
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)    # 每过 8 轮，学习率衰减乘以 0.1

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1) 
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * correct_train / total_train
        
        # 测试阶段
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()
                
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_acc = 100. * correct_test / total_test
        
        # 学习率更新
        scheduler.step()

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)
        
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1}/{num_epochs}] (LR: {current_lr:.5f}) | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

    print("测试完毕，ResNet 训练结束！")

    # 后处理与可视化
    save_path = 'svhn_classification.pth'
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存至 {save_path}")

    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc (%)', marker='o')
    plt.plot(epochs_range, history['test_acc'], label='Test Acc (%)', marker='s')
    plt.title('Train/Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['test_loss'], label='Test Loss', marker='s')
    plt.title('Train/Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png') 
    print("变化曲线已保存为 learning_curves.png")
    plt.show() 

if __name__ == '__main__':
    main()