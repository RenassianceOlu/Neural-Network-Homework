import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.ao.quantization as quant
from torch.ao.quantization import QuantStub, DeQuantStub
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy
import sys
import warnings
warnings.filterwarnings("ignore")

# ======================== 全局配置 ========================
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

BATCH_SIZE = 128
NUM_CALIBRATION = 2000                
NUM_WARMUP = 50                       
NUM_TIMING = 200                      
DATA_PATH = "../Classification_Homework/data"
MODEL_PATH = "../Classification_Homework/svhn_classification.pth"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ======================== 模型定义 ========================
class BasicBlock(nn.Module):
    """
    基础残差块。使用 nn.ReLU 模块（非 F.relu）以支持 PyTorch 融合量化。
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)          

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.skip_add.add(out, identity)
        out = self.relu2(out)
        return out

    def fuse_manual(self):
        """手动触发本 block 内的模块融合"""
        # conv1 + bn1 + relu1
        quant.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        # conv2 + bn2
        quant.fuse_modules(self, ['conv2', 'bn2'], inplace=True)
        # shortcut conv + bn
        if len(self.shortcut) > 0:
            quant.fuse_modules(self.shortcut, ['0', '1'], inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.quant = QuantStub()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion, num_classes)
        )

        self.dequant = DeQuantStub()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        out = self.dequant(out)
        return out

    def fuse_manual(self):
        """手动融合所有可融合模块"""
        quant.fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)

        for name, child in self.named_children():
            if name.startswith('layer'):
                for block in child:
                    block.fuse_manual()


def CustomResNet(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# ======================== 数据加载 ========================
def get_data_loaders():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.SVHN(
        root=DATA_PATH, split='test', download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    calib_dataset = torchvision.datasets.SVHN(
        root=DATA_PATH, split='train', download=True, transform=test_transform)
    indices = torch.randperm(len(calib_dataset))[:NUM_CALIBRATION]
    calib_subset = Subset(calib_dataset, indices)
    calib_loader = DataLoader(calib_subset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    return test_loader, calib_loader


# ======================== Task 1: FP32 基线 ========================
def load_fp32_model(device=GPU):
    model = CustomResNet()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_accuracy(model, dataloader):
    """计算分类准确率（数据自动送到模型所在 device）"""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        try:
            device = next(model.buffers()).device
        except StopIteration:
            device = CPU
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def get_model_size(model):
    """返回模型大小（MB），兼容 FP32 和 INT8 量化模型"""
    total_bytes = 0
    for val in model.state_dict().values():
        if isinstance(val, torch.Tensor):
            total_bytes += val.numel() * val.element_size()
    return total_bytes / (1024 * 1024)


def measure_inference_latency(model, dataloader):
    """测量 CPU 上单张图片的平均推理延迟（ms）—— 作业要求 CPU 对比"""
    model_cpu = copy.deepcopy(model).to(CPU)
    model_cpu.eval()
    images, _ = next(iter(dataloader))
    single_image = images[0:1].to(CPU)

    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model_cpu(single_image)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_TIMING):
            _ = model_cpu(single_image)
    end = time.perf_counter()

    avg_ms = (end - start) / NUM_TIMING * 1000
    return avg_ms


# ======================== Task 2: 手动量化函数 ========================
def linear_quantize(x, num_bits=8):
    qmin = -(2 ** (num_bits - 1))       
    qmax = (2 ** (num_bits - 1)) - 1     

    x_min = x.min().item()
    x_max = x.max().item()

    if x_max == x_min:
        scale = 1.0
        zero_point = 0
        q = torch.zeros_like(x, dtype=torch.int8)
        return q, scale, zero_point

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - x_min / scale
    zero_point = int(round(zero_point))

    zero_point = max(qmin, min(qmax, zero_point))

    q = torch.round(x / scale + zero_point)
    q = q.clamp(qmin, qmax)
    q = q.to(torch.int8)

    return q, scale, zero_point


def linear_dequantize(q, scale, zero_point):
    """
    反量化：整数量化张量 → 浮点张量。
    参数:
        q: int8 tensor
        scale: 量化缩放因子
        zero_point: 零点
    返回:
        x_hat: 反量化后的 float tensor
    """
    return (q.float() - zero_point) * scale


def manual_quantize_verify(model, dataloader):
    print("\n--- 手动量化验证（逐层 per-tensor）---")
    manual_model = copy.deepcopy(model) 
    manual_model.eval()

    for name, param in manual_model.named_parameters():
        if param.requires_grad and 'weight' in name and param.dim() > 1:
            w = param.data
            q, scale, zp = linear_quantize(w, num_bits=8)
            w_hat = linear_dequantize(q, scale, zp)
            param.data.copy_(w_hat)

    acc = evaluate_accuracy(manual_model, dataloader)
    print(f"手动量化后准确率: {acc:.2f}%")
    return acc


# ======================== Task 3: INT8 PTQ ========================
def prepare_fp32_model_for_qat(model):
    model.qconfig = quant.get_default_qconfig('x86') 

    model.fuse_manual()

    model_prepared = quant.prepare(model, inplace=False)

    return model_prepared


def calibrate_model(model_prepared, calib_loader):
    print(f"\n正在校准...（{NUM_CALIBRATION} 张图片）")
    model_prepared.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            images = images.to(CPU)
            _ = model_prepared(images)
    print("校准完成！")


def convert_to_int8(model_prepared):
    model_int8 = quant.convert(model_prepared, inplace=False)
    return model_int8


def get_layer_outputs(model, images):
    outputs = {}

    def hook_fn(name):
        def fn(module, inp, outp):
            outputs[name] = outp.detach().cpu()
        return fn

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU,
                                nn.BatchNorm2d, nn.Dropout,
                                nn.AdaptiveAvgPool2d,
                                torch.nn.intrinsic.ConvReLU2d,
                                torch.nn.intrinsic.LinearReLU,
                                torch.nn.quantized.Conv2d,
                                torch.nn.quantized.Linear,
                                quant.QuantStub,                 
                                quant.DeQuantStub,               
                                torch.nn.quantized.Quantize,     
                                torch.nn.quantized.DeQuantize)): 
            handles.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        model(images)

    for h in handles:
        h.remove()

    return outputs


def compute_per_layer_mse(fp32_model, int8_model, dataloader):
    print("\n--- 计算逐层 MSE ---")
    images, _ = next(iter(dataloader))
    images = images[:64].to(CPU)

    fp32_cpu = copy.deepcopy(fp32_model).to(CPU)
    fp32_cpu.eval()
    fp32_outputs = get_layer_outputs(fp32_cpu, images)
    int8_outputs = get_layer_outputs(int8_model, images)

    mse_dict = {}
    for name, fp32_out in fp32_outputs.items():
        if name in int8_outputs:
            int8_out = int8_outputs[name]
            if fp32_out.shape == int8_out.shape:
                if int8_out.is_quantized:
                    int8_out = int8_out.dequantize()
                if fp32_out.is_quantized:
                    fp32_out = fp32_out.dequantize()
                    
                mse = F.mse_loss(fp32_out.float(), int8_out.float()).item()
                mse_dict[name] = mse

    if mse_dict:
        print(f"共计算 {len(mse_dict)} 层的 MSE")
        for name, mse in mse_dict.items():
            print(f"  {name}: MSE = {mse:.6f}")
    else:
        print("警告：未找到可匹配的层（量化后模块名可能变化，请检查）")

    return mse_dict


def evaluate_int8_model(int8_model, dataloader):
    print("\n--- 评估 INT8 模型 ---")
    acc = evaluate_accuracy(int8_model, dataloader)
    print(f"INT8 模型测试准确率: {acc:.2f}%")
    return acc


# ======================== 可视化 ========================
def plot_results(fp32_acc, int8_acc, manual_acc, fp32_latency, int8_latency,
                 fp32_size, int8_size, mse_dict, save_dir='.'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 子图 1：量化前后精度对比 ---
    models_labels = ['FP32', 'INT8 (PTQ)']
    acc_values = [fp32_acc, int8_acc]
    if manual_acc is not None:
        models_labels.append('INT8 (Manual)')
        acc_values.append(manual_acc)
    colors = ['#2ecc71', '#e74c3c', '#3498db'][:len(models_labels)]

    bars1 = axes[0].bar(models_labels, acc_values, color=colors, width=0.4)
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Quantization Accuracy Comparison')
    axes[0].set_ylim([min(acc_values) - 2, max(acc_values) + 1])
    for bar, val in zip(bars1, acc_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # --- 子图 2：推理延迟对比 ---
    lat_labels = ['FP32', 'INT8']
    lat_values = [fp32_latency, int8_latency]
    bars2 = axes[1].bar(lat_labels, lat_values, color=['#2ecc71', '#e74c3c'], width=0.4)
    axes[1].set_ylabel('Latency per Image (ms)')
    axes[1].set_title('Inference Latency Comparison (CPU)')
    speedup = fp32_latency / int8_latency if int8_latency > 0 else 1
    for bar, val in zip(bars2, lat_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f} ms', ha='center', va='bottom', fontweight='bold')
    axes[1].text(0.5, 0.95, f'Speedup: {speedup:.2f}×',
                 transform=axes[1].transAxes, ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1].grid(axis='y', alpha=0.3)

    # --- 子图 3：模型大小 & 压缩比 ---
    size_labels = ['FP32', 'INT8']
    size_values = [fp32_size, int8_size]
    bars3 = axes[2].bar(size_labels, size_values, color=['#2ecc71', '#e74c3c'], width=0.4)
    axes[2].set_ylabel('Model Size (MB)')
    axes[2].set_title('Model Size Comparison')
    ratio = fp32_size / int8_size if int8_size > 0 else 1
    for bar, val in zip(bars3, size_values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{val:.2f} MB', ha='center', va='bottom', fontweight='bold')
    axes[2].text(0.5, 0.95, f'Compression: {ratio:.2f}×',
                 transform=axes[2].transAxes, ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'quantization_comparison.png'), dpi=150)
    plt.show()
    print(f"可视化结果已保存至 {os.path.join(save_dir, 'quantization_comparison.png')}")

    # --- 单独：逐层 MSE 柱状图 ---
    if mse_dict:
        plt.figure(figsize=(14, 6))
        names = list(mse_dict.keys())
        mses = list(mse_dict.values())
        short_names = [n if len(n) <= 40 else '...' + n[-37:] for n in names]
        plt.bar(range(len(names)), mses, color='steelblue')
        plt.xticks(range(len(names)), short_names, rotation=90, fontsize=6)
        plt.ylabel('MSE')
        plt.title('Per-Layer Quantization MSE (FP32 vs INT8)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'layer_mse.png'), dpi=150)
        plt.show()
        print(f"逐层 MSE 图已保存至 {os.path.join(save_dir, 'layer_mse.png')}")

def main():
    print("=" * 60)
    print("SVHN 模型 INT8 静态量化实验")
    print("=" * 60)
    print(f"设备 — GPU: {'RTX 5060 Laptop' if torch.cuda.is_available() else 'N/A'}  |  CPU: cpu")

    print("\n>>> 加载数据...")
    test_loader, calib_loader = get_data_loaders()

    # ============ Task 1: FP32 基线 ============
    print("\n" + "=" * 60)
    print("Task 1: FP32 基线模型")
    print("=" * 60)

    fp32_model = load_fp32_model()
    fp32_acc = evaluate_accuracy(fp32_model, test_loader)
    fp32_size = get_model_size(fp32_model)
    fp32_latency = measure_inference_latency(fp32_model, test_loader)

    print(f"FP32 测试准确率: {fp32_acc:.2f}%")
    print(f"FP32 模型大小: {fp32_size:.2f} MB")
    print(f"FP32 CPU 推理延迟/张: {fp32_latency:.3f} ms")

    # ============ Task 2: 手动量化函数 ============
    print("\n" + "=" * 60)
    print("Task 2: 手动线性量化/反量化函数")
    print("=" * 60)

    test_tensor = torch.randn(10, 10) * 2 + 1
    q, s, zp = linear_quantize(test_tensor, num_bits=8)
    x_hat = linear_dequantize(q, s, zp)
    mse = F.mse_loss(test_tensor, x_hat).item()
    print(f"手动量化函数测试 — 原始值范围: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    print(f"  scale={s:.4f}, zero_point={zp}")
    print(f"  量化-反量化 MSE = {mse:.6f}")

    manual_acc = manual_quantize_verify(fp32_model, test_loader)

    # ============ Task 3: INT8 PTQ ============
    print("\n" + "=" * 60)
    print("Task 3: INT8 静态量化 (PTQ)")
    print("=" * 60)

    ptq_model = CustomResNet()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    ptq_model.load_state_dict(state_dict, strict=False)
    ptq_model.to(CPU)
    ptq_model.eval()

    print("\n>>> 融合模块 & 插入观察器...")
    model_prepared = prepare_fp32_model_for_qat(ptq_model)

    calibrate_model(model_prepared, calib_loader)

    print("\n>>> 转换为 INT8 模型...")
    int8_model = convert_to_int8(model_prepared)
    int8_model.to(CPU)
    int8_model.eval()
    print("INT8 模型转换完成！")

    int8_acc = evaluate_int8_model(int8_model, test_loader)
    int8_size = get_model_size(int8_model)
    int8_latency = measure_inference_latency(int8_model, test_loader)

    print(f"INT8 模型大小: {int8_size:.2f} MB")
    print(f"INT8 CPU 推理延迟/张: {int8_latency:.3f} ms")

    mse_dict = compute_per_layer_mse(fp32_model, int8_model, test_loader)

    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"{'指标':<25} {'FP32':<15} {'INT8 (PTQ)':<15} {'变化'}")
    print("-" * 70)
    print(f"{'测试准确率 (%)':<25} {fp32_acc:<15.2f} {int8_acc:<15.2f} "
          f"{'↓ ' + str(round(fp32_acc - int8_acc, 2)) + '%' if fp32_acc > int8_acc else '↑ ' + str(round(int8_acc - fp32_acc, 2)) + '%'}")
    print(f"{'模型大小 (MB)':<25} {fp32_size:<15.2f} {int8_size:<15.2f} "
          f"压缩 {fp32_size / int8_size:.2f}×")
    print(f"{'推理延迟/张 (ms)':<25} {fp32_latency:<15.3f} {int8_latency:<15.3f} "
          f"加速 {fp32_latency / int8_latency:.2f}×")
    print(f"{'精度损失 (%)':<25} {'—':<15} {round(fp32_acc - int8_acc, 2):<15} —")

    print("\n>>> 绘制可视化图表...")
    plot_results(fp32_acc, int8_acc, manual_acc,
                 fp32_latency, int8_latency,
                 fp32_size, int8_size,
                 mse_dict, save_dir='.')

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
