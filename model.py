"""轻量级论文首页分类器与单批次训练步骤。
Compact first-page classifier and a single-batch training step.

默认输入为 [N, 1, 128, 96]，输出为三类原始分数 [N, 3]。
Default inputs have shape [N, 1, 128, 96]; outputs are raw class logits [N, 3].
"""
import torch
from torch import nn
import torch.nn.functional as F


class PaperCNN(nn.Module):
    """两个卷积块加自适应池化与线性分类头。
    Two convolutional blocks with adaptive pooling and a linear classifier.
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.readout = nn.AdaptiveAvgPool2d(output_size=(4, 3))
        self.fc = nn.Linear(
            in_features=32 * 4 * 3,
            out_features=num_classes,
        )

    def forward(self, x):
        # 第一组：提取特征并缩小空间尺寸。
        # Block 1: extract features and reduce spatial dimensions.
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 第二组：继续提取特征并缩小空间尺寸。
        # Block 2: refine features and reduce spatial dimensions again.
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # 将每个通道的空间尺寸统一为 4×3。
        # Pool each channel to a fixed 4×3 spatial grid.
        x = self.readout(x)

        # 将每张图片展平成一个 384 维向量。
        # Flatten each image into a 384-dimensional feature vector.
        x = torch.flatten(x, start_dim=1)

        # 输出类别原始分数；交叉熵会在内部处理归一化。
        # Return raw logits; cross-entropy handles normalization internally.
        logits = self.fc(x)
        return logits


def train_batch(model, images, labels, optimizer):
    """更新一次参数，并返回脱离计算图的损失与预测分数。
    Update parameters once and return detached loss and prediction logits.
    """
    # 1. 清除上一批次的梯度。
    # 1. Clear gradients from the previous batch.
    optimizer.zero_grad()

    # 2. 前向传播，得到原始类别分数。
    # 2. Run the forward pass to obtain raw class logits.
    scores = model(images)

    # 3. 计算当前批次的平均交叉熵。
    # 3. Compute the mean cross-entropy loss for this batch.
    loss = F.cross_entropy(scores, labels, reduction="mean")

    # 4. 反向传播，计算参数的梯度。
    # 4. Backpropagate to compute parameter gradients.
    loss.backward()

    # 5. 根据梯度更新参数。
    # 5. Update parameters using those gradients.
    optimizer.step()

    # 6. 返回统计所需的结果，避免保留计算图。
    # 6. Return values for metrics without retaining the computation graph.
    return loss.detach(), scores.detach()
