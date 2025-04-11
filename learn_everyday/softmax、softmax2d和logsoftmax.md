在深度学习中，`softmax`、`softmax2d` 和 `logsoftmax` 是常用的归一化操作，但它们有不同的应用场景和计算方式。下面我们分别实现它们的计算逻辑，并解释它们的用途和区别。

---

## **1. Softmax**
**作用**：将输入张量的值转换为概率分布（所有值在 `[0,1]` 之间，且和为 `1`）。  
**公式**：
\[
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$
\]
**实现**：

```python
import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 防止数值溢出，减去最大值
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)
```
**说明**：
- `dim` 指定在哪个维度计算 `softmax`（默认 `-1` 表示最后一个维度）。
- 先减去最大值（`x_max`）防止数值溢出（`exp` 可能导致 `inf`）。
- 适用于 NLP（Transformer 注意力）、分类任务等。

---

## **2. Softmax2D**
**作用**：对 2D 空间（如图像）的每个位置进行 `softmax`，常用于语义分割（如 FCN、UNet）。  
**公式**：
\[
$$
\text{softmax2d}(x_{h,w,c}) = \frac{e^{x_{h,w,c}}}{\sum_{c'} e^{x_{h,w,c'}}}
$$
\]
**实现**：
```python
def softmax2d(x: torch.Tensor) -> torch.Tensor:
    # 输入形状: (B, C, H, W)
    # 对每个像素点 (H,W) 在通道维度 C 上做 softmax
    x_max = x.max(dim=1, keepdim=True).values  # (B, 1, H, W)
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=1, keepdim=True)
```
**说明**：
- 输入通常是 `(B, C, H, W)`（`B`=batch, `C`=channels, `H`=height, `W`=width）。
- 对每个像素点 `(H, W)` 在通道维度 `C` 上做 `softmax`，得到每个像素的分类概率。
- 主要用于 **语义分割**（如 UNet、DeepLab），输出每个像素的类别概率。

---

## **3. LogSoftmax**
**作用**：计算 `log(softmax(x))`，用于 **数值稳定性** 和 **交叉熵损失（NLLLoss）**。  
**公式**：
\[
$$
\text{logsoftmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right) = x_i - \log\left(\sum_j e^{x_j}\right)
$$
\]
**实现**：
```python
def logsoftmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 防止数值溢出，减去最大值
    x_max = x.max(dim=dim, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(x - x_max).sum(dim=dim, keepdim=True))
    return (x - x_max) - log_sum_exp
```
**说明**：
- 相比 `softmax`，`logsoftmax` 直接计算对数，避免 `log(softmax(x))` 的数值不稳定问题。
- 主要用于 **交叉熵损失（CrossEntropyLoss = NLLLoss + LogSoftmax）**，因为：
  - `logsoftmax` + `NLLLoss` 比 `softmax` + `CrossEntropy` 更稳定（避免 `log(0)` 导致 `-inf`）。
  - PyTorch 的 `CrossEntropyLoss` 内部就是 `LogSoftmax + NLLLoss`。

---

## **为什么需要 `softmax2d` 和 `logsoftmax`？**
| 方法         | 用途                  | 适用场景                       |
| ------------ | --------------------- | ------------------------------ |
| `softmax`    | 归一化概率分布        | Transformer 注意力、分类任务   |
| `softmax2d`  | 2D 空间归一化         | 语义分割（UNet、DeepLab）      |
| `logsoftmax` | 数值稳定 + 交叉熵损失 | 分类任务（避免 `log(0)` 问题） |

- **`softmax2d`**：用于 **图像分割**，对每个像素点做 `softmax`，得到每个像素的类别概率。
- **`logsoftmax`**：用于 **分类任务**，避免 `log(softmax(x))` 的数值不稳定问题，提高训练稳定性。

---

## **总结**
- **`softmax`**：通用归一化，适用于 NLP 和分类任务。
- **`softmax2d`**：适用于 **2D 空间**（如语义分割）。
- **`logsoftmax`**：适用于 **交叉熵损失**，提高数值稳定性。

