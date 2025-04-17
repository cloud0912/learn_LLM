Okay，各位同学，大家好！我是你们的老朋友小雲。之前我们聊了数据并行（DDP）和流水线并行（Pipeline Parallelism），相信大家对如何把巨大的模型和海量数据“塞”进有限的GPU里，有了一些概念。今天，我们要深入探讨另一种极其重要的并行技术——**张量并行（Tensor Parallelism）**，这可是大模型训练和推理的“必备武器”之一，也是面试时展现你技术深度的大好机会！

咱们系列文章的目标依然是：用大白话讲清楚大模型背后的技术原理和实现，让大家能听懂。

今天的主角是 **张量并行**。

---

## 引言：模型那么大，一块GPU装不下怎么办？

我们知道，现在的大模型参数动辄几百上千亿，别说训练了，就是推理时把所有参数加载到一块GPU的显存里都成了奢望。数据并行（DDP）能解决训练数据量大的问题，流水线并行（Pipeline Parallelism）能把模型的不同“层”（Stages）放到不同GPU上，缓解单个GPU的显存压力。

但是，如果模型的一层就特别大呢？比如一个巨大的 Embedding 层，或者 Transformer 里的某个超宽的 MLP 层，单独这一层就可能塞不进一块GPU。这时候怎么办？

**张量并行** 就闪亮登场了！顾名思义，它不是把模型的“层”切分，而是把“层”内部的**张量（Tensor）**，也就是我们常说的权重矩阵（Weight Matrix），给切分开，放到不同的GPU上去计算。所以，它也经常被称为**层内并行（Intra-Layer Parallelism）**。

![img](https://pic4.zhimg.com/v2-33471e8744527920c14c0eb4070859ad_1440w.jpg)

简单来说，张量并行的核心思想就是：**把矩阵运算分解，分块到不同设备计算，最后再把结果合并起来。** 听起来是不是有点像分布式计算里的MapReduce？没错，思想是相通的！

---

## 张量并行的两种基本“刀法”：行并行与列并行

要切分张量（主要是权重矩阵），最直观的方式就是沿着某个维度“下刀”。主要有两种切法：

1.  **按行切（Row Parallelism）**
2.  **按列切（Column Parallelism）**

<img src="https://pic2.zhimg.com/v2-d4d3eed6197f6b38d171c2f2965f8ffb_1440w.jpg" alt="img" style="zoom: 50%;" />



我们用一个最基础的线性层（Linear Layer）的矩阵乘法（GEMM, General Matrix Multiplication）`Y = XA` 来举例说明。这里：
*   `X` 是输入张量 (shape: `[b, s, h_in]`, b=batch size, s=sequence length, h_in=input hidden size)
*   `A` 是权重矩阵 (shape: `[h_in, h_out]`)
*   `Y` 是输出张量 (shape: `[b, s, h_out]`)

为了简化，我们暂时忽略 batch 和 sequence 维度，只看隐藏维度的计算 `Y = XA`，其中 `X` shape `[k, h_in]`, `A` shape `[h_in, h_out]`, `Y` shape `[k, h_out]`.

<img src="https://pic4.zhimg.com/v2-7e0c19c843b1968b4bc9a70507bbd4b3_1440w.jpg" alt="img" style="zoom:50%;" />

### 1. 行并行（Row Parallelism）

“行并行”切的是**权重矩阵 `A` 的行**（也就是 `h_in` 这个维度）。假设我们把 `A` 沿着行切成两块 `A1` 和 `A2`。
$$
XA = \begin{bmatrix} X_1 & X_2 \end{bmatrix} \begin{bmatrix} A_1 \\ A_2 \end{bmatrix} = X_1 A_1 + X_2 A_2 = Y_1 + Y_2 = Y
$$
其中 `A1` shape `[h_in/2, h_out]`, `A2` shape `[h_in/2, h_out]`.

为了让矩阵乘法能够进行 (`X` 的列数必须等于 `A` 的行数)，我们必须**同时对输入 `X` 按列进行切分**（切 `h_in` 这个维度）。

`X = [ X1 | X2 ]`

其中 `X1` shape `[k, h_in/2]`, `X2` shape `[k, h_in/2]`.

<img src="https://pica.zhimg.com/v2-2f55132d57997daf98e3fe71a9e359bc_1440w.jpg" alt="img" style="zoom:50%;" />

**计算流程：**

1.  **分发/切分输入 `X`**：将 `X` 切分为 `X1` 和 `X2`。`X1` 和 `A1` 放在 GPU 0 上，`X2` 和 `A2` 放在 GPU 1 上。
2.  **本地计算**：
    *   GPU 0 计算 `Y1 = X1 @ A1`。
    *   GPU 1 计算 `Y2 = X2 @ A2`。
3.  **通信（聚合）**：将 GPU 0 上的 `Y1` 和 GPU 1 上的 `Y2` **相加** (`AllReduce` 操作) 得到最终的 `Y`。每个 GPU 都会得到完整的 `Y`。

**关键点：** 行并行在计算后需要进行 `AllReduce` **求和** 操作。

### 2. 列并行（Column Parallelism）

“列并行”切的是**权重矩阵 `A` 的列**（也就是 `h_out` 这个维度）。假设我们把 `A` 沿着列切成两块 `A1` 和 `A2`。
$$
XA = X \begin{bmatrix} A_1 & A_2 \end{bmatrix} = \begin{bmatrix} XA_1 & XA_2 \end{bmatrix} = \begin{bmatrix} Y_1 & Y_2 \end{bmatrix} = Y
$$
其中 `A1` shape `[h_in, h_out/2]`, `A2` shape `[h_in, h_out/2]`.

这次，输入 `X` 不需要切分，因为 `X` 的列数 (`h_in`) 和 `A` 的行数 (`h_in`) 始终匹配。

<img src="https://pic3.zhimg.com/v2-1869c6e9d975f7477e0d198ede06be58_1440w.jpg" alt="img" style="zoom:50%;" />

**计算流程：**

1.  **分发输入 `X`**：将完整的 `X` 复制到所有参与计算的 GPU 上（或者说，每个 GPU 从上一层接收到的输入就是完整的 `X`）。`A1` 放在 GPU 0，`A2` 放在 GPU 1。
2.  **本地计算**：
    *   GPU 0 计算 `Y1 = X @ A1`。
    *   GPU 1 计算 `Y2 = X @ A2`。
3.  **通信（聚合）**：将 GPU 0 上的 `Y1` 和 GPU 1 上的 `Y2` **拼接** (`AllGather` 操作的一种变体，或者理解为每个 GPU 拥有部分结果，需要时再收集) 得到最终的 `Y`。注意：这里得到的 `Y` 也是按列切分的，`Y1` 在 GPU 0，`Y2` 在 GPU 1。

**关键点：** 列并行在计算后，输出 `Y` 是天然分片的。如果下一层需要完整的 `Y`，则需要通信（如 `AllGather`）。但如果下一层是行并行，它正好需要按列切分的输入！

---

## 1维张量并行：Megatron-LM 的经典之作

现在我们知道了行并行和列并行这两种基本操作。那么在大模型，尤其是 Transformer 里，怎么应用呢？

**Megatron-LM** 提出的 1D 张量并行方案是目前最经典、应用最广泛的方法之一。它巧妙地组合了行并行和列并行，应用在 Transformer 的核心组件——MLP 块和 Multi-Head Attention (MHA) 块上。

<img src="https://pic2.zhimg.com/v2-d275f6e69f5e0cc230c3f9a68243c055_1440w.jpg" alt="img" style="zoom:67%;" />

一个标准的 Transformer Block 包含一个 MHA 块和一个 MLP 块。

### MLP 块的张量并行

一个典型的 MLP 块通常包含两个线性层（Linear Layer）和一个非线性激活函数（如 GeLU）。我们表示为 `Y = Dropout(GeLU(X @ A) @ B)`。

<img src="https://pic4.zhimg.com/v2-b47f69ac51b72497922f64231c5fe25f_1440w.jpg" alt="img" style="zoom:50%;" />

<center>f 和 g 代表并行操作原语</center>

Megatron-LM 的策略是：

1.  **第一个线性层 (权重 A)：采用列并行 (Column Parallelism)。**
    *   将权重 `A` 按列（输出维度）切分。
    *   输入 `X` 在所有 GPU 上是相同的（是上一层 MHA 的输出）。
    *   每个 GPU 计算 `Z_i = GeLU(X @ A_i)`。
    *   此时，`Z = GeLU(X @ A)` 这个中间结果在 GPU 之间是按列切分的 (`Z = [Z1 | Z2 | ...]`)。
    *   **前向传播 `f`**：输入 `X` 是同步的（或说完整的），输出 `Z` 是分片的。这个过程**不需要通信**。
    *   **反向传播 `f`**：计算梯度 `dL/dX` 时，需要将各个 GPU 上的梯度 `dL/dX_i` 进行 `AllReduce` **求和**，得到完整的 `dL/dX` 传递给上一层。

2.  **第二个线性层 (权重 B)：采用行并行 (Row Parallelism)。**
    *   将权重 `B` 按行（输入维度）切分。
    *   输入 `Z` 正好是上一步按列切分的结果 (`Z = [Z1 | Z2 | ...]`)，完美匹配行并行的输入要求！每个 GPU 拥有 `Z_i` 和 `B_i`。
    *   每个 GPU 计算 `Y_i = Z_i @ B_i`。
    *   **前向传播 `g`**：将各个 GPU 上的 `Y_i` 进行 `AllReduce` **求和**，得到最终的输出 `Y`。`Y` 在所有 GPU 上是相同的（同步的）。
    *   **反向传播 `g`**：计算梯度 `dL/dZ` 时，输入的 `dL/dY` 在所有 GPU 上是相同的，可以直接计算 `dL/dZ_i = dL/dY @ B_i^T`。这个过程**不需要通信**（指 `dL/dZ` 的计算本身，计算 `dL/dB_i` 当然需要本地的 `Z_i`）。

**总结 MLP 块的通信：**

*   **前向传播：** 在第二个线性层（行并行）之后需要一次 `AllReduce` (Sum)。
*   **反向传播：** 在计算第一个线性层（列并行）的输入梯度 `dL/dX` 时需要一次 `AllReduce` (Sum)。

### Multi-Head Attention (MHA) 块的张量并行

MHA 块稍微复杂点，它涉及到 Query (Q), Key (K), Value (V) 的计算，以及最后输出的线性变换。

<img src="https://pic4.zhimg.com/v2-562a08495dd1daa0360da5020561cd97_1440w.jpg" alt="img" style="zoom:50%;" />

Megatron-LM 的策略：

1.  **Q, K, V 的线性层：采用列并行 (Column Parallelism)。**
    *   将 Q, K, V 的投影权重 `W_Q`, `W_K`, `W_V` 都按列（输出维度，即 hidden_size）切分。通常是按 "头" (Head) 来切分的，保证每个头相关的参数在一个 GPU 上（或者一组头在一个 GPU 上）。
    *   输入 `X` 在所有 GPU 上是相同的。
    *   每个 GPU 计算自己负责的那部分头的 `Q_i`, `K_i`, `V_i`。
    *   Attention 计算 (`softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i`) 可以在每个 GPU 内部独立完成，得到部分 Attention 输出 `AttnOutput_i`。
    *   **前向传播 `f` (Attention部分)**：输入 `X` 同步，输出 `AttnOutput` 是按列（隐藏维度）分片的。**不需要通信**。
    *   **反向传播 `f` (Attention部分)**：计算梯度 `dL/dX` 时，需要将各个 GPU 上的 `dL/dX_i` 进行 `AllReduce` **求和**。

2.  **输出线性层 (权重 B)：采用行并行 (Row Parallelism)。**
    *   将输出投影权重 `W_O` 按行（输入维度，即 hidden_size）切分。
    *   输入 `AttnOutput` 正好是上一步按列切分的结果。
    *   每个 GPU 计算 `Y_i = AttnOutput_i @ W_O_i`。
    *   **前向传播 `g` (输出投影)**：将各个 GPU 上的 `Y_i` 进行 `AllReduce` **求和**，得到最终的 Transformer Block 输出 `Y`。`Y` 在所有 GPU 上是相同的。
    *   **反向传播 `g` (输出投影)**：输入的 `dL/dY` 相同，计算 `dL/dAttnOutput_i` **不需要通信**。

<img src="https://pic2.zhimg.com/v2-7a822c58a72a2d2e54061275af327cd3_1440w.jpg" alt="img" style="zoom:50%;" />

<center>*(图片示意: MHA 块的并行方式)*</center>

**总结 MHA 块的通信：**

*   **前向传播：** 在输出线性层（行并行）之后需要一次 `AllReduce` (Sum)。
*   **反向传播：** 在计算 Q, K, V 线性层（列并行）的输入梯度 `dL/dX` 时需要一次 `AllReduce` (Sum)。

### 完整的 Transformer 层与 Embedding 层

把 MLP 和 MHA 组合起来，一个完整的 Transformer 层的张量并行流程如下：

<img src="https://pic1.zhimg.com/v2-0aea46a731a5c62bf1004d79426bd8e8_1440w.jpg" alt="img" style="zoom:50%;" />

**关键结论：在一个 Transformer 层的前向和反向传播中，总共需要 4 次 `AllReduce` 操作。** 这是 Megatron-LM 1D 张量并行的核心通信开销。

**Embedding 层呢？**

*   输入 Embedding 层 (Word Embedding): 参数矩阵是 `[vocab_size, hidden_size]`。通常 `vocab_size` 非常大。Megatron-LM 选择**按列（`hidden_size` 维度）切分**，这属于**列并行**。输入是 token IDs，输出的 embedding 向量是按 `hidden_size` 切分的。
*   输出 Embedding 层 (LM Head): 参数矩阵是 `[hidden_size, vocab_size]`（通常与输入 Embedding 共享权重，即转置关系）。为了计算 logits，需要完整的 `hidden_size` 输入（来自最后一个 Transformer 层的输出，正好是同步的），然后与按行（`hidden_size` 维度）切分的权重 `W_out^T` 相乘。这属于**行并行**。每个 GPU 计算一部分词汇的 logits。如果需要计算完整的 loss (e.g., CrossEntropy)，需要收集所有 logits 或者 label (通过 `AllGather`)。但在 Megatron-LM 中，交叉熵损失计算本身也可以并行化：每个 rank 计算自己负责的那部分 vocabulary 的 loss，最后 `AllReduce` 求和。

**Megatron-LM 初始化示例:**

```python
# (示意性代码，非完整可运行)
# 假设 args 包含了并行相关的参数
from megatron.core import mpu

# 初始化模型并行组 (包括张量并行和流水线并行)
mpu.initialize_model_parallel(
    tensor_model_parallel_size=args.tensor_model_parallel_size, # 张量并行的 GPU 数量
    pipeline_model_parallel_size=args.pipeline_model_parallel_size, # 流水线并行的 Stage 数量
    # ... 其他参数 ...
)

# 后面创建模型时，Megatron 提供的 Layer (如 RowParallelLinear, ColumnParallelLinear)
# 会自动根据初始化好的并行组进行参数切分和通信。
```

**1D 张量并行的成本分析 (理论):**

假设有 `P` 个 GPU 用于张量并行。

| 方面         | 成本比例         | 说明                                                         |
| :----------- | :--------------- | :----------------------------------------------------------- |
| 计算 (FLOPs) | `O(1/P)`         | 计算量被均摊到 P 个 GPU 上。                                 |
| 内存 (参数)  | `O(1/P)`         | 每个 GPU 只存储 1/P 的参数。                                 |
| 内存 (激活)  | `O(1)`           | **关键瓶颈！** 激活值（尤其是列并行层的输入X）需要在所有 P 个 GPU 上复制。 |
| 通信 (带宽)  | `O(2 * (P-1)/P)` | 每个 AllReduce (Ring 算法) 数据量约为 M*(P-1)/P，共4次，但常数量级是 O(1)。 |
| 通信 (延迟)  | `O(2 * (P-1))`   | 每个 AllReduce (Ring 算法) 需要 2*(P-1) 次点对点通信。       |

**面试要点:**

*   Megatron-LM 的 1D TP 通过交替使用列并行和行并行，使得层间传递的张量要么是完整的 (同步的)，要么是分片的，巧妙地减少了不必要的通信。
*   每个 Transformer 层需要 4 次 `AllReduce`。
*   **主要优点：** 显著降低了模型参数占用的显存。实现相对直观。
*   **主要缺点：**
    1.  **激活值显存没有减少：** 很多中间激活值（如列并行的输入 `X`）需要在所有 TP 并行的 GPU 上复制一份，当 batch size 或 sequence length 很大时，这部分显存开销巨大。
    2.  **通信开销随 P 增大而增大：** AllReduce 操作的通信量和延迟都与并行规模 `P` 相关，当 `P` 很大时，通信成为瓶颈。

---

## 多维张量并行：Colossal-AI 的探索与优化

既然 1D 张量并行有激活显存和通信瓶颈，自然就有研究者思考：能不能把激活也切分了？能不能进一步优化通信？**Colossal-AI** 在这方面做了很多出色的工作，提出了 2D、2.5D、3D 的张量并行方案。

<img src="https://pic2.zhimg.com/v2-d299b940ee7e7a1ddad6cceae357d023_1440w.jpg" alt="img" style="zoom:67%;" />

### 2D 张量并行

1D TP 只在一个维度上切分权重。2D TP 则是在**两个维度**上同时切分权重和输入/输出张量。这通常需要将参与计算的 GPU 排列成一个 2D 网格（`q x q = P` 个 GPU）。

给定 \( $P = q \times q$ \) 个处理器（例如 \( q = 2 \)），将输入矩阵 \($\mathbf{X}$\) 和权重矩阵 \($\mathbf{A}$\) 划分为 \( $q \times q$ \) 的子矩阵块：

$$
\mathbf{X} = \begin{bmatrix} \mathbf{X}_{00} & \mathbf{X}_{01} \\ \mathbf{X}_{10} & \mathbf{X}_{11} \end{bmatrix}, \quad
\mathbf{A} = \begin{bmatrix} \mathbf{A}_{00} & \mathbf{A}_{01} \\ \mathbf{A}_{10} & \mathbf{A}_{11} \end{bmatrix}
$$

**步骤 \( t = 1 \)（第一阶段）**

- **广播操作**：  

  - \($\mathbf{X}_0 = \begin{bmatrix} \mathbf{X}_{00} & \mathbf{X}_{01} \end{bmatrix}$\) 在行方向广播（每个处理器行拥有完整的行数据）。  
  - \($\mathbf{A}_0 = \begin{bmatrix} \mathbf{A}_{00} \\ \mathbf{A}_{01} \end{bmatrix}$\) 在列方向广播（每个处理器列拥有完整的列数据）。  

- **乘法与结果**：  
  $$
  \begin{bmatrix} \mathbf{X}_{00} & \mathbf{X}_{01} \end{bmatrix} \cdot \begin{bmatrix} \mathbf{A}_{00} \\ \mathbf{A}_{01} \end{bmatrix} = \begin{bmatrix} \mathbf{X}_{00}\mathbf{A}_{00} & \mathbf{X}_{00}\mathbf{A}_{01} \\ \mathbf{X}_{10}\mathbf{A}_{00} & \mathbf{X}_{10}\mathbf{A}_{01} \end{bmatrix} \quad (1)
  $$

**步骤 \( t = 2 \)（第二阶段）**

- **广播操作**：  

  - \($\mathbf{X}_1 = \begin{bmatrix} \mathbf{X}_{01} \\ \mathbf{X}_{11} \end{bmatrix}$\) 在行方向广播。  
  - \($\mathbf{A}_1 = \begin{bmatrix} \mathbf{A}_{10} & \mathbf{A}_{11} \end{bmatrix}$\) 在列方向广播。  

- **乘法与结果**：  
  $$
  \begin{bmatrix} \mathbf{X}_{01} \\ \mathbf{X}_{11} \end{bmatrix} \cdot \begin{bmatrix} \mathbf{A}_{10} & \mathbf{A}_{11} \end{bmatrix} = \begin{bmatrix} \mathbf{X}_{01}\mathbf{A}_{10} & \mathbf{X}_{01}\mathbf{A}_{11} \\ \mathbf{X}_{11}\mathbf{A}_{10} & \mathbf{X}_{11}\mathbf{A}_{11} \end{bmatrix} \quad (2)
  $$

**最终结果**

将步骤 \( (1) \) 和 \( (2) \) 的结果相加，得到完整的线性变换结果：
$$
\mathbf{Y} = \mathbf{X}\mathbf{A} = \begin{bmatrix} 
\mathbf{X}_{00}\mathbf{A}_{00} + \mathbf{X}_{01}\mathbf{A}_{10} & \mathbf{X}_{00}\mathbf{A}_{01} + \mathbf{X}_{01}\mathbf{A}_{11} \\ 
\mathbf{X}_{10}\mathbf{A}_{00} + \mathbf{X}_{11}\mathbf{A}_{10} & \mathbf{X}_{10}\mathbf{A}_{01} + \mathbf{X}_{11}\mathbf{A}_{11} 
\end{bmatrix}
$$

---

**核心思想 (基于 SUMMA 算法):**

还是看 `Y = XA`。现在我们有 `P = q x q` 个处理器，排列成 `q x q` 的网格。

1.  **切分:**
    *   输入 `X` 被切分成 `q x q` 块 `X_ij` (按行切 `q` 份，按列切 `q` 份)。
    *   权重 `A` 被切分成 `q x q` 块 `A_ij` (按行切 `q` 份，按列切 `q` 份)。
    *   输出 `Y` 也自然地被切分成 `q x q` 块 `Y_ij`。
    *   处理器 `(i, j)` 负责计算 `Y_ij` 的一部分。

2.  **计算 (以 SUMMA 为例):** 计算 `Y_ij = sum(X_ik @ A_kj)` for `k` from 0 to `q-1`.
    *   这通常涉及多轮计算和通信。
    *   **第 `k` 轮：**
        *   处理器 `(i, k)` 沿着**行**广播它的 `X_ik` 给同一行的所有处理器 `(i, 0...q-1)`.
        *   处理器 `(k, j)` 沿着**列**广播它的 `A_kj` 给同一列的所有处理器 `(0...q-1, j)`.
        *   每个处理器 `(i, j)` 收到 `X_ik` 和 `A_kj`，计算 `X_ik @ A_kj`，并累加到本地的 `Y_ij` 上。
    *   重复 `q` 轮。

<img src="https://picx.zhimg.com/v2-0c4259dcd0053e44ba55336ca0fd2c05_1440w.jpg" alt="img" style="zoom:50%;" />

**成本分析 (理论):**

| 方面         | 成本比例      | 对比 1D (P = q*q)         |
| :----------- | :------------ | :------------------------ |
| 计算 (FLOPs) | `O(1/q^2)`    | 相同                      |
| 内存 (参数)  | `O(1/q^2)`    | 相同                      |
| 内存 (激活)  | `O(1/q^2)`    | **优于 1D (O(1))**        |
| 通信 (带宽)  | `O(6(q−1)/q)` | 可能更高或更低 (取决于 P) |
| 通信 (延迟)  | `O(6(q−1))`   | 可能更优 (取决于实现)     |

**关键优势：** 激活值也被切分了！每个 GPU 上的激活显存从 `O(1)` 降低到`O(1/q^2)`，极大地缓解了显存瓶颈，可以支持更大的 batch size 或 sequence length。

**关键挑战：** 通信模式更复杂，涉及行广播和列广播。总通信量可能比 1D 更多（虽然渐进复杂度可能更好）。

**Colossal-AI 2D TP 示例:**

```python
# (示意性代码)
import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

# Colossal-AI 会自动处理设备网格和并行组的创建
# 配置并行策略
CONFIG = dict(
    parallel=dict(
        data=1,
        pipeline=1,
        tensor=dict(size=4, mode='2d'), # 使用 4 个 GPU 进行 2D 张量并行 (2x2 网格)
    )
)
# ... 初始化 colossalai.launch ...

# 定义模型时使用 Colossal-AI 提供的并行层
import colossalai.nn as col_nn
import torch
class MLP(torch.nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        # Colossal-AI 的 Linear 层会自动根据 2D 并行模式切分权重
        self.dense_1 = col_nn.Linear(dim, intermediate_dim)
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)
        self.dropout = col_nn.Dropout(0.1)

    def forward(self, x):
        # 输入 x 也需要根据 2D 模式进行切分
        # Colossal-AI 提供了工具函数或在内部处理输入切分
        # 假设 x 是切分好的输入
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

# ... 模型创建和训练循环 ...

# 输入数据切分示例 (具体实现可能封装在 Colossal-AI 内部)
# x: [batch_size, hidden_dim]
# rank_2d_row = gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)
# rank_2d_col = gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)
# x_chunk_row = torch.chunk(x, q, dim=0)[rank_2d_row] # 按 batch (或 seq) 切分
# x_partition = torch.chunk(x_chunk_row, q, dim=-1)[rank_2d_col] # 按 hidden_dim 切分
```

### 2.5D 张量并行

2D TP 降低了激活显存，但通信可能增加。能不能在激活显存和通信量之间找到一个更好的平衡点？**2.5D 张量并行** 应运而生。

**核心思想:**
它使用 `P = q * q * d` 个处理器。可以看作是有 `d` "层" (depth) 的 2D 网格。

<img src="https://pic2.zhimg.com/v2-554fc9dfe0168fa826bd0e19c356262f_1440w.jpg" alt="img" style="zoom: 50%;" />

1.  **切分:**
    *   输入 `X` 被切分成 `d * q` 行和 `q` 列。可以想象成 `d` 个 `q x q` 的 `X` 块堆叠起来。
    *   权重 `A` 只切分成 `q x q` 块 (与 2D 类似)。
    *   `d` 个 2D 网格 (每个网格 `q x q` 个处理器) 分别处理 `X` 的一个“层”。

2.  **计算:**
    *   在每个“层” `l` (0 to `d-1`) 内部，使用 2D SUMMA 算法计算 `Y_l = X_l @ A`。这步在 `q x q` 个处理器上并行。
    *   将 `d` 个层计算得到的 `Y_l` 通过 `AllReduce` (Sum) 操作在深度 `d` 这个维度上聚合起来，得到最终的 `Y`。

<img src="https://picx.zhimg.com/v2-c7910fe1a95fd670982c2c921d07796f_1440w.jpg" alt="img" style="zoom:67%;" />

*(图片来源: Colossal-AI 文档 - 2.5D TP 计算示意)*

**成本分析 (理论):**

| 方面         | 成本比例          | 说明                 |
| :----------- | :---------------- | :------------------- |
| 计算 (FLOPs) | `O(1/dq^2)`       | 相同                 |
| 内存 (参数)  | `O(1/q^2)`        | **参数只按 2D 切分** |
| 内存 (激活)  | `O(1/dq^2)`       | 比 2D 更优 (`d` 倍)  |
| 通信 (带宽)  | O(3(q−1)(d+1)/dq) |                      |
| 通信 (延迟)  | O(6(q−1))         |                      |

**关键优势：** 相比 2D，进一步降低了激活显存。通过调整 `d` 的值，可以在通信和内存之间做权衡。当 `d=1` 时退化为 2D，当 `d=q` (即 `P=q^3`) 时接近 3D。

**Colossal-AI 2.5D TP 示例:**

```python
# (示意性代码)
# 配置并行策略
CONFIG = dict(
    parallel=dict(
        data=1,
        pipeline=1,
        tensor=dict(size=8, mode='2.5d', depth=2), # 使用 8 GPU (q*q*d=2*2*2), 深度 d=2
    )
)
# ... 初始化和模型定义 (同 2D) ...

# 输入数据切分示例 (更复杂)
# rank_2p5d_dep = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP) # 深度维度
# rank_2p5d_row = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW) # 2D 行维度
# rank_2p5d_col = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) # 2D 列维度
# x_chunk_dep = torch.chunk(x, d, dim=0)[rank_2p5d_dep] # 按 depth 切分
# x_chunk_row = torch.chunk(x_chunk_dep, q, dim=0)[rank_2p5d_row] # 按 2D row 切分
# x_partition = torch.chunk(x_chunk_row, q, dim=-1)[rank_2p5d_col] # 按 2D col 切分
```

### 3D 张量并行

更进一步，3D 张量并行将计算和存储分布在一个 `q x q x q = P` 的 3D 处理器网格中。

<img src="https://pic4.zhimg.com/v2-29b91bfddc777fa91db740464c21d583_1440w.jpg" alt="img" style="zoom: 50%;" />

朴素的 3D 实现可能导致数据冗余。Colossal-AI 的 3D 实现旨在优化通信。

<img src="https://pic2.zhimg.com/v2-0c40b976954c0e07bfa1e34205990229_1440w.jpg" alt="img" style="zoom:50%;" />

*(图片来源: Colossal-AI 文档 - Colossal-AI 的 3D TP)*

**核心思想:**
将 `X`, `A`, `Y` 都看作是 3D 的张量块，分布在 `q x q x q` 的网格上。处理器 `(i, j, k)` 存储 `X_ijk`, `A_kji`, `Y_ijk`。

![img](https://pic1.zhimg.com/v2-8b63e5e00af40611f4060ad49041060e_1440w.jpg)
*(图片来源: Colossal-AI 文档 - 3D TP 数据分布)*

计算 `Y = XA` 的过程非常复杂，涉及到在 3D 网格的各个维度上进行数据收集 (Gather) 和规约分散 (Reduce-Scatter) 操作。

1.  **前向传播：**
    *   在 `k` 维度收集 `X` (`X_ik`)。
    *   在 `i` 维度收集 `A` (`A_kj`)。
    *   本地计算 `X_ik @ A_kj`。
    *   在 `j` 维度进行 Reduce-Scatter 得到 `Y_ij`。

2.  **反向传播：** 涉及 AllGather 梯度 `dY`, 然后进行类似的 Reduce-Scatter 来计算 `dX` 和 `dA` 的梯度。

**成本分析 (理论):**

| 方面         | 成本比例        | 说明                                    |
| :----------- | :-------------- | :-------------------------------------- |
| 计算 (FLOPs) | `O(1/q^3)`      | 相同                                    |
| 内存 (参数)  | `O(1/q^3)`      | **最优**                                |
| 内存 (激活)  | `O(1/q^3)`      | **最优** (与参数切分维度相关)           |
| 通信 (带宽)  | `O(6(q−1)/q^3)` | **理论上最优**，通信量随 P 增大下降最快 |
| 通信 (延迟)  | `O(6(q−1))`     |                                         |

**关键优势：** 理论上具有最低的通信带宽需求，并且参数和激活的内存分布也达到了最优的级别。

**关键挑战：** 实现极为复杂，通信模式难以高效实现，对网络拓扑和通信库要求很高。实际性能可能受延迟和实现效率影响。

<img src="https://pica.zhimg.com/v2-89f00db96384986d1e87f75c1e93cb7a_1440w.jpg" alt="img" style="zoom:50%;" />

**Colossal-AI 3D TP 示例:**

```python
# (示意性代码)
# 配置并行策略
CONFIG = dict(
    parallel=dict(
        data=1,
        pipeline=1,
        tensor=dict(size=8, mode='3d'), # 使用 8 GPU 进行 3D 张量并行 (2x2x2 网格)
    )
)
# ... 初始化和模型定义 (同 2D) ...

# 输入数据切分示例 (极其复杂)
# 需要根据 3D 网格的三个维度 (input, output, weight) 来切分输入 x
# rank_3d_input = gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
# rank_3d_output = gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)
# rank_3d_weight = gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
# x_chunk_input = torch.chunk(x, q, dim=0)[rank_3d_input]
# x_chunk_output = torch.chunk(x_chunk_input, q, dim=-1)[rank_3d_output]
# x_partition = x_chunk_output # Weight 维度不直接切分输入x? (需要查阅具体实现)
```

**面试要点 (多维 TP):**

*   **动机:** 解决 1D TP 的激活显存和通信瓶颈。
*   **2D TP:** 同时切分权重和激活，显著降低激活显存 ，但通信模式更复杂 (行/列广播)。
*   **2.5D TP:** 在 2D 基础上增加深度 `d`，进一步降低激活显存，提供内存和通信的权衡。
*   **3D TP:** 理论上最优的内存分布和通信带宽 ，但实现极其复杂。
*   **权衡:** 没有绝对最好的方法，需要在显存、通信带宽、通信延迟、实现复杂度之间根据具体硬件和模型进行选择。Colossal-AI 提供了这些选项，让用户可以根据需求选择。

---

## PyTorch 原生张量并行支持：走向通用与统一

Megatron-LM 和 Colossal-AI 的张量并行方案非常强大，但它们通常与各自的框架或特定的模型架构（主要是 Transformer）深度绑定。对于通用的深度学习框架 PyTorch 来说，需要一个更底层、更通用的抽象来支持张量并行，并能与其他并行策略（如 DDP, FSDP, 流水线并行）更好地结合。

PyTorch 2.0 之后引入了 **`torch.distributed.tensor` (DTensor)** 这个关键抽象。

**DTensor 的目标:**

*   **统一表示:** 提供一种统一的方式来表示分布在多个设备上的张量，无论它是如何被切分（sharded）或复制（replicated）的。
*   **互操作性:** 让不同的并行策略（TP, DP, PP, FSDP）能够更好地协同工作，比如在一个 FSDP 包裹的模块内部使用张量并行。
*   **易用性:** 提供更高层次的 API (`parallelize_module`) 来自动地对模型进行张量并行化，用户只需指定并行策略和设备网格。
*   **编译器基础:** 作为未来基于编译器的分布式训练优化的基础。

**核心概念:**

1.  **`DeviceMesh`:** 定义一组参与计算的设备（GPU）以及它们的逻辑拓扑结构（例如，一个 1D 或 2D 的网格）。
2.  **`Placement`:** 描述张量在 `DeviceMesh` 上是如何分布的。常见的 `Placement` 有：
    *   `Shard(dim)`: 张量沿着 `dim` 维度被切分到 `DeviceMesh` 的对应维度上。
    *   `Replicate()`: 张量在 `DeviceMesh` 的对应维度上被复制。
    *   `Partial()`: 张量在 `DeviceMesh` 上是部分聚合状态（例如，`AllReduce` Sum 的中间状态）。
3.  **`DTensor`:** 一个包装了普通 `torch.Tensor` 的对象，它额外包含了 `DeviceMesh` 和一个 `Placement` 列表（描述每个 `DeviceMesh` 维度的分布方式）。PyTorch 的分布式算子会自动处理 `DTensor` 的计算和通信。

**PyTorch TP 示例:**

```python
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Shard, Replicate
from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel, ColwiseParallel, RowwiseParallel
from torch.nn.parallel import DistributedDataParallel as DDP # TP可以和DDP结合

# 假设已经初始化了分布式环境 (dist.init_process_group)
# rank = dist.get_rank()
# world_size = dist.get_world_size()

# 1. 创建设备网格 (假设使用所有可用的 GPU 进行 TP)
device_mesh = DeviceMesh("cuda", torch.arange(0, world_size))

# 2. 定义你的普通 PyTorch 模型
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20) # 将被 ColwiseParallel
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20, 10) # 将被 RowwiseParallel

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

model = ToyModel().cuda(rank)

# 3. 使用 parallelize_module 自动应用张量并行
# PairwiseParallel 模拟 Megatron-LM 的 Column -> Row 模式
# ColwiseParallel 只应用列并行
# RowwiseParallel 只应用行并行
parallel_style = PairwiseParallel() # 或者 ColwiseParallel(), RowwiseParallel()
model = parallelize_module(model, device_mesh, parallel_style)

# (可选) 可以继续用 DDP 包裹，实现 TP + DP
# model = DDP(model, device_ids=[rank])

# 4. 正常训练
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(5):
    # 注意：TP 的输入通常需要在所有 rank 上保持一致
    torch.manual_seed(0) # 保证输入一致
    inp = torch.rand(16, 10).cuda(rank) # 输入是 Replicated

    # DTensor 会自动处理内部的切分和通信 (AllReduce/AllGather)
    output = model(inp) # 输出也是 Replicated (因为 PairwiseParallel 最后是 RowParallel)

    # 损失计算和反向传播
    output.sum().backward() # 梯度也是 DTensor
    optimizer.step()
    optimizer.zero_grad()

    if rank == 0:
        print("Iteration done.")

```

**优势：**

*   **通用性：** 不局限于特定模型架构。
*   **组合性：** 设计上更容易与其他并行策略（DDP/FSDP/PP）结合。
*   **框架集成：** 作为 PyTorch 的一部分，享受更好的生态支持和未来优化。

**挑战：**

*   **相对较新：** DTensor 仍在快速发展中，API 和功能可能还在演进。
*   **性能：** 作为一个通用抽象，其性能可能需要持续优化才能媲美高度定制化的方案（如 Megatron-LM 对 Transformer 的优化）。

**面试要点:**

*   知道 PyTorch 提供了原生的 TP 支持 (`DTensor`, `parallelize_module`)。
*   理解 `DeviceMesh`, `Placement` (Shard, Replicate) 的基本概念。
*   了解其目标是提供更通用、组合性更好的分布式训练能力。

---

## 结语：张量并行，大模型训练的“精密切割师”

今天我们深入探讨了张量并行（Tensor Parallelism）的原理与实践。总结一下关键点：

1.  **核心思想：** 将层内的权重张量（有时也包括激活）切分到不同 GPU 上，进行并行计算，以降低单个 GPU 的显存压力和可能的计算时间。
2.  **基本操作：** 行并行（Row Parallelism）和列并行（Column Parallelism）是构建更复杂 TP 策略的基础。
3.  **Megatron-LM (1D TP):** 通过交替使用列并行和行并行，巧妙地应用于 Transformer 的 MLP 和 MHA 块。优点是实现相对简单，有效降低参数显存；缺点是激活显存未减少，通信开销随 GPU 数量增加而变大。
4.  **Colossal-AI (多维 TP):**
    *   **2D TP:** 同时切分权重和激活，显著缓解激活显存瓶颈。
    *   **2.5D TP:** 在 2D 基础上引入深度，提供激活显存和通信之间的权衡。
    *   **3D TP:** 理论上最优的内存和通信扩展性，但实现复杂。
    *   多维 TP 的选择需要在显存、通信、实现复杂度间权衡。
5.  **PyTorch DTensor:** 提供了一个通用的、可组合的框架级 TP 解决方案，是 PyTorch 分布式训练的重要发展方向。



分布式训练是一个庞大而精深的领域，我们今天只是触及了其中的一部分。如果你对某个细节特别感兴趣，或者有任何疑问，欢迎在评论区留言交流！我们下次再见！



参考：

https://zhuanlan.zhihu.com/p/657921100

[[源码解析\] 模型并行分布式训练Megatron (1) --- 论文 & 基础 - 罗西的思考 - 博客园](https://www.cnblogs.com/rossiXYZ/p/15840803.html)