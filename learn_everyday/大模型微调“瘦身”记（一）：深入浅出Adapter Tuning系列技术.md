## 大模型微调“瘦身”记（一）：深入浅出Adapter Tuning系列技术 

**大家好！**今天，咱们聊点硬核又实用的技术——**Adapter Tuning**及其衍生方法。你可能已经知道，像GPT-4、LLaMA这样的大模型能力超群，但想让它们在你的特定任务上（比如写特定风格的代码、做专业的医疗咨询）表现得炉火纯青，直接“全量微调”可是个“力气活”，不仅需要“矿场”级别的算力，还得准备巨大的存储空间，成本高得吓人。

为了解决这个问题，研究者们提出了**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 的思路，而**Adapter Tuning**就是这个领域里的“开山鼻祖”之一。它就像是给身经百战的“大将军”（预训练模型）配备了一套轻便的“特种装备”（Adapter模块），让将军不用换掉全身铠甲就能适应新战场。

这篇文章，我们就来深入剖析Adapter Tuning以及它的两个重要“升级版”：AdapterFusion和AdapterDrop，并带上公式和代码，让你彻底搞懂它们是怎么工作的！

### 一、 Adapter Tuning：轻装上阵的微调先锋

**1. 背景：全量微调的“不能承受之重”**

想象一下，一个拥有几百亿参数的大模型，就像一本极其厚重的百科全书。全量微调就像是要重新修订整本百科全书来适应一个新的小领域，比如“如何给猫咪剪指甲”。这不仅耗时耗力（训练时间长、GPU需求高），而且每修订一个新领域，你就得复制一本几乎同样厚的“猫咪剪指甲版百科全书”，存储成本爆炸。

**Adapter Tuning**（出自论文 *Parameter-Efficient Transfer Learning for NLP*）应运而生，它的核心思想是：**“冻住”大模型主体，只训练“外挂”的小模块。**

**2. 技术原理：瓶颈结构+残差连接**

Adapter Tuning的做法是在Transformer模型的每个（或部分）层中插入**Adapter模块**。通常，它们被放在多头注意力（Multi-Head Attention, MHA）子层和前馈网络（Feed-Forward Network, FFN）子层之后。

```text
[Transformer Layer]
├── MultiHeadAttention
│   └── Adapter1 (投影后插入)
├── FeedForward  
│   └── Adapter2 (FFN后插入)
└── LayerNorm (可微调)
```

<img src="https://pic3.zhimg.com/v2-c7a60e5065a325b848f48cb8031eb26e_1440w.jpg" alt="img" style="zoom: 67%;" />

<center>*(图1: Adapter模块在Transformer层中的插入位置)*(图2: Adapter模块的内部结构)</center>

这个Adapter模块本身结构很简单，像一个“瓶颈”：

*   **降维（Down-project）**：一个线性层将输入的特征维度 `d` (比如768或1024) 降低到一个小得多的中间维度 `m` (比如64)。
*   **非线性激活（Non-linearity）**：通常使用ReLU或GeLU等激活函数。
*   **升维（Up-project）**：另一个线性层将维度从 `m` 恢复到原来的 `d`。
*   **残差连接（Skip Connection）**：最关键的一步！将Adapter模块的输入 `h` 直接加到其输出上。

设Transformer层输出为$h \in \mathbb{R}^{d}$，Adapter操作可形式化为：
$$
h' = h + W_{up} \cdot \sigma(W_{down} \cdot h)
$$
其中：

- $W_{down} \in \mathbb{R}^{m×d}$为降维矩阵（$m \ll d$，典型值$m=d/4$）
- $\sigma$为ReLU激活函数
- $W_{up} \in \mathbb{R}^{d×m}$为升维矩阵

**公式表示：**

假设Adapter模块的输入是 `h` (维度为 `d`)，`W_down` (维度 `d x m`) 和 `b_down` 是降维层的权重和偏置，`W_up` (维度 `m x d`) 和 `b_up` 是升维层的权重和偏置，`f` 是非线性激活函数。那么Adapter模块的计算过程可以表示为：

```
z = f(h @ W_down + b_down)  # 降维并激活
output = z @ W_up + b_up     # 升维
h_adapter = h + output       # 残差连接
```

这里的 `@` 代表矩阵乘法。

**为什么要有残差连接？** 它非常重要！保证了即使在训练初期，当 `W_down` 和 `W_up` 初始化接近于0时，`output` 也接近于0，此时 `h_adapter ≈ h`。这意味着Adapter模块初始时近似于一个恒等映射，不会破坏预训练模型原有的能力，使得训练更加稳定。

**训练过程：**

1.  加载预训练模型。
2.  **冻结**预训练模型的所有原始参数。
3.  只**更新**新添加的所有Adapter模块的参数，以及相关的Layer Normalization层的参数（论文发现微调LayerNorm也有帮助）。

**代码示例 (PyTorch风格):**

```python
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.non_linear = nn.GELU() # 或者 nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, input_dim)

        # 初始化Up-project权重接近0，保证初始时接近恒等映射
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        down = self.down_project(x)
        activated = self.non_linear(down)
        up = self.up_project(activated)
        output = x + up # 残差连接
        return output

# --- 如何在Transformer层中使用 ---
class TransformerLayerWithAdapter(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, bottleneck_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.adapter1 = Adapter(d_model, bottleneck_dim) # 第一个Adapter

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.adapter2 = Adapter(d_model, bottleneck_dim) # 第二个Adapter

    def forward(self, src):
        # Multi-head Attention part
        attn_output, _ = self.self_attn(src, src, src)
        src = src + attn_output # Residual connection for attention
        src = self.norm1(src)
        src = self.adapter1(src) # Apply first adapter

        # Feed-forward part
        ff_output = self.linear2(self.activation(self.linear1(src)))
        src = src + ff_output # Residual connection for FFN
        src = self.norm2(src)
        src = self.adapter2(src) # Apply second adapter
        return src

```

**结果与讨论：**

实验证明，只用少量新增参数（约占模型总参数的0.5%~5%），Adapter Tuning就能达到接近全量微调的效果（性能差距通常在1%以内）。这极大地降低了训练成本和存储需求。不同的任务和数据集可能需要不同的最佳中间维度 `m`。

![img](https://pic4.zhimg.com/v2-9e0d951f3ef22fc92488d3423e808781_1440w.jpg)
<center>*(图3: Adapter Tuning效果与全量微调对比)*</center>

### 二、 AdapterFusion：融合众长，博采众议

**1. 背景：如何优雅地“缝合”多任务知识？**

如果我们想让模型同时掌握多个任务的知识，比如既会写诗，又会写代码，怎么办？

*   **顺序微调（Sequential Fine-tuning）**：先学写诗，再学写代码。问题是容易“忘了”写诗的技巧（灾难性遗忘），而且学习顺序可能影响结果。
*   **多任务学习（Multi-task Learning）**：同时学习写诗和写代码。问题是任务间可能互相干扰，尤其当任务数据量差异很大时难以平衡。

Adapter Tuning的成功启发了研究者：既然每个Adapter都蕴含了特定任务的知识，能不能设计一个机制，**智能地融合**来自不同任务的Adapter知识呢？

**AdapterFusion**（出自论文 *AdapterFusion: Non-Destructive Task Composition for Transfer Learning*）就是干这个的！

**2. 技术原理：两阶段学习 + 注意力融合**

AdapterFusion采用**两阶段**学习策略：

*   **阶段一：知识提取（Knowledge Extraction）**
    *   为每个源任务（比如任务A、B、C）分别训练独立的Adapter（称为**ST-A**, Single-Task Adapters）。或者也可以用多任务学习的方式联合训练所有Adapter（**MT-A**），但ST-A更常用且效果往往更好。
    *   在这个阶段，预训练模型参数**冻结**，只训练对应任务的Adapter。

*   **阶段二：知识组合（Knowledge Composition）**
    *   针对你的**目标任务**（比如任务D），加载预训练模型和**所有**源任务的Adapter（A、B、C的Adapter）。
    *   **冻结**预训练模型和**所有**源任务Adapter的参数。
    *   引入一个新的、可学习的模块——**AdapterFusion模块**。这个模块学习如何根据目标任务D的需求，动态地组合来自Adapter A、B、C的输出。

<img src="https://pica.zhimg.com/v2-9104b71432f7243fdd2e15677306535c_1440w.jpg" alt="img" style="zoom:67%;" />

<center>*(图4: AdapterFusion的两阶段流程示意)*</center>

**AdapterFusion模块结构：**

它本质上是一个**注意力机制**！在Transformer的每一层（通常放在Adapter插入的位置之后，或者替代原始Adapter的位置），它执行以下操作：

1.  **获取输入**：接收来自Transformer子层（如MHA或FFN后）的输出 `h`。
2.  **计算各Adapter输出**：将 `h` 输入到该层对应的**所有**源任务Adapter（Adapter_A, Adapter_B, Adapter_C...）中，得到各自的输出 `o_A, o_B, o_C...`。注意，这里的Adapter参数是**冻结**的。
3.  **注意力计算**：
    *   **Query (Q)**：通常是该Transformer子层的输出 `h` 经过一个线性变换得到。`Q = h @ W_q`
    *   **Key (K)**：每个Adapter的输出 `o_i` 经过一个线性变换得到。`K_i = o_i @ W_k`
    *   **Value (V)**：通常就是每个Adapter的输出 `o_i` 本身，或者经过一个线性变换。`V_i = o_i @ W_v` (或者 `V_i = o_i`)
    *   计算注意力分数：`scores_i = Q @ K_i^T / sqrt(d_k)`
    *   计算注意力权重：`weights = softmax(scores)`
    *   加权求和：`fused_output = sum(weights_i * V_i)`
4.  **整合输出**：将融合后的输出 `fused_output` 与原始的子层输出 `h` 结合（通常是相加或拼接后处理）。

<img src="https://pic4.zhimg.com/v2-c2a2314600b1d805391395f4bdb335f7_1440w.jpg" alt="img" style="zoom:67%;" />

<center>*(图5: AdapterFusion模块的注意力结构)*</center>

**公式表示 (简化版):**

假设有N个源任务Adapter，其输出分别为 `o_1, o_2, ..., o_N`。令Transformer子层输出为 `h`。

```
# Query, Key, Value 投影 (W_q, W_k, W_v 是可学习参数)
Q = h @ W_q
K_i = o_i @ W_k
V_i = o_i @ W_v

# 注意力计算
scores_i = Q @ K_i.T / sqrt(dimension_of_key)
weights = softmax([scores_1, scores_2, ..., scores_N])

# 加权融合
fused_adapter_output = sum(weights[i] * V_i for i in range(N))

# 最终输出 (例如，加到原始输入上)
h_fused = h + fused_adapter_output
```

**代码示例 (PyTorch风格，概念性):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterFusion(nn.Module):
    def __init__(self, d_model, num_adapters):
        super().__init__()
        # 假设所有Adapter输出维度与d_model一致
        self.num_adapters = num_adapters
        # 可学习的参数 for query, key, value projections
        # 注意：这里简化了，实践中可能对每个adapter的K/V或共享参数有不同设计
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # 或者更复杂的参数化，例如为每个adapter分配不同的key/value权重

    def forward(self, transformer_output, adapter_outputs):
        """
        Args:
            transformer_output (Tensor): Shape (batch_size, seq_len, d_model)
            adapter_outputs (list[Tensor]): List of N tensors, each shape (batch_size, seq_len, d_model)
                                            N = num_adapters
        """
        assert len(adapter_outputs) == self.num_adapters

        query = self.query_proj(transformer_output) # (batch, seq_len, d_model)

        # (batch, seq_len, num_adapters, d_model)
        keys = torch.stack([self.key_proj(adapter_out) for adapter_out in adapter_outputs], dim=2)
        values = torch.stack([self.value_proj(adapter_out) for adapter_out in adapter_outputs], dim=2)

        # Attention scores: (batch, seq_len, num_adapters)
        # Need to reshape query for batch matmul: (batch * seq_len, 1, d_model)
        # Need to reshape keys for batch matmul: (batch * seq_len, num_adapters, d_model) -> transpose
        q_reshaped = query.view(-1, 1, query.size(-1))
        k_reshaped = keys.view(-1, self.num_adapters, keys.size(-1))
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) / (keys.size(-1) ** 0.5)
        attn_scores = attn_scores.view(query.size(0), query.size(1), self.num_adapters) # Reshape back

        attn_weights = F.softmax(attn_scores, dim=-1) # (batch, seq_len, num_adapters)

        # Weighted sum of values
        # attn_weights shape (batch, seq_len, num_adapters, 1)
        # values shape (batch, seq_len, num_adapters, d_model)
        fused_output = (attn_weights.unsqueeze(-1) * values).sum(dim=2) # (batch, seq_len, d_model)

        # 通常将融合后的输出加回原始输入
        final_output = transformer_output + fused_output
        return final_output

```

**结果与讨论：**

AdapterFusion在很多任务上都取得了比单独使用Adapter Tuning或全量微调更好的效果，尤其是在需要综合多种技能的任务上。它成功地在不“破坏”原有知识（模型主体和源Adapter冻结）的情况下，学会了如何“取长补短”。通常，第一阶段使用ST-A（独立训练）+第二阶段AdapterFusion效果最好。

<img src="https://pic4.zhimg.com/v2-cadecd9c428752b45480ea7de79fe7c3_1440w.jpg" alt="img" style="zoom:67%;" />

<center>*(图6: AdapterFusion与其他方法的性能对比)*</center>

然而，它的缺点是在**推理时**需要加载和计算所有相关的源Adapter以及AdapterFusion模块，参数量和计算量会比单个Adapter Tuning要大，可能导致推理变慢。

### 三、 AdapterDrop：为效率而生，动态“瘦身”

**1. 背景：Adapter虽好，推理也需提速**

虽然Adapter Tuning训练快、省显存，但因为在每一层都增加了额外的计算（Adapter模块），推理时相比原始模型或全量微调模型，会慢一点点（论文提到约4%-6%）。当部署多个任务的Adapter时，这个问题更突出。

**AdapterDrop**（出自论文 *AdapterDrop: On the Efficiency of Adapters in Transformers*）就是为了解决这个推理效率问题而提出的。

**2. 技术原理：丢弃不必要的Adapter**

核心思想非常直接：**在推理（甚至训练）时，丢弃（Drop）一部分Adapter模块，尤其是在Transformer的较低层。**

**为什么是较低层？**

研究者发现，Transformer的较低层通常学习更通用的语言特征，而较高层则更侧重于任务相关的、更抽象的特征。因此，较低层的Adapter对于特定任务的适应性可能没那么关键，丢弃它们对性能的影响相对较小，但却能实实在在地减少计算量。

<img src="https://pic2.zhimg.com/v2-314db36574cdc556165340b905cad935_1440w.jpg" alt="img" style="zoom: 67%;" />

<center>*(图7: AdapterDrop示意图，丢弃了部分层的Adapter)*</center>

**实现方式：**

*   **静态丢弃**：在部署前就确定好要丢弃哪些层的Adapter（比如，前5层的Adapter全部移除）。
*   **动态丢弃**：根据输入或其他条件动态决定是否执行某个Adapter（更复杂）。

**代码示例 (概念性修改 TransformerLayerWithAdapter):**

```python
class TransformerLayerWithAdapterDrop(nn.Module):
    # ... (初始化类似 TransformerLayerWithAdapter) ...

    def forward(self, src, drop_adapter1=False, drop_adapter2=False):
        # Multi-head Attention part
        attn_output, _ = self.self_attn(src, src, src)
        src = src + attn_output
        src = self.norm1(src)
        if not drop_adapter1: # 只有在不丢弃时才计算Adapter
            src = self.adapter1(src)

        # Feed-forward part
        ff_output = self.linear2(self.activation(self.linear1(src)))
        src = src + ff_output
        src = self.norm2(src)
        if not drop_adapter2: # 只有在不丢弃时才计算Adapter
            src = self.adapter2(src)
        return src

# 在模型推理时，根据层索引决定 drop_adapter1 和 drop_adapter2 的值
# e.g., if layer_index < 5: drop_adapter1=True; drop_adapter2=True
```

**结果与讨论：**

实验表明，丢弃较低层的Adapter（比如前5层）可以在多任务推理场景下带来显著的速度提升（例如提升39%），而性能损失很小。

<img src="https://pic3.zhimg.com/v2-29d2ca5a17f4f2701fbe9fb074e78d5e_1440w.jpg" alt="img" style="zoom:50%;" />

<center>*(图8: AdapterDrop不同丢弃策略下的速度与性能)*</center>

AdapterDrop的思想同样适用于**AdapterFusion**。可以对参与融合的源Adapter进行剪枝，只保留最重要的少数几个Adapter，也能大幅提升推理速度（比如保留2个Adapter效果接近8个，速度提升68%）。

<img src="https://picx.zhimg.com/v2-d6431b06b2a4be614e0155f2aad438ad_1440w.jpg" alt="img" style="zoom:50%;" />            <img src="https://picx.zhimg.com/v2-bf86f888ceb53604d0d0efddb8435429_1440w.jpg" alt="img" style="zoom: 50%;" />                                                
*(图9: AdapterFusion中的Adapter剪枝)*                                        *(图10: AdapterFusion剪枝后的效果)*

因此，AdapterDrop提供了一种在保持性能的同时优化Adapter推理效率的有效手段，特别是在资源受限的部署环境中。

### 总结

Adapter系列技术是PEFT领域的重要组成部分，它们展示了如何通过巧妙地添加和管理少量参数，高效地将大模型适配到下游任务：

*   **Adapter Tuning**：奠基之作，通过插入小型“瓶颈”模块实现高效微调，核心是**可训练的小模块 + 残差连接**。
*   **AdapterFusion**：解决多任务知识融合问题，核心是**两阶段学习 + 注意力机制融合**冻结的源Adapter。
*   **AdapterDrop**：优化推理效率，核心是**丢弃（尤其是低层）Adapter**以减少计算量。

**面试时，如果你被问到Adapter系列，可以这样准备：**

1.  **Why Adapter?** 清晰说明全量微调的痛点（计算、存储、遗忘），引出PEFT和Adapter的动机。
2.  **Adapter Tuning原理?** 描述瓶颈结构（降维-激活-升维）、残差连接的重要性、训练方式（冻结主体、训练Adapter+LN）。能画出结构图或写出核心公式/伪代码更佳。
3.  **AdapterFusion解决了什么问题？如何解决？** 针对多任务知识融合，通过两阶段学习和注意力机制动态组合源Adapter知识。
4.  **AdapterDrop的作用？** 提升推理效率，通过移除部分（特别是低层）Adapter减少计算开销。
5.  **优缺点/Trade-offs?**
    *   Adapter Tuning: 训练高效，存储少，但有微小推理延迟。
    *   AdapterFusion: 融合效果好，但推理时参数和计算量增加。
    *   AdapterDrop: 提升推理速度，但可能牺牲一点点性能。

掌握了这些，相信你对Adapter系列技术就有了扎实的理解，无论是做项目选型还是应对面试，都能更有底气！

希望这篇详细的博客文章能帮到你！如果还有其他问题，随时可以继续交流探讨！





## 参考文献  

 https://github.com/liguodongiot/llm-action  
 Adapter Tuning原始论文《Parameter-Efficient Transfer Learning for NLP》  
 AdapterFusion论文《AdapterFusion: Non-Destructive Task Composition for Transfer Learning》  
 LoRA论文《LoRA: Low-Rank Adaptation of Large Language Models》