# 大模型微调“瘦身”记（四）：深入解析LoRA系列系列——LoRA、AdaLoRA与QLoRA

Okay, 大家好！我是你们的老朋友小雲。今天，咱们来聊聊PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）家族里的几位“明星”成员：**LoRA**、**AdaLoRA** 和 **QLoRA**。

大模型虽好，但“胃口”太大，全量微调（Fine-tuning）动辄需要几百G显存，这让很多同学望而却步。PEFT技术的出现，就是为了解决这个问题，让我们用更少的资源撬动大模型的强大能力。这几项技术不仅原理巧妙，也是面试中的高频考点。弄懂它们，能让你在面试时更有底气！

---

## LoRA：给模型“打补丁”的艺术

### 背景：大模型的“低秩”之谜

想象一下，我们的大语言模型就像一个极其复杂的神经网络，里面有很多进行矩阵乘法的全连接层。这些层的权重矩阵通常是“满秩”的，意味着它们包含的信息非常丰富，能够处理各种通用任务。

然而，当我们将这个通用大模型针对某个 *特定* 任务（比如做特定领域的客服、写特定风格的代码）进行微调时，研究者发现，参数的 *改动量*（也就是微调前后权重的差值 `ΔW`）其实并不需要那么复杂。这个 `ΔW` 矩阵具有“**低秩（Low Rank）**”特性，可以理解为，为了适配这个特定任务，需要调整的信息维度并不高，可以用一个更简单的、秩更低的矩阵来近似表示。

LoRA（**LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS**）正是基于这个洞察提出的。

### 技术原理：LoRA是如何工作的？

LoRA 的核心思想是：**我不直接修改你原来的大矩阵 W，我在旁边加一个“旁路”，用两个小的、低秩的矩阵 A 和 B 来学习这个改动量 ΔW。**

具体来说，对于模型中某个需要进行矩阵乘法的层（例如 Attention 层的 Wq, Wk, Wv, Wo 或 MLP 层的权重），原始的计算是 `h = Wx`。LoRA 在旁边增加了一个新的计算路径：`h = Wx + BAx`。

这里：

1.  `W`：原始预训练模型的权重矩阵（维度 `d x k`），在 LoRA 训练中 **保持冻结，不更新**。
2.  `x`：输入向量。
3.  `A`：一个降维矩阵（维度 `r x k`），将输入 `x` 从 `k` 维投影到低秩 `r` 维。`r` 就是我们设定的“秩”，远小于 `d` 和 `k` (`r << min(d, k)`）。
4.  `B`：一个升维矩阵（维度 `d x r`），将 `r` 维的表示投影回原来的 `d` 维。
5.  `BA`：这两个小矩阵的乘积（维度 `d x k`）就近似了我们想要学习的改动量 `ΔW`。

<img src="https://pic3.zhimg.com/v2-1ee7ae98a860e5f9aff51d5b1c833296_1440w.jpg" alt="LoRA 结构图" style="zoom:50%;" />

<center>*图片来源：LoRA 论文*</center>

**公式表示**：
模型的输出 `h` 可以表示为：
`h = W₀x + ΔWx`
其中，`W₀` 是预训练模型的原始权重（冻结），`ΔW` 是任务相关的参数改变量。
LoRA用低秩分解 `BA` 来近似 `ΔW`，即 `ΔW ≈ BA`。
所以，最终的计算是：
`h = W₀x + α * (BA)x`

这里引入了一个**缩放因子 `α`** (通常是 `alpha/r`)，用来调整旁路 `BA` 对最终结果的影响力。这个 `α` 是一个超参数。

**训练过程**：

1.  **冻结 W₀**：原始模型的参数完全不动。
2.  **初始化**：矩阵 A 通常用高斯分布初始化，矩阵 B 初始化为全零。这很关键！因为 B=0，所以在训练刚开始时，`BA=0`，LoRA旁路不起作用，模型输出 `h = W₀x`，保持了预训练模型的初始状态，让训练更稳定。
3.  **只训练 A 和 B**：在微调过程中，只有矩阵 A 和 B 的参数会被更新。

**参数量对比**：

*   原始 W 的参数量是 `d * k`。
*   LoRA旁路 A 和 B 的参数量是 `d*r + r*k = r * (d + k)`。
*   由于 `r << min(d, k)`，所以 `r * (d + k)` 远小于 `d * k`。例如，如果 d=k=4096, r=8，参数量可以减少到原来的 `8 * (4096 + 4096) / (4096 * 4096) ≈ 0.004`，即千分之四左右！

**推理过程**：

推理时，为了不增加计算量（即避免额外的矩阵乘法 `BAx`），我们可以将训练好的 `BA` 和原始的 `W₀` **合并**：
`W' = W₀ + α * BA`
然后用新的权重 `W'` 进行正常的推理计算 `h = W'x`。这样，推理速度和原始模型完全一样，没有任何额外的延迟！这是LoRA相比于Adapter等方法的一个巨大优势。

**应用范围与秩的选择**：

*   LoRA通常应用于Transformer模型中的**Attention层**的权重矩阵，如 `Wq`, `Wk`, `Wv`, `Wo`。论文实验表明，同时调整 `Wq` 和 `Wv` 效果较好。
*   **秩 `r` 的选择**：`r` 不需要很大，通常 `4`, `8`, `16` 就能取得不错的效果。论文发现，增加 `r` 带来的性能提升很快会饱和，选择合适的 `r` 比盲目增大 `r` 更重要。

<img src="https://picx.zhimg.com/v2-d5b64ac7875887d3143939169db7e467_1440w.jpg" alt="LoRA 秩 r 的影响" style="zoom:50%;" />

<center>*图片来源：LoRA 论文*</center>

**效果**：

LoRA在很多任务上，用极少的训练参数（通常不到总参数的0.1%）就能达到甚至超过全量微调的效果。

<img src="https://pica.zhimg.com/v2-9d63eb91cc84736b230d04f673f982ea_1440w.jpg" alt="LoRA 效果对比" style="zoom:50%;" />

*<center>图片来源：LoRA 论文</center>*

**LoRA代码示例 (自己实现和使用`peft`库)**：        

```python
class LoRALayer(nn.Module):
    def __init__(self, d_model, r=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_model, r))
        self.B = nn.Parameter(torch.zeros(r, d_model))
        
    def forward(self, x):
        return x @ self.A @ self.B  # 低秩更新项

# 在Transformer中注入LoRA
class LoRAAttention(nn.Module):
    def __init__(self, orig_attn, r=8):
        super().__init__()
        self.orig_attn = orig_attn
        self.lora = LoRALayer(orig_attn.in_features, r)
        
    def forward(self, x):
        orig_out = self.orig_attn(x)
        lora_out = self.lora(x)
        return orig_out + lora_out
```



```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载预训练模型
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,                             # LoRA 的秩 r
    lora_alpha=16,                   # LoRA 的 alpha 参数 (相当于上面公式里的 α*r)
    target_modules=["q_proj", "v_proj"], # 指定要应用 LoRA 的模块 (如 Attention 的 Q 和 V 线性层)
    lora_dropout=0.05,               # LoRA 层的 dropout 概率
    bias="none",                     # 是否训练 bias 参数 ('none', 'all', 'lora_only')
    task_type=TaskType.CAUSAL_LM     # 任务类型
)

# 3. 应用 LoRA 到模型
lora_model = get_peft_model(model, lora_config)

# 打印可训练参数量
lora_model.print_trainable_parameters()
# 输出类似: trainable params: 458,752 || all params: 331,669,760 || trainable%: 0.1383

# 接下来就可以用 lora_model 进行正常的训练了
# ... training code ...

# 推理时可以合并权重 (可选)
# merged_model = lora_model.merge_and_unload()
# merged_model.save_pretrained("opt-350m-lora-merged")
```

**LoRA小结**：

*   **优点**：参数效率极高、训练快、推理无额外延迟、易于部署（切换任务只需替换很小的A/B矩阵）。
*   **缺点**：秩 `r` 需要预先设定，且所有层使用相同的 `r`，可能不是最优的参数分配方式。

---

## AdaLoRA：让LoRA更智能的参数分配

### 背景：LoRA的局限与改进动机

LoRA很棒，但它有一个“一刀切”的问题：**给所有应用LoRA的层都分配了相同的秩 `r`**。

然而，在微调过程中，不同层、不同模块（比如Attention的Q和V矩阵）的重要性是不同的。有些地方可能需要更精细的调整（需要更大的 `r`），而有些地方稍微调整一下就够了（可以用更小的 `r`）。固定 `r` 可能会导致：

*   在不重要的层浪费参数预算。
*   在重要的层参数预算不足，限制了模型性能。

此外，LoRA主要关注Attention层，但FFN（Feed-Forward Network）层的参数量其实更大，对模型能力影响也很大。

**AdaLoRA (Adaptive LoRA) 的目标**：根据不同权重矩阵在微调过程中的**重要性**，**动态地、自适应地**分配参数预算（也就是秩 `r`）。

### 技术原理：AdaLoRA如何实现自适应？

AdaLoRA的核心思想是：**重要的地方多给点参数，不重要的地方少给点参数，甚至不给参数（裁剪掉）**。

它是如何做的呢？

1.  **参数化方式改变**：AdaLoRA不再直接用 `BA` 来表示 `ΔW`，而是采用了类似**奇异值分解（SVD）**的形式来参数化增量更新 `ΔW`。
    `ΔW ≈ PΛQᵀ`
    *   `P` 和 `Q` 是两个正交（或近似正交）的矩阵，包含了主要的变换方向（类似SVD中的左右奇异向量）。
    *   `Λ` 是一个对角矩阵（或只保留对角线元素的向量），包含了每个方向上的重要性大小（类似SVD中的奇异值）。对角线元素的数量，就对应了这一层的**有效秩**。

    <img src="https://pic2.zhimg.com/v2-d04c618ee963a38a249389793d1aa029_1440w.jpg" alt="AdaLoRA SVD 参数化" style="zoom: 25%;" />
    
2.  **重要性评分与动态预算分配**：
    
    *   AdaLoRA在训练过程中会**动态评估**每个奇异值（`Λ` 中的元素）的重要性。通常基于梯度信息或者参数值的大小来计算重要性得分。
    *   根据这些得分，AdaLoRA会**重新分配参数预算**。它会提高重要奇异值对应方向上的预算（保留或增加其对应的P和Q的部分），并**裁剪掉**那些不重要的奇异值及其对应的P和Q部分。
    *   这个分配和裁剪的过程是在训练中**周期性**进行的，而不是一次性设定。
    
3.  **预算约束**：整个模型的可训练参数总量（所有层的 `P`, `Λ`, `Q` 加起来）需要满足一个预设的**总预算**（比如，总参数量不超过原始模型的0.5%）。AdaLoRA的目标是在这个总预算内，通过智能分配，最大化模型性能。

4.  **正则化**：由于精确的SVD计算成本很高，AdaLoRA通过在损失函数中添加**惩罚项**来鼓励 `P` 和 `Q` 保持近似正交，从而避免了显式的SVD分解，并稳定训练过程。

**AdaLoRA过程（简化理解）**：

*   **初始化**：给每个可能应用LoRA的层分配一个初始的、较大的最大秩 `r_max`，并用 `PΛQᵀ` 初始化。
*   **训练与评估**：正常训练模型，同时周期性地计算每个奇异值（`Λ` 中元素）的重要性得分。
*   **裁剪与分配**：根据得分和总预算限制，保留最重要的奇异值，裁剪掉不重要的。这意味着某些层的有效秩会降低，甚至降为0（完全不更新），而某些重要层的有效秩会保持较高水平。参数预算在层与层之间流动。
*   **重复**：持续这个“训练-评估-裁剪/分配”的循环。

**公式表达**：

AdaLoRA通过动态分配秩\(r\)实现参数效率最大化：

​	1.**参数重要性评估**：
$$
\Delta W_i = \sum_{k=1}^{r_i} \sigma_k \cdot u_k v_k^T \quad \text{其中} \quad r_i \propto \text{重要性评分}
其中：
$$

- $(\sigma_k)$：奇异值（按重要性排序）
- $(u_k, v_k)$：左/右奇异向量

2. **正交性约束**：

$$
\mathcal{L}_{orth} = λ(||U^TU-I||_F + ||V^TV-I||_F)
$$

**效果**：

AdaLoRA通过这种自适应的预算分配，能够在相同的参数预算下，通常取得比LoRA更好的性能，尤其是在参数预算非常有限的情况下。

<img src="https://pic4.zhimg.com/v2-b15d686c210f00891a57c0dd7faa5525_1440w.jpg" alt="AdaLoRA 效果对比" style="zoom:50%;" />

**AdaLoRA小结**：

*   **优点**：参数分配更智能，相同预算下性能通常优于LoRA，能自动识别并侧重重要参数。
*   **缺点**：实现相对LoRA更复杂，训练过程中需要额外的计算（重要性评估和裁剪）。

---

## QLoRA：让超大模型也能“平民化”微调

### 背景：当模型大到LoRA也吃力时

LoRA 和 AdaLoRA 极大地降低了 *可训练参数* 的数量，但它们并没有减少 *基础模型本身* 占用的内存。对于像 LLaMA 65B 这样拥有数百亿参数的巨型模型，即使只加载模型本身（不做任何训练），也需要几百 GB 的显存（例如，65B 参数 * 2 bytes/参数 (BF16) ≈ 130GB，加上激活值、梯度等可能远超此数）。这使得在单张消费级或专业级 GPU（如 24GB, 48GB, 80GB）上进行微调几乎不可能。

**量化（Quantization）** 技术可以将模型的权重从常用的 16 位浮点数（如 FP16, BF16）或 32 位浮点数（FP32）压缩到更低的位数（如 8 位整数 INT8，甚至 4 位）。这能显著减小模型体积，降低内存占用，加速推理。但传统的量化方法通常用在 *推理* 阶段，如果在量化后的模型上进行 *微调*，往往会导致严重的性能下降。

**QLoRA（QLORA: Efficient Finetuning of Quantized LLMs）** 横空出世，它开创性地提出了一种能在 **4-bit 量化模型** 上进行 **高效且几乎无损性能** 微调的方法。

### 技术原理：QLoRA的“三板斧”

QLoRA的核心思想是：**用更低的数据精度（如4-bit）来存储预训练模型的主体参数，同时只对LoRA部分的参数（A和B）使用较高精度（如16-bit BFloat16）进行训练和更新**。

它引入了几个关键技术：

1.  **4-bit NormalFloat (NF4) 量化**：
    *   这是一种专门为**正态分布**（通常模型权重近似服从正态分布）设计的数据类型。
    *   相比传统的FP4（浮点4-bit）或Int4（整数4-bit），NF4能在4-bit的精度下更好地保留原始权重的信息，理论上信息损失最小。
    *   **如何工作**：它通过一种称为“分位数量化”（Quantile Quantization）的技术，确保每个量化区间包含相同数量的原始权重值。这使得它对异常值不那么敏感，能更均匀地表示权重分布。
    *   **关键点**：模型的主体权重 `W₀` 以 **NF4** 格式存储在显存中，大大减少了存储空间。
2.  **双重量化 (Double Quantization, DQ)**：
    *   量化过程本身会产生一些“量化常数”（比如缩放因子、零点等），这些常数虽然数量不多，但累积起来也占用一定显存。
    *   DQ就是对这些量化常数**再进行一次量化**（比如用8-bit量化32-bit的常数），进一步节省显存。虽然节省量不大（大约每个参数节省0.5 bit），但对于大模型来说也算锦上添花。
3.  **分页优化器 (Paged Optimizers)**：
    *   优化器状态（如Adam的动量和方差）通常需要和模型参数一样多的显存（甚至两倍，因为需要存储32-bit的参数）。
    *   当显存不足以容纳所有优化器状态时（尤其是在梯度累积或梯度检查点技术下可能出现峰值），分页优化器利用NVIDIA的**统一内存（Unified Memory）**特性，自动将**暂时不用**的优化器状态从**GPU显存“分页”到CPU内存（RAM）**中，等需要时再调回GPU。
    *   这就像操作系统在内存不足时使用硬盘作为虚拟内存一样，避免了因为显存瞬间 OOM（Out-Of-Memory）导致训练崩溃。
4.  **计算与存储的分离**：
    *   **存储**：基础模型的权重以 4-bit NF4 格式存储。
    *   **计算**：在进行前向和反向传播计算时，**需要将 4-bit 的权重动态地反量化（Dequantize）回 BFloat16 (BF16)** 格式。所有的矩阵乘法等运算都在 BF16 下进行。
    *   **LoRA 适配器**：添加的 LoRA 矩阵 A 和 B 本身 **保持在 BF16 格式** 进行训练和存储。
    *   **梯度计算**：梯度通过 LoRA 适配器反向传播，并流经 *反量化后的 BF16 权重*。注意，**基础模型的 4-bit 权重本身不直接更新**，更新的是 LoRA 适配器。

<img src="https://pic4.zhimg.com/v2-8f5ee665d6de71414097733f80ec7c11_1440w.jpg" alt="QLoRA 架构示意图" style="zoom: 50%;" />

**简单理解：** QLoRA 就像给大模型穿上了一件极度压缩的“紧身衣”（4-bit NF4 量化），让它能挤进小小的 GPU 显存里。然后，只给它加上轻便灵活的“外骨骼”（LoRA 适配器）进行训练。计算时，需要暂时“解开”紧身衣的一部分（反量化到 BF16）来完成动作（矩阵计算），但计算完又马上恢复压缩状态。同时，还配备了一个“内存调度系统”（分页优化器）来处理临时性的内存紧张。

**QLoRA的训练流程**：

1.  **加载与量化**：加载预训练模型 `W₀`，并将其参数量化为 **NF4** 格式存储。
2.  **添加LoRA**：在量化后的 `W₀` 之上添加LoRA模块（矩阵 A 和 B）。LoRA模块的参数通常保持较高的精度，如 **BFloat16 (BF16)**。
3.  **前向传播**：
    *   当计算 `h = W₀x + α * (BA)x` 时：
    *   对于 `W₀x`：需要将 NF4 格式的 `W₀` **实时地、按需地反量化（Dequantize）**回 **计算精度**（通常也是 BF16），然后执行矩阵乘法。计算完成后，这个 BF16 版本的 `W₀` 就可以丢弃了，显存中始终保存的是 NF4 版本。
    *   对于 `BAx`：矩阵 A 和 B 本身就是 BF16 精度，直接进行计算。
4.  **反向传播**：计算梯度。由于 `W₀` 是冻结的，我们只需要计算 LoRA 参数 A 和 B 的梯度。梯度流过反量化操作，传到 A 和 B。
5.  **参数更新**：使用**分页优化器**更新 BF16 精度的 LoRA 参数 A 和 B。如果此时显存不足，优化器状态会被自动换出到CPU内存。

**效果：**

*   **显著的内存节省**：QLoRA 使得在单张 48GB GPU 上微调 65B 参数模型成为可能，而在单张 24GB GPU 上也能微调 33B 模型。这在以前是不可想象的。
*   **性能保持**：最令人惊讶的是，通过 QLoRA 微调（使用 4-bit NF4 量化 + LoRA），其性能几乎与使用 BF16 进行全参数微调或 BF16 LoRA 微调相当！这表明 4-bit NF4 量化配合 LoRA 微调可以有效地恢复量化带来的精度损失。

<img src="https://picx.zhimg.com/v2-54882cd01a46d53fb9e26869091beda5_1440w.jpg" alt="QLoRA 效果与不同数据类型对比" style="zoom:50%;" />

<center>*图片来源：QLoRA 论文*</center>

**QLoRA代码示例 (使用`transformers`和`bitsandbytes`库)**：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-hf" # 示例模型

# 1. 定义 QLoRA 配置 (通过 BitsAndBytesConfig)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # 启用4-bit量化加载
    bnb_4bit_quant_type="nf4",           # 指定量化类型为 NF4
    bnb_4bit_compute_dtype=torch.bfloat16, # 指定计算精度为 bfloat16
    bnb_4bit_use_double_quant=True,      # 启用双重量化
)

# 2. 加载量化后的模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # 自动将模型分片到可用设备 (GPU/CPU)
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 禁用缓存 (在使用 gradient checkpointing 时建议)
model.config.use_cache = False

# 准备模型进行 k-bit 训练 (启用 gradient checkpointing, 转换 layer norms 和 head 为 float32)
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# 3. 定义 LoRA 配置 (与普通 LoRA 类似)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 应用 LoRA 到更多层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. 应用 LoRA 到量化后的模型
qlora_model = get_peft_model(model, lora_config)

qlora_model.print_trainable_parameters()
# 输出 trainable params 百分比会更低，因为总参数量很大

# 接下来使用 qlora_model 进行训练
# 需要配合支持 Paged Optimizer 的训练器 (如 transformers.Trainer 自动支持)
# ... training code using Trainer ...

# QLoRA 推理时通常不需要合并权重，直接使用 qlora_model 推理即可
# 因为 W₀ 本身就是量化的，合并需要特殊处理或保持分离
```

**QLoRA小结**：

*   **优点**：极大地降低了**基础模型**的显存占用，使得在消费级硬件上微调超大模型成为可能；性能损失极小；训练稳定。
*   **缺点**：依赖特定的量化库（`bitsandbytes`）；反量化会带来一定的计算开销（虽然通常被显存节省的好处所抵消）。

---

## 总结与面试锦囊

| 特性            | LoRA (Low-Rank Adaptation)        | AdaLoRA (Adaptive LoRA)             | QLoRA (Quantized LoRA)                                  |
| :-------------- | :-------------------------------- | :---------------------------------- | :------------------------------------------------------ |
| **核心思想**    | 用低秩矩阵 BA 近似 ΔW             | 动态分配秩预算给重要的 ΔW (SVD形式) | 量化基础模型 W₀ + 标准 LoRA (BA)                        |
| **解决问题**    | 全量微调参数量大、计算/存储成本高 | LoRA 固定秩，参数分配不够优化       | 超大模型基础权重加载显存爆炸，无法微调                  |
| **基础模型 W₀** | 冻结，高精度 (FP16/BF16)          | 冻结，高精度 (FP16/BF16)            | 冻结，**低精度量化 (NF4)**                              |
| **旁路 ΔW**     | BA，低秩 `r` 固定                 | PΛQᵀ，有效秩 `r` **自适应**         | BA，低秩 `r` 固定 (同 LoRA)                             |
| **旁路精度**    | 高精度 (FP16/BF16)                | 高精度 (FP16/BF16)                  | 高精度 (BF16)                                           |
| **计算精度**    | 高精度 (FP16/BF16)                | 高精度 (FP16/BF16)                  | 中等精度 (**BF16**) (W₀需反量化)                        |
| **关键技术**    | 低秩分解                          | SVD参数化, 重要性评分, 动态裁剪     | **NF4量化**, **双重量化**, **分页优化器**               |
| **推理延迟**    | **无** (可合并权重)               | 无 (理论上可合并，但较复杂)         | **轻微增加** (需要实时反量化 W₀)                        |
| **适用场景**    | 通用 PEFT 场景，对推理速度敏感    | 对参数效率要求极致，预算有限        | **显存极其有限**，需要微调**超大模型**                  |
| **主要缺点**    | 固定秩，不够灵活                  | 实现相对复杂                        | 依赖特定硬件/库支持 (CUDA, bitsandbytes)；计算仍需 BF16 |

**面试时，你可以这样展示你的理解**：

*   **当被问到“如何高效微调大模型”时**：你可以首先提到全量微调的挑战（显存、计算、存储），然后引出PEFT的概念。接着可以介绍LoRA作为其中的代表，解释其低秩适应的核心原理（冻结原模型，加旁路BA，只训练BA），强调其参数高效和推理无损的优点。
*   **当被问到LoRA的局限性时**：你可以指出其固定秩`r`的问题，然后引出AdaLoRA，解释它如何通过SVD参数化和重要性评估来动态分配参数预算，实现更优的参数利用率。
*   **当被问到“如何在单卡上微调非常大的模型（如65B）”时**：QLoRA就是标准答案。你需要解释它不仅仅是减少训练参数（那是LoRA做的），更关键的是通过4-bit量化（特别是NF4）大幅压缩了基础模型的体积，配合双重量化和分页优化器，解决了模型加载和训练过程中的显存瓶颈。要强调计算时需要反量化到BF16。
*   **对比三者**：清晰地说明它们解决的核心问题、技术手段和优缺点的差异，如上表所示。

掌握了LoRA、AdaLoRA和QLoRA的原理和应用场景，你就能在大模型微调这个话题上展现出扎实的功底和与时俱进的技术视野。

希望这篇博客对你有帮助！如果你有任何疑问，欢迎在评论区留言讨论！



参考：

https://zhuanlan.zhihu.com/p/636215898