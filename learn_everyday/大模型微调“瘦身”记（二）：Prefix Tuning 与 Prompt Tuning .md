## 大模型微调“瘦身”记（二）：Prefix Tuning 与 Prompt Tuning  

上次我们聊了Adapter Tuning系列，今天我们继续深入PEFT（参数高效微调）的世界，聚焦另外两种极具影响力的技术：**Prefix Tuning** 和 **Prompt Tuning**。

想象一下，你想让一个博学的“通才”大模型（比如GPT或T5）变成某个领域的“专家”，但又不想重新训练它所有的“脑细胞”（参数）。传统方法是人工设计一些“提示语”（Discrete Prompts），比如问模型“翻译成法语：中国的首都是哪里？”。但这种方法费时费力，而且提示语稍微变动一点，效果可能天差地别。自动化搜索提示语又太贵。

Prefix Tuning和Prompt Tuning就是为了解决这些问题而生的。它们的核心思想相似：**我们不改变大模型本身，而是给它加上一些可学习的“引导信号”（Continuous Prompts 或 Virtual Tokens），让模型在处理特定任务时表现更好。** 这就像给大模型一个“魔法咒语”或“专属指令”，而我们只需要学习这个“咒语”怎么念！

### 一、 Prefix Tuning：为模型注意力机制注入“引导”前缀

**1. 背景：离散提示的困境与微调的代价**

Prefix Tuning（出自论文 *Prefix-Tuning: Optimizing Continuous Prompts for Generation*）主要想解决两个问题：

*   **离散提示语的局限性**：人工设计的提示语（如"Translate English to German: ..."）效果不稳定，对措辞敏感；自动化搜索离散提示语计算成本高。
*   **全量微调的成本**：为每个任务保存一个完整的大模型副本，存储和计算开销巨大。

Prefix Tuning提出：**我们不改变模型主体，只在模型处理输入时，给它的“注意力”机制（Attention）加上一小段可学习的、连续的“前缀”（Prefix）参数。** 这样，每个任务只需要存储和训练对应的小小的前缀参数即可。

<img src="https://picx.zhimg.com/v2-f13d18b75046452ba0cd4986d7605177_1440w.jpg" alt="img" style="zoom:50%;" />

<center>*(图1: Prefix Tuning 示意图，冻结LM，只训练Prefix)*</center>

**2. 技术原理：深入每一层的“虚拟”引导**

Prefix Tuning的关键在于，它不是简单地在输入文本前加几个词，而是**在Transformer的每一层（或部分层）的注意力计算中，都引入可学习的Prefix向量。**

*   **输入形式**：
    *   对于**自回归模型 (Autoregressive, 如GPT系列)**：逻辑上将输入构造成 `[PREFIX; x; y]` 的形式，其中 `PREFIX` 是可学习的连续向量序列，`x` 是输入文本，`y` 是要生成的文本。
    *   对于**编码器-解码器模型 (Encoder-Decoder, 如T5, BART)**：通常在Encoder的输入和Decoder的输入前都加上Prefix，形式如 `[PREFIX_enc; x; PREFIX_dec; y]`。

<img src="https://picx.zhimg.com/v2-1dda287347e7eeed655598f2df63d295_1440w.jpg" alt="img" style="zoom:67%;" />

<center>*(图2: Prefix Tuning在不同模型架构下的应用)*</center>

*   **核心机制：修改注意力层**
    Prefix Tuning并不直接修改模型的词嵌入层。它真正的作用点在**多头注意力（Multi-Head Attention）** 模块内部。

    我们知道，注意力机制的核心是计算Query (Q), Key (K), 和 Value (V) 向量。对于一个Transformer层 `i`，原始的注意力计算（简化表示）是：
    `h_i = Attention(Q_i, K_i, V_i)`
    其中 `Q_i = x_i @ W_q`, `K_i = x_i @ W_k`, `V_i = x_i @ W_v` ( `x_i` 是该层的输入)。

    Prefix Tuning的做法是：保持原始的 `Q_i`, `K_i`, `V_i` 计算方式不变（因为 `W_q, W_k, W_v` 属于冻结的预训练模型部分），但是**将可学习的Prefix向量序列 `P_k` 和 `P_v`（长度为 `L_prefix`）拼接到原始的 `K_i` 和 `V_i` 序列之前**。

    `K'_i = concat([P_k, K_i])`
    `V'_i = concat([P_v, V_i])`
    `h_i = Attention(Q_i, K'_i, V'_i)`

    这意味着，在计算注意力得分时，Query向量会同时关注原始输入序列和这段可学习的Prefix；在加权求和Value时，也会包含来自Prefix的信息。这段Prefix就像一个“隐形”的任务指令，引导着模型在每一层的信息处理。

    
    
    **公式表示 (单个注意力头，第 `i` 层)：**
    
    设Transformer层输入为$H \in \mathbb{R}^{n×d}$，前缀参数$P \in \mathbb{R}^{l×d}$，则扩展后的键值对为：
    $$
    K' = [P_k; W_kH], \quad V' = [P_v; W_vH]
    $$
    注意力计算变为：
    $$
    \text{Attention}(Q,K',V') = \text{softmax}\left(\frac{QK'^T}{\sqrt{d}}\right)V'
    $$
    
    
    
    设 `P_k` 和 `P_v` 是该层对应的、可学习的Prefix Key和Value矩阵 (维度 `L_prefix x d_k` 和 `L_prefix x d_v`)。
    `K'_i = [P_k ; K_i]` (矩阵拼接，维度变为 `(L_prefix + L_seq) x d_k`)
    `V'_i = [P_v ; V_i]` (矩阵拼接，维度变为 `(L_prefix + L_seq) x d_v`)
    `AttentionScores = softmax((Q_i @ K'_i^T) / sqrt(d_k))` (维度 `L_seq x (L_prefix + L_seq)`)
    `h_i = AttentionScores @ V'_i` (维度 `L_seq x d_v`)
    
    
    
*   **参数生成：Reparameterization技巧**
    直接优化高维度的Prefix向量 `P_k` 和 `P_v` (维度 `L_prefix x d_model`) 可能导致训练不稳定。因此，Prefix Tuning采用了一个**重参数化（Reparameterization）**技巧：
    
    1.  创建一个更小的**Prefix参数矩阵 `P_θ`** (维度 `L_prefix x k`)。
    2.  使用一个**小型的前馈网络 (MLP)** 将 `P_θ` 映射到实际用于注意力计算的 `P_k` 和 `P_v`。
        `[P_k ; P_v] = MLP(P_θ)` (这里 `[P_k ; P_v]` 表示拼接后的维度 `L_prefix x (d_k + d_v) * num_heads` 或类似结构)
    3.  在训练时，只**更新 `P_θ` 和 MLP 的参数**。
    4.  训练完成后，**MLP可以丢弃**，只需要存储最终计算得到的 `P_k` 和 `P_v` 即可用于推理。这个MLP只在训练阶段帮助稳定优化。
    
    <img src="https://pic1.zhimg.com/v2-0bf4ac54160cb44f5cbdfa7cb38a4c1c_1440w.jpg" alt="img" style="zoom:67%;" />

    <center>*(图4: 使用MLP进行重参数化生成Prefix向量)*</center>
    
*   **代码示例 (PyTorch风格，概念性展示Attention修改)**

    ```python
    import torch
    import torch.nn as nn

    class PrefixAttention(nn.Module):
        def __init__(self, d_model, nhead, prefix_len):
            super().__init__()
            self.nhead = nhead
            self.d_head = d_model // nhead
            self.prefix_len = prefix_len

            # 预训练模型的Q, K, V投影层 (假设已加载并冻结)
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            # 可学习的Prefix参数 (实际应用中可能通过MLP生成)
            # 这里简化为直接存储 P_k, P_v
            # 维度: (num_heads, prefix_len, d_head)
            self.prefix_k = nn.Parameter(torch.randn(self.nhead, self.prefix_len, self.d_head))
            self.prefix_v = nn.Parameter(torch.randn(self.nhead, self.prefix_len, self.d_head))

        def forward(self, query, key, value, attention_mask=None):
            # query, key, value shape: (batch_size, seq_len, d_model)
            batch_size = query.size(0)
            seq_len = query.size(1)

            # 1. 计算原始Q, K, V
            q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2) # (bs, nh, seq, dh)
            k = self.k_proj(key).view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2) # (bs, nh, seq, dh)
            v = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2) # (bs, nh, seq, dh)

            # 2. 扩展Prefix到batch维度并与原始K, V拼接
            # prefix_k/v shape: (nh, prefix_len, dh) -> (bs, nh, prefix_len, dh)
            prefix_k_batch = self.prefix_k.unsqueeze(0).expand(batch_size, -1, -1, -1)
            prefix_v_batch = self.prefix_v.unsqueeze(0).expand(batch_size, -1, -1, -1)

            # k' shape: (bs, nh, prefix_len + seq_len, dh)
            k_prime = torch.cat([prefix_k_batch, k], dim=2)
            # v' shape: (bs, nh, prefix_len + seq_len, dh)
            v_prime = torch.cat([prefix_v_batch, v], dim=2)

            # 3. 计算注意力
            # q shape: (bs, nh, seq, dh)
            # k_prime.transpose(-2, -1) shape: (bs, nh, dh, prefix_len + seq_len)
            attn_scores = torch.matmul(q, k_prime.transpose(-2, -1)) / (self.d_head ** 0.5) # (bs, nh, seq, prefix+seq)

            # Apply attention mask (需要处理好prefix部分的mask)
            if attention_mask is not None:
                 # Assuming attention_mask shape (bs, seq, prefix+seq) or similar
                 # Need to expand mask to (bs, nh, seq, prefix+seq)
                 # Handle prefix mask carefully: query tokens should attend to prefix, but not vice versa if causal
                 # For simplicity, we might assume prefix is always attended to.
                 # A common mask might be (bs, 1, seq_len, prefix_len + seq_len)
                 # prefix_mask = torch.zeros(batch_size, 1, seq_len, self.prefix_len, device=query.device)
                 # combined_mask = torch.cat([prefix_mask, attention_mask], dim=-1) # Example, details matter
                 # attn_scores = attn_scores.masked_fill(combined_mask == 0, -1e9)
                 pass # Masking logic here

            attn_weights = torch.softmax(attn_scores, dim=-1)
            # attn_weights shape: (bs, nh, seq, prefix+seq)
            # v_prime shape: (bs, nh, prefix+seq, dh)
            attn_output = torch.matmul(attn_weights, v_prime) # (bs, nh, seq, dh)

            # 4. Reshape and output projection
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.nhead * self.d_head) # (bs, seq, d_model)
            output = self.out_proj(attn_output)
            return output
    ```

*   **实验发现**：
    *   **层级影响**：只在输入嵌入层加Prefix（类似Prompt Tuning）效果远不如在每一层都加Prefix。这表明修改模型深层的激活状态对于引导模型行为至关重要。
    *   **位置影响**：Prefix-Tuning (`[PREFIX; x; y]`) 通常优于Infix-Tuning (`[x; INFIX; y]`)。

<img src="https://pic3.zhimg.com/v2-c24840aa6ebac63b5fcc450cd8354aa0_1440w.jpg" alt="img" style="zoom:50%;" />                 <img src="https://pic1.zhimg.com/v2-f7be2bbc41070aeb0f3ed6edf727a28c_1440w.jpg" alt="img" style="zoom:50%;" />
(图5: 消融实验 - 仅调整Embedding层 vs 全层调整)                                                             *(图6: 消融实验 - Prefix vs Infix)*

### 二、 Prompt Tuning：极简主义的“提示”艺术

**1. 背景：寻求更极致的简洁与高效**

Prompt Tuning（出自论文 *The Power of Scale for Parameter-Efficient Prompt Tuning*）可以看作是Prefix Tuning的一个**极大简化版**。它同样旨在解决全量微调的成本问题和离散提示的局限性，但它追求的是**更少的修改和更少的参数**。

**2. 技术原理：仅在输入层添加“虚拟”Token**

Prompt Tuning的核心思想极其简洁：

*   **冻结整个预训练语言模型**的所有参数。
*   **只在输入嵌入层（Input Embedding Layer）添加少量可学习的“Prompt”向量（Virtual Tokens）**。

<img src="https://pica.zhimg.com/v2-d2eaf41d3da078a87ebe9e63b4c199d8_1440w.jpg" alt="img" style="zoom: 67%;" />

<center>*(图7: Prompt Tuning 示意图，仅在输入层添加可学习Prompt)*</center>

具体来说，假设原始输入序列 `X = [x_1, x_2, ..., x_n]` 对应的词嵌入是 `E = [e_1, e_2, ..., e_n]`。Prompt Tuning会初始化 `k` 个可学习的Prompt向量 `P = [p_1, p_2, ..., p_k]`，然后将它们拼接到输入嵌入序列 `E` 的前面：

`E'_input = concat([P, E]) = [p_1, ..., p_k, e_1, ..., e_n]`

这个拼接后的序列 `E'_input` 作为Transformer模型第一层的输入。**后续所有Transformer层的计算都保持不变，也不再引入额外的参数。**

**训练时，只更新这 `k` 个Prompt向量 `P` 的参数。**

*   **代码示例 (PyTorch风格，概念性展示输入层修改)**

    ````python
    import torch
    import torch.nn as nn
    
    class PromptTuningWrapper(nn.Module):
        def __init__(self, pretrained_model, prompt_len=20):
            super().__init__()
            self.model = pretrained_model
            self.prompt_len = prompt_len
    
            # 获取原始模型的embedding层和配置
            self.embeddings = self.model.get_input_embeddings()
            self.embedding_dim = self.embeddings.embedding_dim
    
            # 初始化可学习的Prompt向量
            self.prompt_embeddings = nn.Parameter(torch.randn(self.prompt_len, self.embedding_dim))
    
            # 冻结预训练模型所有参数
            for param in self.model.parameters():
                param.requires_grad = False
            # 解冻Prompt参数
            self.prompt_embeddings.requires_grad = True
    
        def forward(self, input_ids, attention_mask=None, **kwargs):
            # input_ids shape: (batch_size, seq_len)
            batch_size = input_ids.size(0)
    
            # 1. 获取原始输入的嵌入
            # inputs_embeds shape: (batch_size, seq_len, embedding_dim)
            inputs_embeds = self.embeddings(input_ids)
    
            # 2. 扩展Prompt向量并与输入嵌入拼接
            # prompt_embeds shape: (prompt_len, embedding_dim) -> (batch_size, prompt_len, embedding_dim)
            prompt_embeds_batch = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    
            # combined_embeds shape: (batch_size, prompt_len + seq_len, embedding_dim)
            combined_embeds = torch.cat([prompt_embeds_batch, inputs_embeds], dim=1)
    
            # 3. 构造新的attention mask (非常重要!)
            # 原始mask shape: (batch_size, seq_len)
            if attention_mask is not None:
                # 创建prompt部分的mask (全1，表示都有效)
                prompt_mask = torch.ones(batch_size, self.prompt_len, dtype=attention_mask.dtype, device=attention_mask.device)
                # 拼接mask shape: (batch_size, prompt_len + seq_len)
                combined_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            else:
                # 如果没有提供mask，则假设所有输入都有效
                combined_attention_mask = torch.ones(batch_size, self.prompt_len + input_ids.size(1), device=input_ids.device)
                    
            # 4. 将拼接后的嵌入和新mask传入模型
            # 注意：需要传入inputs_embeds而不是input_ids
            #       需要传入新的attention_mask
            outputs = self.model(inputs_embeds=combined_embeds,
                                 attention_mask=combined_attention_mask,
                                 **kwargs) # 传递其他可能需要的参数
            return outputs
    
    ```



*   **关键发现：规模的力量 (The Power of Scale)**
    Prompt Tuning 最重要的发现是：**它的效果与模型规模密切相关。**
    
    *   对于较小的模型（例如T5-Base/Large），Prompt Tuning的效果通常不如全量微调，甚至不如Prefix Tuning。
    *   但是，**当模型规模增大到非常大（例如T5-XXL 11B参数）时，Prompt Tuning的效果可以追平甚至超过全量微调！**
    
    这说明，对于足够大的模型，仅仅通过调整输入端的少量连续提示，就足以引导模型适应下游任务，而无需改变其内部参数。

<img src="https://pic2.zhimg.com/v2-06a7fd88bd29877341a3b6fc0bbcbb69_1440w.jpg" alt="img" style="zoom:67%;" />

<center>*(图8: Prompt Tuning 性能随模型规模增长而逼近全量微调)*</center>

*   **Prompt Ensembling**
    Prompt Tuning还提出了一种简单的集成方法：**Prompt Ensembling**。即针对同一个任务，训练多个不同的Prompt（初始化不同或长度不同），在推理时将它们的预测结果进行集成（例如投票或平均概率）。这比传统的模型集成（训练多个完整模型）成本低得多。

<img src="https://picx.zhimg.com/v2-57f517168ec95f694ec9f5020b95b4cf_1440w.jpg" alt="img" style="zoom: 50%;" />

<center>*(图9: Prompt Ensembling 示意图)*</center>

*   **初始化与长度**
    *   **初始化**：用任务相关的词汇（如类别标签对应的词）的嵌入来初始化Prompt向量，通常比随机初始化效果更好，尤其是在模型规模不大时。但对于超大模型，初始化方法的影响会减小。
    *   **Prompt长度**：Prompt长度通常不需要很长，实验表明长度在20左右通常就能达到不错的效果。增加长度带来的边际效益递减。同样，对于超大模型，即使较短的Prompt也能表现良好。

<img src="https://pica.zhimg.com/v2-d0e8a236f95fc534595511377775d352_1440w.jpg" alt="img" style="zoom:67%;" />

<center>*(图10: Prompt 初始化和长度对性能的影响)*</center>

### 三、 Prefix Tuning vs. Prompt Tuning：如何抉择？

| 特性                 | Prefix Tuning                          | Prompt Tuning                              |
| :------------------- | :------------------------------------- | :----------------------------------------- |
| **修改位置**         | Transformer **每一层**的Attention K, V | **仅输入嵌入层**                           |
| **参数量**           | 较少 (e.g., 0.1% - 3% of total)        | **极少** (e.g., <0.1% of total)            |
| **是否改动模型结构** | 是 (逻辑上修改了Attention计算)         | 否 (仅改变输入表示)                        |
| **重参数化(MLP)**    | 通常需要 (稳定训练)                    | **不需要**                                 |
| **性能**             | 在中小型模型上通常优于Prompt Tuning    | 在**超大模型**上可匹敌甚至超越全量微调     |
| **实现复杂度**       | 相对较高                               | **非常简单**                               |
| **适用场景**         | 对性能要求高，模型规模适中，生成任务等 | 追求极简，模型规模巨大，对性能要求不是极端 |

**总结来说：**

*   如果你使用的是**非常巨大**的模型（如 >10B 参数），且追求**极致的简洁和参数效率**，**Prompt Tuning** 是一个非常有吸引力的选择，其效果可能不输全量微调。
*   如果你的模型规模**不是特别巨大**，或者你对**性能有更高要求**（尤其是在生成任务或复杂推理任务上），**Prefix Tuning** 可能是更稳妥的选择，因为它通过影响模型的每一层来提供更强的引导能力，尽管参数量和实现复杂度稍高。

### 四、 面试小贴士

当面试官问到Prefix Tuning和Prompt Tuning时，他们可能想了解：

1.  **它们解决了什么问题？** (回答：全量微调成本高、存储大；离散提示效果不稳定、设计困难。)
2.  **核心思想是什么？** (回答：冻结大模型，添加少量可学习的连续向量/虚拟Token来引导模型适应任务。)
3.  **Prefix Tuning怎么工作的？** (回答：在**每层**Attention的K和V前拼接可学习的Prefix向量；通常用MLP重参数化稳定训练。)
4.  **Prompt Tuning怎么工作的？** (回答：**仅在输入嵌入层**前拼接可学习的Prompt向量；无需MLP；性能与模型规模强相关。)
5.  **两者的主要区别？** (回答：修改位置、参数量、是否需要MLP、性能与模型规模的关系。)
6.  **各自的优缺点？** (参考上面的对比表格。)
7.  **你知道“The Power of Scale”在Prompt Tuning里指什么吗？** (回答：指Prompt Tuning的性能随着模型规模增大能逼近全量微调的现象。)

掌握这些要点，并能清晰地解释其原理和区别，将有助于你在面试中展现对PEFT技术的深入理解。

### 结语

Prefix Tuning和Prompt Tuning代表了参数高效微调领域中“Prompting”流派的两种重要方法。它们巧妙地利用了大型预训练模型的强大基础能力，通过学习少量“引导”参数，就能在下游任务上取得优异表现，极大地降低了模型适配的门槛。理解它们不仅有助于你选择合适的微调策略，更是深入理解大模型工作机制的重要一步。

参考：

https://zhuanlan.zhihu.com/p/635686756

 Prefix-Tuning原始论文《Prefix-Tuning: Optimizing Continuous Prompts for Generation》  
 Prompt Tuning论文《The Power of Scale for Parameter-Efficient Prompt Tuning》 

