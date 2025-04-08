# LLM Learning & Algorithm Notes

📚 本仓库用于记录我在学习大语言模型（LLM）过程中的技术思考与算法题解笔记，持续更新中。

## 项目结构

### 📖 learn_everyday 
- **LLM技术探究笔记**：包含对模型组件的逐行分析、实验记录与原理推导，例如：
  - 激活函数对比（ReLU, GELU, Swish等）
  - PyTorch自动微分机制详解
  - 前向传播与反向传播中的激活值传递路径
  - 模型参数初始化策略实验
  - 注意力机制计算过程可视化

### ⚙️ 算法
- **LeetCode/剑指Offer题解**：按题型分类的解题模板与优化思路：
  - 该文件夹将用于存放我在刷算法题过程中记录的一些笔记和解题思路。
  - 未来我会在这里分享一些常见算法和数据结构的总结，以及我在解决具体问题时的思考过程。
  
## 使用说明
1. **克隆仓库**：
   ```bash
   git clone https://github.com/cloud0912/learn_LLM.git
2. **查看笔记**：各文件夹内包含Markdown格式的技术文档，推荐使用[VSCode](https://code.visualstudio.com/) + [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)插件获得最佳阅读体验。

## 知识体系

笔记中会交叉引用相关论文与开源实现，例如：

- Transformer架构图解 → 参考《Attention Is All You Need》
- 混合精度训练实现 → 关联NVIDIA Apex库
- 梯度裁剪策略 → 对比PyTorch官方文档

## 🤝 贡献

欢迎通过Issue或PR提交：

- 技术观点讨论
- 代码实现改进
- 文献资料补充

## License

本仓库采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 协议，学术使用请注明出处。   
