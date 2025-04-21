  

# 注意力机制 Attention

注意力提示分为自主和非自主。先前我们处理数据的方式都是非自主的（有什么看什么），注意力机制提供一种自主（有侧重）的思路。

上面提到的搜索方法对于大量且长短不一的数据操作起来十分麻烦。借鉴数据库查询，引入Query查询，键Key-值Value机制。key-value就是非自住提示，query是我们想要的自主提示。

## 定义注意力

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$

中间alfa(q,k)成为注意力权重，这个操作称为Attention Pooling，也就是对目标值跟据权重加权求和。

注意力权重可由softmax计算

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}.$$

注意力机制可以设计成可微分的深度学习模型，或者不可微分的强化学习模型。过去还有不可学习（参数）的统计模型。

![[image 13.png|image 13.png]]

讲一下一个过去类似的模型

### **Nadaraya-Watson 核回归**

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$

这也是一种注意力模型，中间的分式就是注意力权重。K是核kernel函数。

常用的kernel有

$$\begin{split}\begin{aligned}\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gaussian;} \\\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}\end{aligned}\end{split}$$

### 注意力评分函数 Attention Score Function

类比核函数，计算数据点于目标点的相似度，据此分配注意力权重

![[image 1 2.png|image 1 2.png]]

  

### 和注意力Additive attention

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$

把query，key都映射到一个隐藏空间R^h，相加激活。

可以理解为把q，k放入一个单层感知机。最后于另一个长为h的向量点积输出一个数作为注意力分数

### 点积注意力 Dot Product Attention

$$a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$

根号d用于将方差归一化（q，k从高斯分布中取样，点积方差为d）

可以用点积作为注意力函数是因为（取高斯核中的指数项）

$$a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2.$$

q，k的L2距离只于点积项有关。k方项由于层归一化（避免某些k过大主导注意力），影响差别不大，q方项都一样。

不过点击注意力一般要求q，k长度一致，不一致的话可以经过一个W调整。i.e.qT*W*k

## Bahdanau注意力的Seq2Seq

$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

c是上下文变量，s是query，h既是key也是value。

- **如果 Key = Value**：一般用于 **自注意力（Self-Attention）** 或 **Encoder-Decoder Attention**，例如机器翻译。
- **如果 Key ≠ Value**：一般用于 **跨模态匹配（Cross-Modal Matching）** 或 **信息检索（Retrieval Tasks）**，例如搜索引擎、阅读理解、视觉问答等。

![[image 2 3.png|image 2 3.png]]

于之前的seq2seq区别就是，**用注意力分数而非整个隐藏状态H作为上下文变量**，从而让机器有针对性地翻译。

## 多头注意力 Multi-head Attention

![[image 3 2.png|image 3 2.png]]

一般attention层中的映射空间的h就用RNNs（如果是）的隐藏层。

多头注意力将n维映射空间分给多个头head，允许不同的头关注不同表征子空间（representation subspace）或角度，最终经过FC汇聚。

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v}\\\begin{split}\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.\end{split}\\p_q h = p_k h = p_v h = p_o$$

这么做可以最大化并行计算。

## 自注意力 Self-Attention

query，key，value都是同一个序列

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

  

三个架构处理序列模型

![[image 4 2.png|image 4 2.png]]

自注意力机制可以轻松学到远距离依赖关系，但是计算成本巨大

### 位置编码

自注意力为了并行计算放弃了顺序关系，而顺序对序列来说很重要。

通过在输入表示中添加 _位置编码_（positional encoding）来注入绝对的或相对的位置信息。 位置编码可以通过学习得到也可以直接固定得到。  
基于三角函数的固定位置编码  

输入X(Rn*d)表示一个词元的n维嵌入，位置编码P（shape(X)), X+P

由位置嵌入矩阵实现（max_len, num_embed）

$$\begin{split}\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}\end{split}$$

第i行，第2j和2j+1列的元素。行数代表词元，移动相位；列数代表嵌入维度，改变频率（负相关）

![[image 5 2.png|image 5 2.png]]

  

之所以使用三角函数，是因为任意位置的偏移offset可以用偏移量delta线性变换得到，与具体位置无关。

$$\begin{split}\begin{aligned}&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\=&\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},\end{aligned}\end{split}$$

# Transformer

## 基本架构

Add residual残差

所以任何sublayer的input和ouput大小一致

FFN positionwise forward feedback前馈神经网络

注意：现代llm架构，add+norm通常在attention/FFN前面

![[image 6 2.png|image 6 2.png]]

![[image 7 2.png|image 7 2.png]]

这里使用的是层归一化LN。CNN中常用批量归一化BN。

BN是对整个 batch 的某个特征在多个样本上归一化，”减少内部协变量偏移”，注重特征提取在不同样本之间的不变性

LN是对单个样本内所有特征之间的归一化，注重不同序列样本之间的独立性

decoder的key-value在每一步会加上已生成的输出作为输入，训练时会mask掉当前时间步之后的target

![[image 8 2.png|image 8 2.png]]

对于自然语言，样本是一段完整的文字序列，len是时间步长或序列长度或词元数量或position，d是特征维度或num_hidden units或嵌入维度

## transformer for vision (ViT)

![[image 9 2.png|image 9 2.png]]

## 预训练Transformer 三种模式

### 仅编码器 BERT（Bidirectional Encoder Representations from Transformers）

![[image 10 2.png|image 10 2.png]]

预训练任务：

Masked Language Model (MLM) – 掩码语言模型

Next Sentence Prediction (NSP) – 下一句预测：  
特性：  

双向上下文学习，理解句子含义时更加全面（NLU natural language understanding）

自监督，不需要人工标注的数据

下游任务：

- 文本分类
- 问答系统
- 文本生成
- 情感分析
- 机器翻译

## 编码器-解码器 T5 (Text-to-Text Transfer Transformer)

![[image 11 2.png|image 11 2.png]]

![[image 12 2.png|image 12 2.png]]

预训练任务：

Span Corruption（文本跨度填空）（mask多个词）

特性：

生成能力强，可以生成任意长度的序列

下游任务：

翻译、摘要、QA、文本分类等

## 仅解码器 GPT（Generative pre-trained transformer)

![[image 13 2.png|image 13 2.png]]

预训练任务：

Causal Language Modeling（CLM，因果语言建模）

特性：

基于注意力机制的自回归（autoregressive）语言模型

下游任务：

对话、写作、代码生成