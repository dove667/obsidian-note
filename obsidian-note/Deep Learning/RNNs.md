  

# 循环神经网络

![[image 2.png|image 2.png]]

循环指的是在预测是，上一个token的输出作为下一次输入。右图沿时序展开。

但是在训练时，输入的是target，然后计算predict和target之间的损失

![[image 1.png]]

预测就是，计算既定事实的条件下未来事件的概率

$$P(x_t \mid x_{t-1}, \ldots, x_1)$$

处理序列模型的两种方法：1. 马尔可夫模型（Markov Model, MM），构建自回归模型。

这种策略的假设是当下状态只于前t个状态有关。

2. 隐马尔科夫模型（Hidden Markov Model, HMM），构建隐藏变量自回归模型。这种策略的假设是状态是隐藏的，我们只能通过观测值来推测背后的真实状态。

  

_语言模型_（language model）的目标是估计序列的联合概率

$$P(x_1, \ldots, x_T) = \prod_{t=T}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$

对于MM，有n元语法

$$\begin{split}\begin{aligned}P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).\end{aligned}\end{split}$$

现实中n元语法有这样的统计规律，称为_齐普夫定律_（Zipf’s law）  
  
  

![[image 2 2.png|image 2 2.png]]

$$n_i \propto \frac{1}{i^\alpha} \\ \log n_i = -\alpha \log i + c,$$

极少数常用词（stop words）占据大头，然后词频迅速衰减，最后是极少出现的n元组。

基于古典概型的MM自回归模型存在很多缺陷。深度学习基于HMM，设计RNNs。

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1})\\h_t = f(x_{t}, h_{t-1}).$$

![[image 3.png]]

RNNs只是中间递归层，即输出H，要映射到vocab才能最为预测输出，也就是还要过一个Dense layer

在经典RNNs的按时间反向传播中，存在矩阵递归相乘，容易产生梯度爆炸/消失。需要进行梯度剪裁**Gradient Clipping。**

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

同时如果是连小批量采样，还要进行梯度截断，减少计算量。

![[image 4.png]]

# 序列数据预处理

1. 将文本作为字符串加载到内存中。
2. 将字符串拆分为词元token（例如单词或字符）。
3. 建立词表vocabulary，将拆分的词元映射到数字索引。
4. 将文本转换为数字索引序列。

现代神经网络

GRU 门控循环单元

![[image 5.png]]

长短期记忆网络

![[image 6.png]]

深度循环神经网络

![[image 7.png]]

双向循环神经网络

![[image 8.png]]

编码器-解码器架构

![[image 9.png]]

![[image 10.png]]

- “编码器－解码器”架构可以将**长度可变**的序列作为输入和输出，因此适用于机器翻译等序列转换问题。
- 编码器将长度可变的序列作为输入，并将其转换为具有**固定形状**的编码状态（上下文变量，cat到解码器的每次输入中）。
- 解码器将具有固定形状的编码状态映射为长度可变的序列。

序列到序列学习Seq2Seq

![[image 11.png]]

语言模型的预测就是在语料corpus中选择概率最大的token，相当于多元分类，用交叉熵损失。

Dense输出的是个token的预测分数logit，用nn.softmax归一。

但是有些特殊token如<ukn><bos><eos>不参与softmax，用布尔掩码mask修饰。

## 布尔掩码

- **定义**
    
    布尔掩码就是一个元素为布尔值（True 或 False）的张量或数组，其形状通常与目标数据张量相同或是能通过广播规则匹配目标数据张量的形状。
    
    它用来指示哪些位置符合某个条件或需要被选中。例如，如果我们有一个张量 `X`，执行 `mask = X > 0` 后，`mask` 中为 True 的位置表示 `X` 中对应元素大于 0。
    
- **用途**
    - **条件筛选**：从数据中选出满足条件的元素。
    - **条件赋值**：将满足条件的位置赋予新值。
    - **索引操作**：通过布尔索引获取或修改数据。

# 评估方法 BLEU（bilingual evaluation understudy）

$$\exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$

pn表示n元语法的精度。exp保正，由于短序列出现的概率比长序列高，min项惩罚短序列，1/2^n项奖励长序列。当预测序列与标签序列完全相同时，BLEU为1。

# 束搜索 BeamSearch

- 序列搜索策略包括贪心搜索、穷举搜索和束搜索。
- 贪心搜索所选取序列的计算量最小，但精度相对较低。
- 穷举搜索所选取序列的精度最高，但计算量最大。
- 束搜索通过灵活选择束宽，在正确率和计算代价之间进行权衡。

![[image 12.png]]