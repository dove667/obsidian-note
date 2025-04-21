# 使用 Transformers库

## pipeline

![[image 14.png|image 14.png]]

包含整个流水线，从预处理到后处理

```Python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

```Python
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

  

Pipeline的工作流程包含以下步骤：

根据输入的任务类型和 checkpoint，自动加载对应的 tokenizer 和 model

- 预处理 (Preprocessing):- 将输入文本转换为模型可以理解的格式- 进行分词 (tokenization)- 添加特殊标记（如[CLS], [SEP]等）- 将token转换为对应的ID
- 模型推理 (Model Inference):- 将处理后的输入传入预训练模型- 模型进行计算并输出原始预测结果
- 后处理 (Post-processing):- 将模型的原始输出转换为人类可理解的格式- 对结果进行格式化（如标签和置信度分数）

输入：

- 可以是单个文本字符串或文本列表
- 支持不同任务类型的特定输入格式

输出：

- 返回包含预测结果的字典或字典列表
- 结果通常包含：- label: 预测的标签- score: 预测的置信度分数（0-1之间）

## Tokenizer

进行数据预处理和后处理

```Python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

在 Hugging Face 的 transformers 库中，AutoClass（如 AutoModel、AutoTokenizer 等）并不会在你提供的 checkpoint 名字“不完全对”时自动加载“最相近”的模型或 tokenizer。它的工作方式是基于精确匹配的逻辑，而不是模糊匹配或猜测。如果你提供的 checkpoint 名称有误，AutoClass 会尝试直接加载该名称对应的资源，如果找不到，就会抛出错误。

AutoClass 的核心功能是通过 from_pretrained() 方法，根据你提供的 checkpoint 名称（通常是 Hugging Face Hub 上的模型或 tokenizer 的标识符，或者本地路径），自动推断并加载对应的模型架构或 tokenizer 类型。它的“智能”体现在以下几个方面：

1. **自动推断类型**：你只需提供 checkpoint 的名称（例如 "bert-base-uncased"），AutoClass 会根据该 checkpoint 的配置文件（config.json）中的 model_type 字段，自动选择正确的模型类（如 BertModel）或 tokenizer 类（如 BertTokenizer）。
2. **一致性检查**：如果 checkpoint 名称有效，AutoClass 会确保加载的模型或 tokenizer 与该 checkpoint 的配置相匹配。它不会尝试加载一个“相近”但不完全匹配的模型。
3. **错误处理**：如果 checkpoint 名称拼写错误、不存在，或者本地缓存中没有相应的文件，AutoClass 会抛出类似 OSError 或 ValueError 的异常，提示你资源无法找到。
    
      
    
    ```Python
    from transformers import AutoTokenizer
    
    # 1. 加载 AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",        # checkpoint 名称
        use_fast=True,              # 使用快速分词器（推荐，基于 Rust 实现）
        add_prefix_space=False,     # 是否在开头添加空格（对某些模型如 RoBERTa 有用）
        cache_dir="./cache",        # 指定缓存目录，避免重复下载
    )
    
    # 2. 输入文本
    text = "Hello, how are you today? I am Grok, built by xAI!"
    
    # 3. 使用 tokenizer 处理文本（设置常用参数）
    encoded_output = tokenizer(
        text,                       # 输入文本（字符串或字符串列表）
        add_special_tokens=True,    # 是否添加特殊标记（如 [CLS], [SEP]）
        max_length=20,             # 最大序列长度（超过会截断）
        padding="max_length",       # 填充到 max_length（可选："longest", False 等）
        truncation=True,            # 超过 max_length 时截断
        return_tensors="pt",        # 返回 PyTorch 张量（可选："tf" 或 None）
        return_attention_mask=True, # 返回 attention mask
        return_token_type_ids=True, # 返回 token type IDs（用于区分句子对。在句子对任务中，第一个句子为 0，第二个句子为 1）
    )
    
    # 4. 输出结果 字典
    print("原始文本:", text)
    print("\nTokenizer 输出:")
    print("input_ids:", encoded_output["input_ids"])
    print("attention_mask:", encoded_output["attention_mask"])
    print("token_type_ids:", encoded_output["token_type_ids"])
    print("\n解码回文本:", tokenizer.decode(encoded_output["input_ids"][0]))
    print("分词结果:", tokenizer.convert_ids_to_tokens(encoded_output["input_ids"][0]))
    
    # 5. 额外功能：批量输入
    batch_text = ["Hello world!", "I am Grok."]
    batch_encoded = tokenizer(
        batch_text,
        padding=True,               # 自动填充到最长序列长度
        truncation=True,
        max_length=10,
        return_tensors="pt"
    )
    print("\n批量输入结果:")
    print("input_ids:", batch_encoded["input_ids"])
    print("attention_mask:", batch_encoded["attention_mask"])
    ```
    
    ```Plain
    原始文本: Hello, how are you today? I am Grok, built by xAI!
    
    Tokenizer 输出:
    input_ids: tensor([[  101,  7592,  1010,  2129,  2024,  2017,  2651,  1029,  1045,  2572,
              20057,  1010,  2328,  2011,  1060,  4881,   999,   102,     0,     0]])
    attention_mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    token_type_ids: tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    解码回文本: [CLS] hello, how are you today? i am grok, built by xai! [SEP]
    分词结果: ['[CLS]', 'hello', ',', 'how', 'are', 'you', 'today', '?', 'i', 'am', 'grok', ',', 'built', 'by', 'x', '#\#ai', '!', '[SEP]', '[PAD]', '[PAD]']
    
    批量输入结果:
    input_ids: tensor([[ 101, 7592, 2088,  999,  102,    0,    0,    0,    0,    0],
            [ 101, 1045, 2572, 20057, 1012,  102,    0,    0,    0,    0]])
    attention_mask: tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    ```
    
    **输入参数**
    
    1. **from_pretrained() 参数**：
        - "bert-base-uncased": 指定预训练模型/checkpoint。
        - use_fast=True: 使用 Rust 实现的快速分词器，性能更好。
        - cache_dir: 指定缓存路径，避免重复下载。
        - add_prefix_space: 对某些模型（如 RoBERTa）有用，BERT 不需要。
    2. **tokenizer() 参数**：
        - text: 输入可以是单个字符串或字符串列表。
        - add_special_tokens: 添加模型特定的标记（如 [CLS] 和 [SEP]）。
        - max_length: 限制序列长度。
        - padding: 填充方式（"max_length" 填充到指定长度，"longest" 填充到批次中最长序列）。
        - truncation: 超过 max_length 时截断。
        - return_tensors: 指定返回类型（"pt" 为 PyTorch，"tf" 为 TensorFlow，None 为 Python 列表）。
        - return_attention_mask: 返回注意力掩码，用于区分有效 token 和填充。
        - return_token_type_ids: 返回 token 类型 ID，用于句子对任务。
    
    **输出内容**
    
    1. **input_ids**：
        - 将文本转换为 token ID 的序列，每个 ID 对应词汇表中的一个 token。
        - [CLS]（101）和 [SEP]（102）是特殊标记，[PAD]（0）是填充。
    2. **attention_mask**：
        - 二进制掩码，1 表示有效 token，0 表示填充。
        - 用于告诉模型哪些部分需要关注。
    3. **token_type_ids**：
        - 用于区分不同句子（在单句输入中通常全为 0）。
        - 在句子对任务中，第一个句子为 0，第二个句子为 1。
    4. **解码和分词**：
        - decode(): 将 input_ids 转换回可读文本。
        - convert_ids_to_tokens(): 显示具体的分词结果（如 "x" 和 "#\#ai" 是子词）。
    
    **批量输入**
    
    - 当输入是列表时，padding=True 会自动对齐序列长度，填充较短的句子。
    - max_length 限制仍然有效。
    
    **扩展功能**
    
    1. **保存 tokenizer**：python
        
        ```Python
        tokenizer.save_pretrained("./my_tokenizer")
        ```
        
    2. **处理句子对**：python
        
        ```Python
        encoded_pair = tokenizer("Hello!", "How are you?", return_tensors="pt")
        print(encoded_pair["token_type_ids"])  # 区分两个句子
        ```
        
    3. **自定义词汇表**：python
        
        ```Python
        tokenizer.add_tokens(["new_token"])
        ```
        

## Model

```Python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

1. **加载模型**：
    - 根据 checkpoint 的配置文件，自动推断模型类型（如 BertModel）。
    - 使用 AutoModel.from_pretrained(checkpoint) 加载预训练模型。
2. **输入数据**：
    - AutoModel 需要接收 input_ids（来自 tokenizer 的输出）以及其他可选输入（如 attention_mask）。
    - 输入通常是张量（PyTorch 或 TensorFlow）。
3. **输出**：
    - 模型返回一个包含隐藏状态（hidden states）和其他输出的对象（具体取决于模型类型和配置）。
    - 不同任务的 AutoModel 变体返回值不同，需根据任务选择合适的类。
    - 默认 AutoModel 不包含头部（head），仅返回隐藏状态。

|   |   |   |   |
|---|---|---|---|
|**变体**|**主要输出字段**|**输出形状示例（BERT）**|**典型任务**|
|AutoModel|last_hidden_state, pooler_output|[1, 10, 768], [1, 768]|特征提取|
|AutoModelForSequenceClassification|logits|[1, 2]|文本分类|
|AutoModelForTokenClassification|logits|[1, 10, 3]|NER、词性标注|
|AutoModelForQuestionAnswering|start_logits, end_logits|[1, 10], [1, 10]|问答|
|AutoModelForCausalLM|logits|[1, 10, 30522]|文本生成（非 BERT）|
|AutoModelForMaskedLM|logits|[1, 10, 30522]|掩码预测|
|AutoModelForSeq2SeqLM|logits, encoder_last_hidden_state|[1, 10, vocab_size], [1, 10, 768]|翻译、摘要（非 BERT）|

返回logit的话到分类还要经过  
  
  
`torch.nn.functional.softmax`

代码示例

```Python
import torch
from transformers import AutoTokenizer, AutoModel

# 1. 加载 tokenizer 和 model
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
model = AutoModel.from_pretrained(
    checkpoint,
    cache_dir="./cache",         # 缓存目录
    output_attentions=True,      # 返回注意力权重（可选）
    output_hidden_states=True,   # 返回所有层的隐藏状态（可选）
)

# 2. 输入文本
text = "Hello, how are you today? I am Grok, built by xAI!"

# 3. 使用 tokenizer 预处理文本
inputs = tokenizer(
    text,
    return_tensors="pt",         # 返回 PyTorch 张量
    max_length=20,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
)

# 4. 将输入传递给模型
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)

# 5. 输出结果
print("原始文本:", text)
print("\nModel 输出:")
print("最后一层隐藏状态 (last_hidden_state):", outputs.last_hidden_state.shape)
print("池化输出 (pooler_output):", outputs.pooler_output.shape)
print("所有隐藏状态数量:", len(outputs.hidden_states))
print("注意力权重数量:", len(outputs.attentions))

# 6. 批量输入示例
batch_text = ["Hello world!", "I am Grok."]
batch_inputs = tokenizer(
    batch_text,
    padding=True,
    truncation=True,
    max_length=10,
    return_tensors="pt"
)
batch_outputs = model(**batch_inputs)  # 使用 ** 解包字典输入
print("\n批量输入结果:")
print("最后一层隐藏状态:", batch_outputs.last_hidden_state.shape)

# 7. 移动到 GPU（可选）
if torch.cuda.is_available():
    model = model.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model(**inputs)
    print("\nGPU 上最后一层隐藏状态:", outputs.last_hidden_state.shape)
```

```Plain
原始文本: Hello, how are you today? I am Grok, built by xAI!

Model 输出:
最后一层隐藏状态 (last_hidden_state): torch.Size([1, 20, 768])
池化输出 (pooler_output): torch.Size([1, 768])
所有隐藏状态数量: 13
注意力权重数量: 12

批量输入结果:
最后一层隐藏状态: torch.Size([2, 10, 768])

GPU 上最后一层隐藏状态: torch.Size([1, 20, 768])
```

**输入参数**

1. **from_pretrained() 参数**：
    - checkpoint: 指定模型名称（如 "bert-base-uncased"）。
    - cache_dir: 指定缓存路径。
    - output_attentions: 是否返回注意力权重。
    - output_hidden_states: 是否返回所有层的隐藏状态。
2. **model() 参数**：
    - input_ids: 来自 tokenizer 的 token ID 张量。
    - attention_mask: 注意力掩码，区分有效 token 和填充。
    - 可选：token_type_ids（句子对任务）、position_ids 等。

**输出内容**

1. **last_hidden_state**：
    - 形状 [batch_size, sequence_length, hidden_size]。
    - 表示最后一层的隐藏状态，每个 token 有一个 768 维向量（对于 BERT-base）。
2. **pooler_output**：
    - 形状 [batch_size, hidden_size]。
    - [CLS] token 的隐藏状态经过池化层（线性 + tanh）后的输出，常用于分类任务。
3. **hidden_states**（可选）：
    - 一个元组，包含所有层的隐藏状态（包括嵌入层和 12 个 Transformer 层，共 13 个）。
    - 每个形状为 [batch_size, sequence_length, hidden_size]。
4. **attentions**（可选）：
    - 一个元组，包含每层的注意力权重（12 层）。
    - 每个形状为 [batch_size, num_heads, sequence_length, sequence_length]，num_heads=12。

开启 output_hidden_states 或 output_attentions 会显著增加内存使用。

**批量输入**

- 输入多个句子时，padding=True 确保长度对齐。
- 输出张量的 batch_size 变为输入样本数（这里是 2）。

**GPU 支持**

- 使用 .to("cuda") 将模型和输入移动到 GPU。
- 确保 PyTorch 和模型输入都在同一设备上。

**保存模型**：

```Python
model.save_pretrained("./my_model")
```

**冻结参数**：python

```Python
for param in model.base_model.parameters():
    param.requires_grad = False  # 冻结底层参数
```

# 微调

## DataPreproccessing

  

```Python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

```Python
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

下载数据集，默认路径_~/.cache/huggingface/datasets。可以通过HF_HOME环境变量修改_

1. **加载数据集**：
    - 使用 load_dataset() 从 Hub 或本地路径加载数据集。
    - 数据集通常分为 train、validation 和 test 三个拆分（split）。
2. **访问数据**：
    - 数据以类似字典的形式存储，支持切片、迭代和列访问。
    - 每个样本是一个字典，键对应特征（如 text、label）。
3. **处理数据**：
    - 支持映射（map）、过滤（filter）、分片（shard）等操作。
    - 可以无缝与 transformers 的 tokenizer 结合。
4. **输出**：
    - 数据可以直接访问，或转换为 PyTorch/TensorFlow 张量。

代码示例

```Python
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. 加载数据集
dataset = load_dataset(
    "emotion",                  # 数据集名称
    split="train",              # 指定拆分（可选：train/validation/test）
    cache_dir="./cache",        # 缓存目录
    download_mode="reuse_dataset_if_exists",  # 复用已下载的数据
)

# 2. 查看数据集基本信息
print("数据集信息:", dataset)
print("特征:", dataset.features)
print("样本数量:", len(dataset))
print("第一个样本:", dataset[0])

# 3. 访问和切片数据
sample_text = dataset["text"][:3]  # 前 3 个样本的文本
sample_labels = dataset["label"][:3]  # 前 3 个样本的标签
print("\n前 3 个样本文本:", sample_text)
print("前 3 个样本标签:", sample_labels)

# 4. 结合 tokenizer 预处理
# 该函数接收一个字典（与 dataset 的项类似）并返回一个包含 id(input_ids) ，(attention_mask) 和 token_type_ids 键的新字典
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

# 应用映射函数
# Datasets 库中的数据集是以 Apache Arrow 格式存储在磁盘上的，因此你只需将接下来要用的数据加载在内存中，而不是加载整个数据集
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,              # 批量处理
    remove_columns=["text"],   # 删除原始文本列
    desc="Tokenizing dataset"  # 进度条描述
)
print("\n预处理后的第一个样本:", tokenized_dataset[0])

# 5. 过滤数据（示例：只保留标签为 0 的样本）
filtered_dataset = dataset.filter(
    lambda example: example["label"] == 0,
    desc="Filtering for label 0"
)
print("\n过滤后样本数量:", len(filtered_dataset))
print("过滤后第一个样本:", filtered_dataset[0])

# 6. 转换为 PyTorch 格式
tokenized_dataset.set_format(
    type="torch",              # 转换为 PyTorch 张量
    columns=["input_ids", "attention_mask", "label"]
)
print("\n转换为 PyTorch 后的第一个样本:", tokenized_dataset[0])

# 7. 批量访问（示例：取前 2 个样本）
batch = tokenized_dataset[:2]
print("\n批量数据 (input_ids):", batch["input_ids"].shape)
print("批量数据 (labels):", batch["label"])
```

```Plain
数据集信息: Dataset({
    features: ['text', 'label'],
    num_rows: 16000
})
特征: {'text': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=6, names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None)}
样本数量: 16000
第一个样本: {'text': 'i didnt feel humiliated', 'label': 0}

前 3 个样本文本: ['i didnt feel humiliated', 'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake', 'i feel grouchy']
前 3 个样本标签: [0, 1, 3]

预处理后的第一个样本: {'label': 0, 'input_ids': [101, 1045, 2134, 2102, 2514, 26608, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

过滤后样本数量: 5362
过滤后第一个样本: {'text': 'i didnt feel humiliated', 'label': 0}

转换为 PyTorch 后的第一个样本: {'label': tensor(0), 'input_ids': tensor([ 101, 1045, 2134, 2102, 2514, 26608,  102,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}

批量数据 (input_ids): torch.Size([2, 20])
批量数据 (labels): tensor([0, 1])
```

**输入参数**

1. **load_dataset() 参数**：
    - "emotion": 数据集名称。
    - split: 指定加载的拆分（可选，默认加载所有拆分）。
    - cache_dir: 自定义缓存路径。
    - download_mode: 控制下载行为（"reuse_dataset_if_exists" 复用已下载数据）。
2. **map() 参数**：
    - function: 自定义处理函数（如分词）。
    - batched=True: 批量处理，提高效率。
    - remove_columns: 删除不需要的列（如原始 text）。
3. **filter() 参数**：
    - function: 过滤条件（返回布尔值）。
    - desc: 进度条描述。
4. **set_format() 参数**：
    - type: 输出格式（"torch", "tensorflow", "numpy", 或 None）。
    - columns: 指定转换的列。

**输出内容**

1. **原始数据集**：
    - 一个 Dataset 对象，包含特征和样本。
    - 访问方式类似字典或列表。
2. **预处理后数据集**：
    - 包含 tokenizer 输出（如 input_ids, attention_mask）。
    - 移除原始列后更适合模型输入。
3. **过滤后数据集**：
    - 仅保留满足条件的样本。
4. **PyTorch 格式**：
    - 数据转换为张量，直接可用于训练。

**批量访问**

- 切片（如 [:2]）返回一个字典，键对应列名，值是张量。
- 形状反映批量大小（batch_size）和序列长度。

**保存数据集**：python

```Python
tokenized_dataset.save_to_disk("./my_dataset")
```

**动态填充**

将所有示例填充到该 batch 中最长元素的长度

```Python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch = data_collator(samples)
```

  

## Trainer API

准备数据示例

```Python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

定义`TrainingArguments` 类，包含超参数

```Python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer") 
\#用于保存训练后的模型以及训练过程中的 checkpoint 的目录。
\#对于其余的参数你可以保留默认值，这对于简单的微调应该效果就很好了。
from transformers import AutoModelForSequenceClassification
# Bert没有再句子分类的数据集上训练过，会报错，意思是已有的输出头没有使用，新的输出头是随机初始化的
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

定义一个 `Trainer` 把到目前为止构建的所有对象 —— `model` ，`training_args`，训练和验证数据集， `data_collator` 和 `tokenizer` 传递给 `Trainer`

```Python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

**评估**

```Python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
# predict() 方法的输出一个带有三个字段的命名元组: predictions label_ids 和 metrics
# metrics 字段将只包含所传递的数据集的损失,以及一些时间指标(总共花费的时间和平均预测时间)
import numpy as np
# predictions.prediction (batch,num_class)
preds = np.argmax(predictions.predictions, axis=-1)
import evaluate
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

最后把所有东西打包在一起，我们就得到了 `compute_metrics()` 函数  
  
该函数必须接收一个   
`EvalPrediction` 对象（它是一个带有 `predictions` 和 `label_ids` 字段的参数元组），并将返回一个字符串映射到浮点数的字典（字符串是返回的指标名称，而浮点数是其值）

```Python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

```Python
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}
```

为了查看模型在每个训练周期结束时的好坏，下面是我们如何使用 `compute_metrics()` 函数定义一个新的 `Trainer`

```Python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
# 每个epoch评估一次
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

## Training Process

```Python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

在正式开始编写我们的训练循环之前，我们需要定义一些对象。首先是我们将用于迭代 batch 的数据加载器。但在定义这些数据加载器之前，我们需要对我们的 `tokenized_datasets` 进行一些后处理，以自己实现一些 Trainer 自动为我们处理的内容。具体来说，我们需要：

- 删除与模型不需要的列（如 `sentence1` 和 `sentence2` 列）。
- 将列名 `label` 重命名为 `labels` （因为模型默认的输入是 `labels` ）。
- 设置数据集的格式，使其返回 PyTorch 张量而不是列表。

```Python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```

```Python
["attention_mask", "input_ids", "labels", "token_type_ids"]
```

定义数据加载器DataLoader

|   |   |   |
|---|---|---|
|**参数**|**类型**|**含义**|
|dataset|Dataset|要加载的数据集对象，必须是 torch.utils.data.Dataset 的实例。|
|batch_size|int|每个批次包含的样本数，默认是 1。|
|shuffle|bool|是否在每个 epoch 开始时打乱数据，默认是 False。|
|num_workers|int|使用多少个子进程加载数据，默认是 0（主进程加载）。|
|drop_last|bool|如果数据集大小不能整除 batch_size，是否丢弃最后一个不完整的批次，默认是 False。|
|collate_fn|callable|自定义函数，用于将多个样本组合成一个批次，默认会自动转为张量。|
|sampler|Sampler|自定义采样器，控制数据加载顺序（与 shuffle=True 互斥）。|

```Python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

已完成数据预处理。实例化模型

```Python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

优化器和学习率调度器

```Python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

```Python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

GPU

```Python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```

tqdm 是一个 Python 库，全称是 **"taqadum"**（阿拉伯语中的“进度”），用于在循环或迭代过程中添加进度条

|   |   |   |
|---|---|---|
|**参数**|**类型**|**含义**|
|iterable|可迭代对象|要包装的迭代对象（如 range(100)、列表等）。|
|desc|str|进度条前缀描述，例如 "Training"。|
|total|int|总迭代次数（可选，若 iterable 无长度则需手动指定）。|
|leave|bool|完成后是否保留进度条，默认 True。|
|unit|str|迭代单位，默认 "it"（iterations），可设为 "epoch"、bytes 等。|
|ncols|int|进度条宽度（字符数），默认自适应终端宽度。|
|mininterval|float|更新进度条的最小时间间隔（秒），默认 0.1。|

```Python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

评估

```Python
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

使用Accelerator分布式训练

```Python
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

把这个放在 `train.py` 文件中，可以让它在任何类型的分布式设置上运行。要在分布式设置中试用它，请运行以下命令：

```Plain
accelerate config
```

这将询问你几个配置的问题并将你的回答保存到此命令使用的配置文件中：

```Plain
accelerate launch train.py
```

这将启动分布式训练

这将启动分布式训练。如果你想在 Notebook 中尝试此操作（例如，在 Colab 上使用 TPU 进行测试），只需将代码粘贴到一个 `training_function()` 函数中，并在最后一个单元格中运行：

```Python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```