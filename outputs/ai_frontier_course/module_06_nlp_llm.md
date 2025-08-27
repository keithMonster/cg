# 模块六：自然语言处理与大语言模型

## 课程信息
- **模块编号**：Module 06
- **模块名称**：自然语言处理与大语言模型
- **学习目标**：掌握NLP核心技术、Transformer架构和大语言模型的原理与应用
- **学时安排**：理论课程 16学时，实践课程 14学时

## 第一章：自然语言处理基础

### 1.1 语言学基础

#### 语言的层次结构

**音韵学（Phonology）**
- 语音单位：音素、音节
- 语音规则：音变、重音
- 计算应用：语音识别、语音合成

**形态学（Morphology）**
- 词素分析：词根、词缀
- 词形变化：屈折、派生
- 计算处理：词干提取、词形还原

**句法学（Syntax）**
- 短语结构：名词短语、动词短语
- 句法树：成分分析、依存关系
- 文法理论：上下文无关文法、依存文法

**语义学（Semantics）**
- 词汇语义：同义词、反义词、上下位关系
- 组合语义：语义角色、事件结构
- 形式语义：逻辑表示、真值条件

**语用学（Pragmatics）**
- 语境依赖：指代消解、省略恢复
- 言语行为：断言、疑问、命令
- 会话含义：合作原则、礼貌原则

#### 计算语言学基础

**形式语言理论**

*Chomsky层次*
```
类型0：无限制文法（图灵机等价）
类型1：上下文相关文法（线性有界自动机）
类型2：上下文无关文法（下推自动机）
类型3：正则文法（有限状态自动机）
```

*上下文无关文法（CFG）*
```
G = (V, Σ, R, S)
V：非终结符集合
Σ：终结符集合
R：产生式规则
S：起始符号

例子：
S → NP VP
NP → Det N | N
VP → V NP | V
Det → the | a
N → cat | dog
V → chases | sees
```

**概率语言模型**

*n-gram模型*
```
P(w₁w₂...wₙ) = ∏ᵢ₌₁ⁿ P(wᵢ|w₁...wᵢ₋₁)

近似（马尔可夫假设）：
Bigram: P(wᵢ|w₁...wᵢ₋₁) ≈ P(wᵢ|wᵢ₋₁)
Trigram: P(wᵢ|w₁...wᵢ₋₁) ≈ P(wᵢ|wᵢ₋₂wᵢ₋₁)
```

*平滑技术*
```
Laplace平滑:
P(wᵢ|wᵢ₋₁) = (C(wᵢ₋₁wᵢ) + 1) / (C(wᵢ₋₁) + V)

Good-Turing平滑:
C* = (C + 1) × N_{C+1} / N_C

Kneser-Ney平滑:
P_{KN}(wᵢ|wᵢ₋₁) = max(C(wᵢ₋₁wᵢ) - D, 0) / C(wᵢ₋₁) + λ(wᵢ₋₁)P_{KN}(wᵢ)
```

### 1.2 文本预处理

#### 分词与标准化

**英文分词**
- 空格分割
- 标点符号处理
- 缩写展开
- 大小写标准化

**中文分词**

*基于词典的方法*
```
最大匹配算法（MM）:
1. 从左到右扫描
2. 每次匹配最长词
3. 处理歧义

双向最大匹配（Bi-MM）:
1. 正向最大匹配
2. 反向最大匹配
3. 比较结果选择
```

*基于统计的方法*
- 隐马尔可夫模型（HMM）
- 条件随机场（CRF）
- 神经网络方法

*基于字符的标注*
```
BIES标注体系:
B：词首字符
I：词中字符
E：词尾字符
S：单字词

例子："自然语言处理"
B-I-E-B-I-E
```

#### 词性标注

**标注集合**

*Penn Treebank标注*
```
名词：NN, NNS, NNP, NNPS
动词：VB, VBD, VBG, VBN, VBP, VBZ
形容词：JJ, JJR, JJS
副词：RB, RBR, RBS
介词：IN
限定词：DT
代词：PRP, PRP$
```

*中文词性标注*
```
名词：n, nr, ns, nt, nz
动词：v, vd, vn, vshi, vyou
形容词：a, ad, an, ag
副词：d, dg, dl
介词：p, pba, pbei
```

**标注算法**

*隐马尔可夫模型*
```
P(T|W) = P(W|T)P(T) / P(W)

其中：
P(T) = ∏ᵢ P(tᵢ|tᵢ₋₁)  # 转移概率
P(W|T) = ∏ᵢ P(wᵢ|tᵢ)  # 发射概率

Viterbi算法求解最优路径
```

*条件随机场*
```
P(T|W) = (1/Z(W)) exp(∑ᵢ ∑ⱼ λⱼfⱼ(tᵢ₋₁, tᵢ, W, i))

特征函数：
f₁(tᵢ₋₁, tᵢ, W, i) = 1 if tᵢ₋₁=NN and tᵢ=VB and wᵢ="run"
f₂(tᵢ₋₁, tᵢ, W, i) = 1 if tᵢ=JJ and wᵢ ends with "-ly"
```

#### 命名实体识别

**实体类型**
- 人名（PERSON）
- 地名（LOCATION）
- 机构名（ORGANIZATION）
- 时间（TIME）
- 数量（QUANTITY）
- 货币（MONEY）

**识别方法**

*基于规则*
```
正则表达式：
电话号码：\d{3}-\d{3}-\d{4}
邮箱地址：[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
日期：\d{4}-\d{2}-\d{2}
```

*基于机器学习*
- 特征工程：词汇特征、上下文特征、字符特征
- 序列标注：BIO标注、BIOES标注
- 模型选择：CRF、LSTM-CRF、BERT-CRF

*BIO标注示例*
```
句子："苹果公司的CEO蒂姆·库克访问了中国"
标注：
苹果 B-ORG
公司 I-ORG
的   O
CEO  O
蒂姆 B-PER
·    I-PER
库克 I-PER
访问 O
了   O
中国 B-LOC
```

### 1.3 句法分析

#### 成分句法分析

**上下文无关文法解析**

*CYK算法*
```
动态规划算法，时间复杂度O(n³|G|)

算法步骤：
1. 初始化：填充长度为1的子串
2. 递推：计算长度为k的子串
3. 回溯：构建解析树

for length = 1 to n:
    for i = 1 to n-length+1:
        j = i + length - 1
        for k = i to j-1:
            for each rule A → BC:
                if B ∈ table[i][k] and C ∈ table[k+1][j]:
                    add A to table[i][j]
```

*Earley算法*
```
自顶向下解析，处理左递归

状态表示：[A → α·β, i, j]
A → α·β：产生式规则
i：起始位置
j：当前位置

操作：
1. Predictor：预测新规则
2. Scanner：匹配终结符
3. Completer：完成规则
```

**概率上下文无关文法（PCFG）**

*文法定义*
```
G = (V, Σ, R, S, P)
P：规则概率

约束：∑_{A→α} P(A → α) = 1

句子概率：
P(T) = ∏_{A→α∈T} P(A → α)
```

*参数估计*
```
最大似然估计：
P(A → α) = Count(A → α) / Count(A)

EM算法（Inside-Outside）：
处理无标注数据
```

#### 依存句法分析

**依存关系**

*基本概念*
- 核心词（Head）
- 修饰词（Dependent）
- 依存弧（Dependency Arc）
- 依存标签（Relation Label）

*依存关系类型*
```
主谓关系（nsubj）：主语-谓语
动宾关系（dobj）：动词-宾语
定中关系（amod）：形容词-名词
状中关系（advmod）：副词-动词
介宾关系（prep）：介词-宾语
```

**解析算法**

*基于转移的解析*
```
Arc-Standard系统：
配置：(σ, β, A)
σ：栈
β：缓冲区
A：弧集合

转移操作：
1. SHIFT：将缓冲区首词移入栈
2. LEFT-ARC(r)：在栈顶两词间建立左弧
3. RIGHT-ARC(r)：在栈顶两词间建立右弧
```

*基于图的解析*
```
最大生成树算法（MST）：
1. 构建完全图
2. 边权重为依存概率
3. 寻找最大生成树
4. 确保树的约束

Chu-Liu/Edmonds算法：
处理有向图的最大生成树
```

## 第二章：深度学习在NLP中的应用

### 2.1 词向量表示

#### 传统词表示

**One-hot编码**
```
词汇表：{"cat", "dog", "bird"}

cat:  [1, 0, 0]
dog:  [0, 1, 0]
bird: [0, 0, 1]

问题：
1. 维度灾难
2. 无法表示语义相似性
3. 稀疏表示
```

**词袋模型（Bag of Words）**
```
文档表示：词频向量

文档1："cat sits on mat"
文档2："dog runs in park"

词汇表：["cat", "sits", "on", "mat", "dog", "runs", "in", "park"]
文档1：[1, 1, 1, 1, 0, 0, 0, 0]
文档2：[0, 0, 0, 0, 1, 1, 1, 1]
```

**TF-IDF**
```
TF(t,d) = Count(t,d) / |d|
IDF(t) = log(|D| / |{d: t ∈ d}|)
TF-IDF(t,d) = TF(t,d) × IDF(t)

特点：
- 降低高频词权重
- 提升区分性词权重
- 广泛用于信息检索
```

#### 分布式词表示

**Word2Vec**

*Skip-gram模型*
```
目标：给定中心词预测上下文词

P(wₒ|wc) = exp(vₒᵀvc) / ∑w exp(vwᵀvc)

其中：
vc：中心词向量
vₒ：上下文词向量

目标函数：
J = -∑c ∑o log P(wo|wc)
```

*CBOW模型*
```
目标：给定上下文词预测中心词

P(wc|Context) = exp(vcᵀh) / ∑w exp(vwᵀh)

其中：
h = (1/C)∑c∈Context vc
```

*优化技术*
```
分层Softmax：
- 使用Huffman树
- 时间复杂度：O(log|V|)

负采样：
- 采样负例词
- 简化计算
- 目标函数：
J = log σ(vₒᵀvc) + ∑k E[log σ(-vkᵀvc)]
```

**GloVe（Global Vectors）**

*共现矩阵*
```
Xᵢⱼ：词i在词j上下文中出现次数

目标函数：
J = ∑ᵢ,ⱼ f(Xᵢⱼ)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²

权重函数：
f(x) = (x/xₘₐₓ)^α if x < xₘₐₓ
f(x) = 1 otherwise
```

*优势*
- 结合全局统计信息
- 训练效率高
- 性能稳定

**FastText**

*子词信息*
```
词表示 = 词向量 + 子词向量和

例如："where" = <wh + whe + her + ere + re>

优势：
1. 处理未登录词
2. 利用形态信息
3. 适合形态丰富语言
```

### 2.2 循环神经网络

#### 基础RNN

**网络结构**
```
hₜ = tanh(Whₕhₜ₋₁ + Wₓₕxₜ + bₕ)
yₜ = Wyₕhₜ + by

其中：
hₜ：隐状态
xₜ：输入
yₜ：输出
W, b：参数
```

**训练算法**

*通过时间反向传播（BPTT）*
```
损失函数：
L = ∑ₜ Lₜ(yₜ, ŷₜ)

梯度计算：
∂L/∂Wyₕ = ∑ₜ ∂Lₜ/∂yₜ ∂yₜ/∂Wyₕ
∂L/∂Whₕ = ∑ₜ ∂Lₜ/∂hₜ ∂hₜ/∂Whₕ

其中：
∂hₜ/∂hₜ₋₁ = diag(1-tanh²(·)) × Whₕ
```

**梯度问题**

*梯度消失*
```
∂hₜ/∂hₜ₋ₖ = ∏ᵢ₌₁ᵏ ∂hₜ₋ᵢ₊₁/∂hₜ₋ᵢ

当 ||∂hₜ₊₁/∂hₜ|| < 1 时，梯度指数衰减
```

*梯度爆炸*
```
当 ||∂hₜ₊₁/∂hₜ|| > 1 时，梯度指数增长

解决方案：
1. 梯度裁剪
2. 权重正则化
3. 更好的初始化
```

#### LSTM

**网络结构**

*门控机制*
```
遗忘门：fₜ = σ(Wf[hₜ₋₁, xₜ] + bf)
输入门：iₜ = σ(Wi[hₜ₋₁, xₜ] + bi)
候选值：C̃ₜ = tanh(WC[hₜ₋₁, xₜ] + bC)
细胞状态：Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
输出门：oₜ = σ(Wo[hₜ₋₁, xₜ] + bo)
隐状态：hₜ = oₜ * tanh(Cₜ)
```

*关键思想*
- 细胞状态：长期记忆
- 隐状态：短期记忆
- 门控：信息流控制

**变体**

*GRU（Gated Recurrent Unit）*
```
重置门：rₜ = σ(Wr[hₜ₋₁, xₜ])
更新门：zₜ = σ(Wz[hₜ₋₁, xₜ])
候选状态：h̃ₜ = tanh(W[rₜ * hₜ₋₁, xₜ])
隐状态：hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ

优势：
- 参数更少
- 训练更快
- 性能相当
```

#### 双向RNN

**网络结构**
```
前向：h⃗ₜ = RNN(h⃗ₜ₋₁, xₜ)
后向：h⃖ₜ = RNN(h⃖ₜ₊₁, xₜ)
输出：hₜ = [h⃗ₜ; h⃖ₜ]

优势：
- 利用双向上下文
- 更好的表示能力
- 适合序列标注任务
```

### 2.3 注意力机制

#### 基础注意力

**动机**
- 解决长序列信息瓶颈
- 动态关注相关信息
- 提供可解释性

**计算过程**
```
1. 计算注意力分数：
eᵢⱼ = a(sᵢ₋₁, hⱼ)

2. 归一化（Softmax）：
αᵢⱼ = exp(eᵢⱼ) / ∑ₖ exp(eᵢₖ)

3. 加权求和：
cᵢ = ∑ⱼ αᵢⱼhⱼ

4. 输出计算：
sᵢ = f(sᵢ₋₁, yᵢ₋₁, cᵢ)
```

**注意力函数**

*加性注意力*
```
a(s, h) = vᵀ tanh(Ws s + Wh h)
```

*乘性注意力*
```
a(s, h) = sᵀ h  (点积)
a(s, h) = sᵀ W h  (一般乘性)
```

*缩放点积注意力*
```
Attention(Q, K, V) = softmax(QKᵀ/√dk)V

其中：
Q：查询矩阵
K：键矩阵
V：值矩阵
dk：键向量维度
```

#### 自注意力机制

**Self-Attention**
```
输入序列：X = [x₁, x₂, ..., xₙ]

Q = XWQ
K = XWK  
V = XWV

Attention(X) = softmax(QKᵀ/√dk)V

特点：
- 序列内部关系建模
- 并行计算
- 长距离依赖
```

**多头注意力**
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)WO

其中：
headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

优势：
- 多个表示子空间
- 不同类型关系
- 增强表达能力
```

## 第三章：Transformer架构

### 3.1 Transformer基础

#### 整体架构

**编码器-解码器结构**
```
编码器：
- N=6个相同层
- 每层包含：多头自注意力 + 前馈网络
- 残差连接和层归一化

解码器：
- N=6个相同层  
- 每层包含：掩码多头自注意力 + 编码器-解码器注意力 + 前馈网络
- 残差连接和层归一化
```

**位置编码**
```
正弦位置编码：
PE(pos, 2i) = sin(pos/10000^(2i/dmodel))
PE(pos, 2i+1) = cos(pos/10000^(2i/dmodel))

其中：
pos：位置
i：维度索引
dmodel：模型维度

特点：
- 相对位置信息
- 外推到更长序列
- 确定性编码
```

#### 关键组件

**层归一化**
```
LayerNorm(x) = γ ⊙ (x - μ)/σ + β

其中：
μ = (1/H)∑ᵢ xᵢ
σ² = (1/H)∑ᵢ (xᵢ - μ)²
γ, β：可学习参数

位置：
- Pre-LN：归一化在子层之前
- Post-LN：归一化在子层之后
```

**前馈网络**
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

特点：
- 两层线性变换
- ReLU激活函数
- 位置独立处理
- 增加非线性
```

**残差连接**
```
Output = LayerNorm(x + Sublayer(x))

优势：
- 缓解梯度消失
- 加速训练收敛
- 允许更深网络
```

### 3.2 训练与优化

#### 训练策略

**教师强制（Teacher Forcing）**
```
训练时：使用真实目标序列作为解码器输入
推理时：使用模型预测作为下一步输入

问题：训练推理不一致（Exposure Bias）
```

**标签平滑**
```
原始标签：y = [0, 0, 1, 0, 0]
平滑标签：y' = (1-ε)y + ε/K

其中：
ε：平滑参数
K：类别数

作用：
- 防止过拟合
- 提高泛化能力
- 增强鲁棒性
```

**学习率调度**
```
Warmup + Cosine Decay：
lr = base_lr × min(step^(-0.5), step × warmup_steps^(-1.5))

阶段：
1. Warmup：线性增长
2. Decay：余弦衰减
```

#### 优化技巧

**梯度累积**
```
# 模拟大批量训练
for i in range(accumulation_steps):
    loss = model(batch[i]) / accumulation_steps
    loss.backward()

optimizer.step()
optimizer.zero_grad()
```

**混合精度训练**
```
# 使用FP16加速训练
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3.3 Transformer变体

#### 编码器模型

**BERT（Bidirectional Encoder Representations from Transformers）**

*预训练任务*
```
1. 掩码语言模型（MLM）：
   输入："The [MASK] is running"
   目标：预测被掩码的词

2. 下一句预测（NSP）：
   输入：句子对
   目标：判断是否连续
```

*模型结构*
```
BERT-Base：
- 12层Transformer编码器
- 768维隐状态
- 12个注意力头
- 110M参数

BERT-Large：
- 24层Transformer编码器
- 1024维隐状态
- 16个注意力头
- 340M参数
```

**RoBERTa**

*改进*
- 移除NSP任务
- 动态掩码
- 更大批量
- 更多数据
- 更长训练

**ELECTRA**

*替换检测*
```
生成器：小型BERT，生成替换词
判别器：检测哪些词被替换

优势：
- 所有位置都有学习信号
- 训练效率更高
- 小模型也能达到好效果
```

#### 解码器模型

**GPT系列**

*GPT-1*
```
结构：12层Transformer解码器
训练：无监督预训练 + 有监督微调
任务：语言建模
```

*GPT-2*
```
改进：
- 更大模型（1.5B参数）
- 更多数据
- 零样本学习能力
- 任务条件化
```

*GPT-3*
```
规模：
- 175B参数
- 96层
- 12288维隐状态
- 96个注意力头

能力：
- Few-shot学习
- 上下文学习
- 涌现能力
```

#### 编码器-解码器模型

**T5（Text-to-Text Transfer Transformer）**

*统一框架*
```
所有任务转化为文本到文本：
翻译："translate English to German: Hello" → "Hallo"
分类："sentiment: I love this movie" → "positive"
问答："question: What is AI? context: ..." → "answer"
```

*预训练*
```
去噪自编码：
输入："Thank you <X> me to your party <Y> week."
目标："<X> for inviting <Y> last <Z>"
```

**BART**

*去噪预训练*
```
噪声类型：
1. 词删除
2. 词替换
3. 句子排列
4. 文档旋转
5. 文本填充
```

## 实践项目

### 项目一：文本分类系统

**数据预处理**
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 初始化
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 数据加载
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

**模型训练**
```python
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 训练循环
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
```

### 项目二：机器翻译系统

**Transformer实现**
```python
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.output_projection(output)
```

### 项目三：问答系统

**BERT问答**
```python
from transformers import BertForQuestionAnswering, BertTokenizer

class QASystem:
    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def answer_question(self, question, context):
        # 编码输入
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取答案位置
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # 解码答案
        input_ids = inputs['input_ids'][0]
        answer_tokens = input_ids[start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer

# 使用示例
qa_system = QASystem()
question = "What is the capital of France?"
context = "France is a country in Europe. The capital of France is Paris."
answer = qa_system.answer_question(question, context)
print(f"Answer: {answer}")
```

## 学习评估

### 理论评估
1. **语言学基础**：形态学、句法学、语义学理解
2. **模型原理**：RNN、LSTM、Transformer架构
3. **训练技巧**：优化策略、正则化方法
4. **应用场景**：不同任务的模型选择

### 实践评估
1. **编程实现**：从零实现Transformer
2. **模型微调**：预训练模型的下游任务适配
3. **性能优化**：训练加速和推理优化
4. **项目开发**：端到端NLP应用

### 综合评估
1. **技术报告**：深度技术分析和创新
2. **代码质量**：工程实践和代码规范
3. **系统设计**：架构设计和扩展性
4. **效果评估**：模型性能和实用性

## 延伸学习

### 前沿研究方向
1. **大语言模型**：GPT-4、Claude、ChatGPT
2. **多模态模型**：CLIP、DALL-E、GPT-4V
3. **高效训练**：LoRA、AdaLoRA、QLoRA
4. **推理优化**：量化、剪枝、蒸馏
5. **对齐技术**：RLHF、Constitutional AI

### 应用领域
1. **对话系统**：智能客服、虚拟助手
2. **内容生成**：文本创作、代码生成
3. **信息抽取**：知识图谱、关系抽取
4. **机器翻译**：多语言翻译、同声传译
5. **文档理解**：智能阅读、自动摘要

### 工具和资源
1. **深度学习框架**：PyTorch、TensorFlow
2. **NLP库**：Transformers、spaCy、NLTK
3. **预训练模型**：Hugging Face Model Hub
4. **数据集**：GLUE、SuperGLUE、SQuAD
5. **评估工具**：BLEU、ROUGE、BERTScore

## 总结

自然语言处理是人工智能的核心领域，从传统的统计方法到现代的深度学习技术，特别是Transformer架构的出现，彻底改变了NLP的发展轨迹。大语言模型的兴起更是开启了通用人工智能的新篇章。

通过本模块的学习，学生将掌握：
1. NLP的基础理论和核心概念
2. 深度学习在NLP中的应用
3. Transformer架构的设计原理
4. 大语言模型的训练和应用
5. 实际NLP系统的开发技能

这些知识将为学生在NLP领域的研究和应用提供坚实的理论基础和实践能力。