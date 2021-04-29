###任务目标
根据问题从多个文档中找到正确答案：
比如：
```python
# 文档id
docid:jwejrew023kj # 政策文档1
    # 每个政策文档中的段落切开赋予独立id
    paramid:jwejrew023kj_0, jwejrew023kj_1, jwejrew023kj_2
    # 每个段落id对应的内容
    text:'基于疫情需要，广西在网上开通捐赠渠道，微信注册即可登录'
question: 广西网上捐赠渠道如何登录？
answer: 微信登录
```

### 数据集
https://www.datafountain.cn/competitions/424/datasets
![image](https://user-images.githubusercontent.com/68730894/115325626-8ad4eb80-a1be-11eb-81c2-e56ac9c0488f.png)


### 核心技术
BM25从多个文档中召回top-k个包含正确答案，全局负采样和局部负采样构建负样本；loss函数权重调节；设置标签不均衡来加大正样本学习难度；

### 特征工程
* BM25算法召回top-k个含有正确答案的段落

由于是跨文档找正确答案，因此我们需要在所有段落中找最有可能的含有正确答案的算法。该数据集的数据有8000多个，如果只使用BERT召回，时空复杂度很高，BM25是工业界常用的无监督召回策略。我们通过BM25算法召回与问题相关的50个文档，在训练集的召回正确率达到了0.9以上，因此我们确定召回文档数为50。

BM25大致原理：对Query进行语素解析，生成语素qi（对于汉语来说，通常是指分词后的词）；然后，对于每个搜索结果D，计算每个语素qi与D的相关性得分，最后，将qi相对于D的相关性得分进行加权求和，从而得到Query与D的相关性得分

此外，8000多个文档是有重复的，在处理的过程中删除重复文档，降低时空复杂度。

* 答案修正

找到答案字更多的段落作为正确答案所在段落，相当于匹配覆盖答案字数多的段落

如果是单文档阅读理解，可以不使用BM25算法召回top-k段落，直接把数据喂给BERT召回正确答案的段落就行

## 构造正负样本

召回的top-k个段落含有正确答案的标记为1，剩下的不含正确答案标记为0，参考Airbnb负样本构建方法，作者还随机选择一部分非top-k范围内的样本作为负样本，大约提升0.5个点

参考Airbnb推荐系统，正样本：负样本 = 1:10 或 1:5

测试集构建(一个问题对应K个候选段落)

测试集与训练集的候选段落是一样的

### 切分文本：BERT输入长度限制为512
为了尽可能保留切割段落的信息完整性，我们参考ROUGE-L让含有答案的段落中答案不被切为2个段落。

### 数据构造问答对的格式
input <问题·文档>, output<是否为相关文档，答案>

```python
start 表示答案开始的位置，
context是得到答案的文本answer
score =1 是 表示正样本，score=0是负样本

question_id    query                start             context                    answer  score
# 正样本
aj32232312a   广西晚上捐赠渠道如何登录？  18    基于疫情需要，广西在网上开通捐赠渠道，微信登录  微信登录     1
# 局部负采样：BM25召回top-k范围内的负样本
aj32232312a   广西晚上捐赠渠道如何登录？  -1    基于疫情需要，山东在网上开通捐赠渠道，微信登录   QQ登录     0
# 全局负采样：非top-k范围内的负样本
aj32232312a   广西晚上捐赠渠道如何登录？  -1    除夕期间，昆明市长向昆明市民拜年             微信登录     0
```

## 定义loss

找到正确答案所在段落的loss(分类任务)权重远大于判断正确答案位置的loss(回归任务)

训练过程中遇到loss很小，但是感觉模型还是欠拟合的情况。模拟人工阅读：先定位答案段落再找答案位置。把loss简单相加调整为 0.99 * 是否有答案 + 0.01 * 答案起始位置

* Pooling：BERT 12层隐藏层的向量进行加权，与文本分类项目一致

Ganesh Jawahar等人[3]证明BERT每一层对文本的理解都不同，因此将BERT的十二层transformer生成的表示赋予一个权重,(下文称为BERT动态融合)，初始化公式为：

α_i =Dense_(unit=1) (represent_i)

ouput=Dense_(unit=512) （∑_(i=1)^nα_i · represent_i）

BERT的动态融合作为模型embedding的成绩会优于BERT最后一层向量作为模型的embedding，且收敛速度更快，epochs=3-4个就可以了

权重学习易造成高时空复杂度，我们还可以使用SumPooling、MeanPooling、MaxPooling等方法进行融合，选择层数偏7 8 9 10 11 12层


### 模型内部结构 Pooling
* 使用残差网络解决BERT下游任务模型退化问题

下游任务使用LSTM/GRU/Transformer(3选1) 等结构对 BERT动态融合结果 进行特征提取时，神经网络发生退化问题，模型没有欠拟合。我们都是基于BERT开发的模型，所以主体结构变化不大，主要修改下游任务。

原因是：BERT的12层向量融合完成很好的提取了特征，这种情况复杂的模型反而效果会减弱。这在推荐系统中很常见，特征工程之后用个逻辑回归LR就能解决问题，可能对于LR来说，它只需要发挥自己的记忆能力，把特征工程整理出来的情况都记录在自己的评分卡中，辅以查表和相法就可完成任务。

笔者设计的残差网络不受向量维度的限制，因为回到ResNet的核心，非线性激活函数的存在导致特征变化不可逆，因此造成模型退化的根本原因是非线性激活函数。因此F(x)= f(x) + x 可以理解为f(x)为非线性特征，x为线性特征，
遇到维度不相等，可以直接用`nn.Linear(), tf.keras.layers.Dense()`让维度一致。然后再对位相加即可。

* 多种下游任务模型：GRU、LSTM、Transformer、RNN+Transfromer、CNN
BERT下游结构任务的核心是输入一个矩阵，得到一个矩阵，对于Transformer来说，输出矩阵和输出矩阵的维度是一样的，对于RNN、LSTM来说，输入和输出矩阵维度可以不同，

### 结果
超越该赛题的最佳方案

### 可改进的点
BERT是无监督学习方式，该任务是文本分类，因此我们可以为BERT单独增加损失函数，大致的思路是 bert的输出的logits(标量)作为一部分loss，可以理解为该值带有bert的信息，

### 核心修改的代码：
* 正负样本构造
```python
"""pos"""
pos_text = []
for text in context:
    if text.count(answer):
        start = text.find(answer)
        doc_dict = {'q_id': qid, 'query': question, 'start': start, 'context': text, 'answer': answer,
                    'score': 1}
        pos_text.append(text)
        Doc_train.append(doc_dict)
"""neg"""
#分词
query = tokenization(question)
# 通过BM25计算匹配对
scores = bm25Model.get_scores(query) # 用question匹配答案
scores = np.array(scores)
# 局部负采样
sort_index = np.argsort(-scores)[:50]  # 500选三非正样本作负样本 # random.choice(scores, 5)
# paraid_list是训练集所有段落的id集合，因此需要对其进行采样
para_ids = [paraid_list[i] for i in sort_index] # 使用第一步召回的段落构建负样本
if dev_num: # 验证集不需要全局采样
    para_id = random.sample(para_ids, dev_num) # dev_num = 25个负样本

else: # train_data的正负样本构造按照正样本：负样本=1：5
    # para_id = random.sample(para_ids, len(pos_text))  # 从50选正样本个数做负样本 正样本：负样本=1:5或1:10
    # 召回段落负采样 + 全部段落负采样  正:负 = 1:5 或 1:10
    num_neg_text = len(pos_text) * 5
    para_id = random.sample(para_ids, num_neg_text) + random.sample(paraid_list, num_neg_text) # 两个列表相加不是对应的值相加，而是类似字符串拼接那样进行拼接的
para_train = generate_para(doc_train, id2para, bm25Model)  # 生成BM25模型，找出topk个候选段落, 正负样本由1：1改成1：5
para_dev = generate_para(doc_dev, id2para, bm25Model, dev_num=25)  # 生成BM25模型，找出topk个候选段落
para_train['score'] = 0
para_dev['score'] = 0
```

* MRC（回归任务）
```python
self.start_outputs = nn.Linear(config.hidden_size, 1) # 缩放成起始点位置
self.end_outputs = nn.Linear(config.hidden_size, 1)
```
* 是否包含答案（分类任务）
```python
self.task2 = nn.Linear(config.hidden_size, 1)
start_logits = self.start_outputs(feats).squeeze(-1)  # [batch, seq_len]
end_logits = self.end_outputs(feats).squeeze(-1)  # [batch, seq_len]
```

* Mask
```python
start_logits = self.mask_logits(start_logits, attention_mask)
end_logits = self.mask_logits(end_logits, attention_mask)
start_pre = torch.sigmoid(start_logits)  # batch x seq_len
end_pre = torch.sigmoid(end_logits)
```
* 任务二：是否有答案
```python
cls_logits = self.task2(pooled_output)  # batch_size,1
cls_pre = torch.sigmoid(cls_logits)
```
* 定义loss
为防止bert找不到正确答案，我们增加了含有正确答案段落的学习难度
```python
def imbalanced_qa_loss(probs, labels, inbalance_rate=None):
    if inbalance_rate != None:
        weight = labels * (inbalance_rate - 1) + 1
        loss_func = nn.BCELoss(weight=weight.float()).to(device)
        loss = loss_func(probs, labels.float())
    else:
        loss_func = nn.BCELoss().to(device)
        loss = loss_func(probs, labels.float())
    return loss
# MRC Loss
start_loss = imbalanced_qa_loss(start_pre, start_positions, inbalance_rate=10)
end_loss = imbalanced_qa_loss(end_pre, end_positions, inbalance_rate=10)
Mrc_loss = start_loss + end_loss

# 正确答案所在段落内，答案的位置
CLS_loss = nn.BCELoss()(cls_pre, cls_label.unsqueeze(-1).float())
total_loss = 0.01 * Mrc_loss + 0.99 * CLS_loss
loss.backward()
```

* LSTM的残差连接
```python
class ResidualWrapper4RNN(nn.Module):
    def __init__(self, model):
        super().__init__() # super(ResidualWrapper, self).__init__()
        self.model = model
    def forward(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)[0]  # params.model_type=='bigru' or 'bilstm'
        return inputs + delta

self.bilstm = ResidualWrapper4RNN(nn.Sequential(
            BiLSTM(self.num_labels, embedding_size=config.hidden_size, hidden_size=params.lstm_hidden,
                             num_layers=params.num_layers, dropout=params.drop_prob, with_ln=True)
                             ))

result = self.bilstm(bert_ouput)
```

## BM25算法原理
BM25是基于TFIDF算法的改进
TFIDF：

词频 (term frequency, TF) 指的是某一个给定的词语在该文件中出现的次数。
![tf](https://user-images.githubusercontent.com/68730894/116521169-33452700-a906-11eb-8392-87acbb2936e1.png)

逆向文件频率 (inverse document frequency, IDF) IDF的主要思想是：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。
![idf](https://user-images.githubusercontent.com/68730894/116521185-3809db00-a906-11eb-81f2-9e09ffc144bf.png)


bm25 是一种用来评价搜索词和文档之间相关性的算法，它是一种基于概率检索模型提出的算法。简单来说，我们有一个query和一批文档Ds，现在要计算query和每篇文档D之间的相关性分数，我们的做法是，先对query进行切分，得到单词$q_i$，然后单词的分数由3部分组成：

* query中每个单词t与文档的d之间相关性
* 单词t与query之间的相似性
* 每个单词的权重
其中 Q 表示一条query， q_i 表示query中的单词。d表示某个搜索文档。

BM25的一般公式![image](https://user-images.githubusercontent.com/68730894/116523008-596bc680-a908-11eb-8211-b4e4ac1edb5d.png)


W_i 表示单词权重，这里其实就是IDF

![image](https://user-images.githubusercontent.com/68730894/116522930-4822ba00-a908-11eb-9621-7b00bd762b61.png)


其中N表示索引中全部文档数，$df_i$为包含了$q_i$的文档的个数。依据IDF的作用，对于某个$q_i$，包含$q_i$的文档数越多，说明$q_i$重要性越小，或者区分度越低，IDF越小，因此IDF可以用来刻画$q_i$与文档的相似性。

### 单词与文档的相关性
BM25的设计依据一个重要的发现：词频和相关性之间的关系是非线性的，也就是说，每个词对于文档的相关性分数不会超过一个特定的阈值，当词出现的次数达到一个阈值后，其影响就不在线性增加了，而这个阈值会跟文档本身有关。因此，在刻画单词与文档相似性时，BM25是这样设计的：

![image](https://user-images.githubusercontent.com/68730894/116522687-0db91d00-a908-11eb-99b5-257b577d43dd.png)

其中，$tf_{td}$是单词t在文档d中的词频，$L_d$是文档d的长度，$L_{ave}$是所有文档的平均长度，变量$k_1$是一个正的参数，用来标准化文章词频的范围，当$k_1=0$，就是一个二元模型（binary model）（没有词频），一个更大的值对应使用更原始的词频信息。b是另一个可调参数（$0<b<1$），他是用决定使用文档长度来表示信息量的范围：当b为1，是完全使用文档长度来权衡词的权重，当b为0表示不使用文档长度。

### 单词与query的相关性
当query很长时，我们还需要刻画单词与query的之间的权重。对于短的query，这一项不是必须的。

![image](https://user-images.githubusercontent.com/68730894/116522766-232e4700-a908-11eb-8716-950863a4454f.png)

这里$tf_{tq}$表示单词t在query中的词频，$k_3$是一个可调正参数，来矫正query中的词频范围。




运行方案顺序
* preprocess/final_para_result.py
* utils.py
* train_fine_tune.py
* predict.py
