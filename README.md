### 任务目标
* 囊括NLP领域的最新paper中提到的模型, 作者开发该框架，对于新的paper，只需要看model.py就可以融入该框架，并且可以考虑迁移到推荐系统中使用
* 推荐系统中最重要的特征是ID类特征，它组成的用户行为句子特征依靠NLP的技术，NLP从最开始end2end到预训练模型提取特征的技术路线为推荐系统提高了可参考的技术路线，2021年推荐系统领域内的模型逐步向预训练BERT靠拢，以求能提取更有效的提取ID组成的用户行为句子特征

### 核心技术
TextRank召回长文本中top-3个关键句；对LSTM使用残差网络；为BERT设置损失函数bert_loss

### 特征工程
* 长文本处理：
使用TextRank召回top-3个关键句以代表整篇新闻。

TextRank算法是PageRank的改进，将每个句子视为一个顶点，句子之间的连接视为边，建立一张图，通过计算边的值得到权重进而召回top-k个关键句。

该任务中，新闻的内容呈现规律是开头和结尾是点睛之笔，因此我们截取开头和结尾召回核心内容：前128个字 和 后 382个字。

2种召回方法在该任务中结果相差不大，但是使用TextRank召回top-k个关键句则更具泛化性。

* Pooling：BERT 12层隐藏层的向量进行加权

Ganesh Jawahar等人[3]证明BERT每一层对文本的理解都不同，因此将BERT的十二层transformer生成的表示赋予一个权重,(下文称为BERT动态融合)，初始化公式为：

α_i =Dense_(unit=1) (represent_i)

ouput=Dense_(unit=512) （∑_(i=1)^nα_i · represent_i）

BERT的动态融合作为模型embedding的成绩会优于BERT最后一层向量作为模型的embedding，且收敛速度更快，epochs=3-4个就可以了

权重学习易造成高时空复杂度，我们还可以使用SumPooling、MeanPooling、MaxPooling等方法进行融合，选择层数偏7 8 9 10 11 12层

### 模型内部结构 Pooling
* 使用残差网络解决BERT下游任务模型退化问题

下游任务使用LSTM/GRU/Transformer(3选1) 等结构对 BERT动态融合结果 进行特征提取时，神经网络发生退化问题，模型没有欠拟合。因此我模拟ResNet中的残差结构，对LSTM/GRU/Transformer进行短接
F1、accracy等指标由原来的0.97 0.95来回跳 变为 稳定在0.96，由于使用残差网络，模型参数量下降了200-300个。

为了取消维度不一致影响残差网络使用率低的问题，作者开发了不受维度限制的残差模块。

原理是：回到ResNet的核心，非线性激活函数的存在导致特征变化不可逆，因此造成模型退化的根本原因是非线性激活函数。因此F(x)= f(x) + x 可以理解为f(x)为非线性特征，x为线性特征，

该残差模块不受维度相等的条件限制

时空复杂度为0的写法：遇到维度不相等，可以直接用`nn.Linear(), tf.keras.layers.Dense()`让维度一致。然后再对位相加即可。

有时空复杂度的写法：向量不对位相加，直接拼接`torch.cat([vector1, vector2],dim=-1), tf.concat([vector1, vector2], axis=-1) tf.keras.layers.concatation()`

原因是：BERT的12层向量融合完成很好的提取了特征，这种情况复杂的模型反而效果会减弱。这在推荐系统中很常见，特征工程之后用个逻辑回归LR就能解决问题，可能对于LR来说，它只需要发挥自己的记忆能力，把特征工程整理出来的情况都记录在自己的评分卡中，辅以查表和相法就可完成任务。

### 结果
动态融合BERT收敛速度比原始BERT快2-3个epoch，且模型也更能学明白。

### 可改进的点
BERT模型动态融合需要BERT预训练模型已经很完美，因此可以使用我们该任务的语料喂给开源的预训练模型再训练20个epoch。
[代码跑通，没有gpu还没跑出结果]BERT是无监督学习方式，该任务是文本分类，因此我们可以为BERT单独增加损失函数，相当于期中考试。具体做法是BERT最后一层的输出（也就是原始BERT）使用LR计算得到pre_label，与true_label计算得到损失

核心代码：
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
* 召回长文本的关键句

人工规则：标题 + 结论 = 一篇新闻文本
```python 
# 提取核心内容： 标题 + 结论 = 一篇新闻文本
def merge_text(text):
    if len(text) < 512:
        return text
    else:
        return text[:128] + text[-382:]
train_df['sentence'] = train_df['text'].apply(merge_text)
dev_df['sentence'] = dev_df['text'].apply(merge_text)
```
TextRank算法召回top-k关键句
```python 
# TextRank召回top-k个核心句子
# !pip install textrank4zh
from textrank4zh import TextRank4Sentence
def key_text(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    import_sentence = []
    for item in tr4s.get_key_sentences(num=3):  # sentence_num是生成关键句的个数
        # index是语句在文本中位置，weight表示权重
        # print('item.index, item.weight, item.sentence',item.index, item.weight, item.sentence)
        # self.import_sentence.append(str(item.index) + ' ' +item.sentence)
        import_sentence.append(item.sentence)
    # key_sentence = [i.sentence for i in tr4s.get_key_sentences(num=3)] # num生成关键句的个数
    # 核心内容是 标题 + top-3文本本体关键句
    key_sentences = import_sentence[0][2:] + import_sentence[1][2:] + import_sentence[2][2:]
    # 过长文本截断
    if len(key_sentences) < 512:
        return key_sentences
    else:
        return key_sentences[:512]

train_df['sentence'] = train_df['text'].apply(key_text)
dev_df['sentence'] = dev_df['text'].apply(key_text)
```

* 修改loss：为bert增加辅助损失函数
```python
# BERT增加损失函数：原始BERT输出直接做分类后计算一次损失（也由于BERT的重要性高于fine_tune部分，其loss权重可以高于fine_tune部分的权重）
ori_pooled_output = self.bert_cls(ori_pooled_output) #(none, 768) -> (none, 10)
bert_cls = F.softmax(ori_pooled_output, dim=-1)
bert_loss = nn.CrossEntropyLoss()(bert_cls, cls_label)

# 或者参考不均衡样本处理方法进行下采样，下采样是通用的思路 使用FocalLoss
class_loss = nn.CrossEntropyLoss()(classifier_logits, cls_label)# weight中设置不均衡的标签
class_loss = 0.8 * bert_loss + 0.2 * class_loss
outputs = class_loss
```

运行方案顺序
* preprocess/preprocess.py
* utils.py
* train_fine_tune.py
* predict.py
