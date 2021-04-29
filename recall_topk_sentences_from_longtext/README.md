### 任务目标

开发一套能快速提取新闻文本核心内容的工具，核心内容包括：关键词、关键句。关键词生成词云图

开发一款召回长文本关键句的工具，核心内容是解决NLP中长文本输入的问题（在text_classification中已实现）

### 原理
使用TextRank算法召回长文本top-k个关键句
* Step1：每个句子作为图中的节点
* Step2：如果两个句子相似，则节点之间存在一条无向有权边
* Step3：相似度 = 同时出现在两个句子中的单词的个数 / 句子中单词个数求对数之和（（分母使用对数可以降低长句在相似度计算上的优势）

![image](https://user-images.githubusercontent.com/68730894/115322192-0e3f0e80-a1b8-11eb-9919-a2bd6b5a5688.png)

TextRank中，windows=3，意味着前后3个句子作为顶点，两两使用边连接起来，表示句子间的关系强度。

![image](https://user-images.githubusercontent.com/68730894/115322208-18f9a380-a1b8-11eb-9ece-5f84c22edf15.png)

NLP的长文本解决方案有（以BERT为例，BERT最长只能输入512个字）
* 人工规则提取核心内容：前128个字 和 后382个字，适用于新闻、文章
* 截断：前510个字，从511个字开始的内容删除；或者后511个字，从后511个字开始的内容删除
* 召回算法：从长文本中召回top-k个关键句。

```python 
tr4s = TextRank4Sentence().analyze(text=text, lower=True, source='all_filters')
# print('摘要：')
# 重要性较高的3个句子
import_sentence = []
for item in tr4s.get_key_sentences(num=self.sentence_num): # sentence_num是生成关键句的个数
    # index是语句在文本中位置，weight表示权重
    # print('item.index, item.weight, item.sentence',item.index, item.weight, item.sentence)
    # import_sentence.append(str(item.index) + ' ' +item.sentence)
    import_sentence.append(item.sentence)

# 召回top-3个关键句代表一篇长文本
key_sentences = import_sentence[0][2:] + import_sentence[1][2:] + import_sentence[2][2:]
if len(key_sentences) <= 510:
    return key_sentences
else:
    return key_sentences[:510]

```

环境要求
pip install textrank4zh
