
任务目标
开发一套能快速提取新闻文本核心内容的工具，核心内容包括：关键词、关键句。关键词生成词云图

开发一款召回长文本关键句的工具，核心内容是解决NLP中长文本输入的问题

原理
使用TextRank算法召回长文本top-k个关键词和关键句

Step1，进行分词和词性标注，将单词添加到图中，删除停用词
Step2，出现在一个窗口（window）中的词形成一条边
Step3，基于PageRank原理进行迭代（20-30次）
Step4，顶点（词）按照分数进行排序，可以筛选指定的词性
注：W_ij: 单词i和j之间的权重 节点的权重不仅依赖于入度，还依赖于入度节点的权重

TextRank中，windows=3，意味着前后3个句子作为顶点，两两使用边连接起来，表示句子间的关系强度。

NLP的长文本解决方案有（以BERT为例，BERT最长只能输入512个字）

人工规则提取核心内容：前128个字 和 后382个字，适用于新闻、文章
截断：前510个字，从511个字开始的内容删除；或者后510个字，从后511个字开始的内容删除
召回算法：从长文本中召回top-k个关键句。
tr4s = TextRank4Sentence().analyze(text=self.text, lower=True, source='all_filters')
# print('摘要：')
# 重要性较高的3个句子
for item in tr4s.get_key_sentences(num=self.sentence_num): # sentence_num是生成关键句的个数
    # index是语句在文本中位置，weight表示权重
    # print('item.index, item.weight, item.sentence',item.index, item.weight, item.sentence)
    # self.import_sentence.append(str(item.index) + ' ' +item.sentence)
    self.import_sentence.append(item.sentence)

# 召回top-3个关键句代表一篇长文本
text4bert = readingNews1.title + readingNews1.import_sentence[0][2:] + readingNews1.import_sentence[1][2:] + readingNews1.import_sentence[2][2:]
环境要求

pip install textrank4zh
