## LDA详解

召回top-k个主题，LDA采用词袋模型，词袋模型就是对于一篇文档，我们仅考虑一个词汇是否出现，而不考虑它出现的顺序，比如：‘我喜欢你’ 和 ‘你喜欢我’在词袋模型看来是一样的。

目的就是要识别主题，即把文档—词汇矩阵变成文档—主题矩阵（分布）和主题—词汇矩阵（分布）

![image](https://user-images.githubusercontent.com/68730894/116525518-2d9e1000-a90b-11eb-9c18-73374d0fac00.png)

### 文本建模：
一篇文档，可以看成是一组有序的词的序列

![image](https://user-images.githubusercontent.com/68730894/116525630-4c9ca200-a90b-11eb-869b-ac0c1574371d.png)

一篇文档包含多个主题，每一个词都由其中的一个主题生成

从统计学角度来看，文档的生成可以看成是上帝抛掷骰子生成的结果，每次抛掷骰子都生成一个词汇，抛掷N词生成一篇文档。于是我们关心两个问题：
  * 上帝都有什么样的骰子 => 选哪个主题
  * 上帝是如何抛掷这些骰子的 => 选哪个词


### LDA是如何实现的？
一篇文档看成一组有序的词的序列：doc=(w_1,w_2,w_3,…,w_n )
一篇文档包括多个主题，每个词都由其中的一个主题生成

从统计学的角度，文档的生成可看成是上帝抛掷骰子生成的结果，每个抛骰子都生成一个词汇，抛掷N次生成一篇文档。那么上帝拥有什么骰子就定了能生成哪些主题，抛骰子的过程就是选哪个词语。此时，文章和单词就由主题决定。因此我们只需要识别出主题就可实现 文档-主题-词汇的结构。

文本建模的具体过程：
Step1，假设事先给定了这几个主题：Arts、Budgets、Children、Education，然后通过学习训练，获取每个主题Topic对应的词语，此时有4个主题，类似上帝的骰子里面只有4面，每面分别对应Arts、Budgets、Children、Education。每个主题topic里面对应该主题下的多个单词words。比如：Arts里面有NEW、FILM...，这些四六级写英文作业类似，环境主题有环境相关的词汇、教育主题有教育相关的地道表述。

![image](https://user-images.githubusercontent.com/68730894/116525986-b2892980-a90b-11eb-8245-8c1fb83f0ff9.png)

Step2：生成文本：先选主题，再选该主题下的词语，如此反复。比如一定概率从4个主题选出Arts，在以一定概率从Arts下面选择FILM；再以一定概率从4个主题里面选Children，再选出SAYS...

![image](https://user-images.githubusercontent.com/68730894/116526030-bd43be80-a90b-11eb-98b8-000224bce6b0.png)

pLSA生成文档的过程：
两种类型的骰子，一种是doc-topic骰子，另一种是topic-word骰子。生成每篇文档之前，先为这篇文章制造一个特定的doc-topic骰子，重复如下过程生成文档中的词
* 投掷这个doc-topic骰子，得到一个topic编号z
* 选择K个topic-word骰子中编号为z的那个，投掷这个骰子，得到一个词

![image](https://user-images.githubusercontent.com/68730894/116526165-e401f500-a90b-11eb-826a-eb08202c935b.png)

### 从概率的角度理解pLSA和LDA：
在pLSA中，doc-topic和topic-word的骰子概率是固定的，LDA中，这两个骰子不固定，但是有范围，相当于给doc-topic骰子变成坛子，从坛子里面选不同的主题k；也把topic-word骰子变成坛子，从坛子里面选择主题k对应的单词。

![image](https://user-images.githubusercontent.com/68730894/116526276-0267f080-a90c-11eb-9a43-03bacaf6e466.png)

LDA相当于pLSA的贝叶斯优化版本。LDA认为存在概率的概率，也就是右图中的数字是不固定的，那么在doc-topic坛子中有多个骰子，比如【30%教育 50%经济 20%交通】【30%教育 35%经济 45%交通】等等，在topic-word坛子有多个骰子，比如【30%大学 50%老师 20%课程】【30%大学 45%老师 35%课程】等等。

最终的做法就是：
Step1：一个装有无穷多个骰子的坛子，里面装有各式各样的骰子，每个骰子有V个面
Step2：现从坛子中抽取一个骰子出来，然后使用这个骰子进行抛掷，直到产生语料库中的所有词汇

![image](https://user-images.githubusercontent.com/68730894/116526338-14499380-a90c-11eb-9b56-a98160189eed.png)

一言以蔽之，LDA在pLSA的基础上：为主题分布和词分布分别加了两个 Dirichlet 先验经验。具体来说，增加了生成第n个词的主题分布，以及生成主题k对应的词语分布。体现在概率中，p(z_k |d_i )和p(w_j |z_k )中，d_i和z_k在pLSA中是固定数字，在LDA中是一种概率，具有不确定性，服从多项式分布。

![image](https://user-images.githubusercontent.com/68730894/116526465-36dbac80-a90c-11eb-891f-491eb003b3ac.png)
