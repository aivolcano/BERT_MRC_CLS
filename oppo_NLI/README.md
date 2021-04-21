

这场比赛很卷，前排打得太激烈了。可能是春节后启动较早的比赛，大家都去尝试它。


比赛是脱敏数据，苏建林给出了vocab对齐的思路，可以直接使用现成的预训练模型搞，参考链接https://kexue.fm/archives/8213

数据增广策略：A=B，B=C => A=C；

针对脱敏文本单独训练BERT；

BERT隐藏层Pooling：我们使用MaxPooling取出特征最明显的一个隐藏层，使用MeanPooling获得12个隐藏层的平均信息量，最后拼接在一起喂给下游任务

Loss修改：
简单的交叉信息熵loss容易遇到【模型loss很小，但是没有学明白的问题】。因此我们把增加12层BERT Pooling后的向量降维成2维作为一部分loss。提升1个点


这场比赛是二分类任务，因此使用机器学习模型也是可以的，这需要完全依赖于特征工程，作者也尝试做了一些特征工程：query1 和 query2的tfidf、word2vec、PageRank，
这个比赛存在leak问题：一个query如果频繁出现，很有可能是同一个query，所以需要在leak上做文章。https://zhuanlan.zhihu.com/p/35093355 。所以使用图论技术挖掘是很有意义的

最后使用NEZHA预训练模型 和 BERT 预训练进行融合，得到90.8的成绩

