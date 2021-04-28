
### 使用开源预训练BERT前，使用任务语料对该BERT进行再次预训练能有效提高 BERT 的信息提取能力

实际上我们使用BERT进行微调任务时，整个模型的表现就完全依赖于 BERT 提取的特征。虽然BERT本身的参数也在更新，但是由于微调任务的epoch次数较少，这时的BERT 与 原始版本相差不大，因此BERT针对任务语料的再次预训练就显得热别重要。

此外，考虑到 BERT 作为主力的特征提取，为了稳定 BERT 的表现，我们也可考虑给 BERT 增加 Loss，让 BERT 在微调中也增加监督学习的loss，让整个模型更好的收敛。

我们对 BERT 增加 loss，作者认为这个loss应该可以不使用 交叉信息熵 或 MSE 来计算与label之间的差距，BERT的输出也可视为一个loss。

使用任务语料对BERT再次预训练15-20个epoch，learning_rate=5e-5是建立的参数配置

如果我们从0开始预训练BERT，learning_rate = 1e-4


