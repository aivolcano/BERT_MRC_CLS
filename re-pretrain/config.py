# -*- coding: utf-8 -*-
# @Time    : 2021/1/12
# @Author  : chenyancan
# @Email   : ican22@foxmail.com
# @Software: PyCharm


class Config(object):
    def __init__(self):
        # -----------ARGS---------------------
        # 原始数据路径
        self.source_data_path = 'E:/30.NLP_training/text_classification/'
        # 预训练数据路径
        self.pretrain_train_path = "E:/30.NLP_training/text_classification/Mybert/256_corpus.txt"
        # 模型保存路径
        self.output_dir = self.source_data_path + "outputs/"
        # MLM任务验证集数据，大多数情况选择不验证（predict需要时间,知道验证集只是表现当前MLM任务效果）
        self.pretrain_dev_path = ""

        # 预训练模型所在路径（文件夹）为''时从零训练，不为''时继续训练。huggingface roberta
        # 下载链接为：https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
        self.pretrain_model_path = 'E:/30.NLP_training/text_classification/pre_model_nezha_base/'
        self.bert_config_json = self.pretrain_model_path + "bert_config.json"  # 为''时从零训练
        # 从0开始预训练：关闭vocab_file 和 init_model
        self.vocab_file = self.pretrain_model_path + "vocab.txt"
        self.init_model = self.pretrain_model_path + 'pretraining_model'# 下载好模型的model  'pretraining_model.bin'

        self.max_seq_length = 256  # 文本长度 preprocess配置
        self.do_train = True
        self.do_eval = False
        self.do_lower_case = False  # 数据是否全变成小写（是否区分大小写）

        self.train_batch_size = 24  # 根据GPU卡而定
        self.eval_batch_size = 32
        # 继续预训练lr：5e-5，重新预训练：1e-4(重新预训练模型学习率)
        self.learning_rate = 5e-5
        self.num_train_epochs = 16  # 预训练轮次
        self.save_epochs = 2  # e % save_epochs == 0 保存
        # 前warmup_proportion的步伐 慢热学习比例
        self.warmup_proportion = 0.1
        self.dupe_factor = 1  # 动态掩盖倍数
        self.no_cuda = False  # 是否使用gpu
        self.local_rank = -1  # 分布式训练
        self.seed = 42  # 随机种子(结果更好可以设置随机数种子)

        # 梯度累积（相同显存下能跑更大的batch_size）1不使用
        self.gradient_accumulation_steps = 1
        self.fp16 = False  # 混合精度训练（哪吒特有）
        self.loss_scale = 0.  # 0时为动态
        # bert Transormer的参数设置
        self.masked_lm_prob = 0.15  # 掩盖率 15%的字会被掩盖
        # 最大掩盖字符数目
        self.max_predictions_per_seq = 20 # 20个字被掩盖
        # 冻结word_embedding参数
        self.frozen = True

        # bert参数解释
        """
        {
          # 乘法attention时，softmax后dropout概率
          "attention_probs_dropout_prob": 0.1,  
          "directionality": "bidi", 
          "hidden_act": "gelu", # 激活函数
          "hidden_dropout_prob": 0.1, #隐藏层dropout概率
          "hidden_size": 768, # 最后输出词向量的维度  (batch_size, seq_len, hidden_size)
          "initializer_range": 0.02, # 初始化范围
          "intermediate_size": 3072, # 升维维度
          "max_position_embeddings": 512, # 最大的
          "num_attention_heads": 12, # 总的头数
          # 隐藏层数 ，也就是transformer的encode运行的次数
          "num_hidden_layers": 12,  #hidden的层数
          "pooler_fc_size": 768, 
          "pooler_num_attention_heads": 12, 
          "pooler_num_fc_layers": 3, 
          "pooler_size_per_head": 128, 
          "pooler_type": "first_token_transform", 
          "type_vocab_size": 2, # segment_ids类别 [0,1]
          "vocab_size": 21128  # 词典中词数 （自动弄一个脱敏数据文本（从0开始预训练） 设置hidden_size和头数
        }
        """
