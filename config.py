# -*- coding: utf-8 -*-
# @Time    : 2021/1/2 14:15
# @Author  : chenyancan
# @Email   : ican22@foxmail.com
# @File    : config.py
# @Software: PyCharm


class Config(object): # params.decay_rate, params.embed_size
    def __init__(self):
        self.base_dir = './processed_data/' # 数据路径
        self.save_model = self.base_dir + 'Savemodel/'  # 模型路径
        self.result_file = 'result/'
        self.label_list = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
        self.TextRank = True # 长文本是否使用TextRank召回top-3个关键词

        self.warmup_proportion = 0.05
        self.use_bert = True
        self.pretrainning_model = 'nezha' # 'roberta'
        self.embed_dense = 512 # BERT 768 --> 512

        self.decay_rate = 0.5

        self.train_epoch = 20

        self.learning_rate = 1e-4 # 全连接层 和 BERT 12个隐藏层融合为一条向量的lr
        self.embed_learning_rate = 5e-5 # BERT的lr

        if self.pretrainning_model == 'roberta':
            model = 'E:/18.BERT_pretrained/pre_model_roberta_base/'  # 中文roberta-base
        elif self.pretrainning_model=='nezha':
            model = 'E:/18.BERT_pretrained/nezha-cn-base/'
        else:
            raise KeyError('albert nezha roberta bert bert_wwm is need')
        self.cls_num = 10
        self.sequence_length = 512
        self.batch_size = 2 #6

        self.model_path = model

        self.bert_file = model + 'pytorch_model.bin'
        self.bert_config_file = model + 'bert_config.json'
        self.vocab_file = model + 'vocab.txt'

        # BERT 动态融合参数
        self.use_origin_bert = 'dym'  # 'ori':使用原生bert, 'dym':使用动态融合bert,'weight':初始化12*1向量
        self.is_avg_pool = 'weight'  #  dym, max, mean, weight
        self.model_type = 'gat'  # Bilstm; Bigru, transformer

        self.rnn_num = 2 # 可以换成transformer
        self.flooding = 0
        self.embed_name = 'bert.embeddings.word_embeddings.weight'  # 词
        self.restore_file = None
        self.gradient_accumulation_steps = 1
        # 模型预测路径
        # self.checkpoint_path = "/Savemodel/runs_0/model_0.9720_0.9720_0.9720_3500.bin"
        self.checkpoint_path = './Savemodel/runs_0/'

        """
        实验记录
        """

if __name__ == "__main__":
    print(torch.__version__())