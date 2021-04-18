import torch.nn as nn
from config import Config
import math
import torch
import torch.nn.functional as F

config = Config()
if config.pretrainning_model == 'nezha':
    from NEZHA.model_nezha import BertPreTrainedModel, NEZHAModel
elif config.pretrainning_model == 'albert':
    from transformers import AlbertModel, BertPreTrainedModel
else:
    # bert,roberta
    from transformers import RobertaModel, BertPreTrainedModel

if config.pretrainning_model == 'nezha':
    from NEZHA.model_nezha import BertPreTrainedModel, BertPreTrainedModel


class ResidualWrapper4RNN(nn.Module):
    def __init__(self, model):
        super().__init__() # super(ResidualWrapper, self).__init__()
        self.model = model
        # self.alpha = nn.Parameter(torch.zeros((params.batch_size,1,1)))

    def forward(self, inputs, *args, **kwargs):
        # MLP中inputs.shape=(N, 512),  delta.shape=(batch_size, 10)
        delta = self.model(inputs, *args, **kwargs)[0]  # params.model_type=='bigru' or 'bilstm'
        return inputs + delta
        # 不使用相加，直接在inputs的后面把delta拼接上去，组成 (N, 512 + 10)的向量。推荐系统中DIEN就是这么弄的
        # model pooling strategy: inputs + self.alpha * delta

class ResidualWrapper4RTransformer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs) # params.model_type == 'transformer'
        return inputs + delta

# 不平衡标签多分类
class Multi_FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True): # alpha=1, gamma=2
        super(Multi_FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets, *args, **kwargs):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class BertForCLS(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.params = params
        self.config = config
        # nezha
        if params.pretrainning_model == 'nezha': #(batch_siz, seq_len, 768)
            self.bert = NEZHAModel(config)
        elif params.pretrainning_model == 'albert':
            self.bert = AlbertModel(config)
        else:
            self.bert = RobertaModel(config)

        # BERT动态权重
        self.classifier = nn.Linear(config.hidden_size, 1)  # for dym's dense # hidden_size=768
        # 新增：为了解决(batch_size, seq_len) * (768, 10)维度不匹配的问题
        self.dym_pool = nn.Linear(params.embed_dense, 1)
        # 768 -> 512
        self.dense_final = nn.Sequential(nn.Linear(config.hidden_size, params.embed_dense), # (batch_size, hidden_size=768, embed_dense=512)
                                         nn.ReLU(True))  # 动态最后的维度 残差连接
        self.dense_emb_size = nn.Sequential(nn.Linear(config.hidden_size, params.embed_dense),
                                            nn.ReLU(True)) # 降维
        # 初始化一个动态权重，让机器自动学习12个隐藏层信息的权重,此时的lr=5e-5，embed_name=bert.embeddings.word_embeddings.weight
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)), # num_hidden_layers12
                                       requires_grad=True)  # tensor([[[[1.]]], [[[1.]]], ..共有12个[[[1.]]].])
        # 使用nn.init.xavier_normal_(self.dym_weight)
        self.pool_weight = nn.Parameter(torch.ones((params.batch_size, 1, 1, 1)),
                                        requires_grad=True)

        if params.model_type == 'bilstm':
            num_layers = self.params.rnn_num
            lstm_num = int(self.params.embed_dense / 2)
            # self.lstm = nn.LSTM(self.params.embed_dense, lstm_num,
            #                     num_layers, batch_first=True,  # 第一维度是否为batch_size
            #                     dropout=0.1,
            #                     bidirectional=True)  # 双向
            self.lstm = ResidualWrapper4RNN(nn.Sequential(
                                nn.LSTM(input_size=self.params.embed_dense, hidden_size=lstm_num,
                                        num_layers=num_layers, batch_first=True, dropout=config.hidden_dropout_prob,
                                        bidirectional=True)))
        elif params.model_type == 'bigru':
            num_layers = self.params.rnn_num
            lstm_num = int(self.params.embed_dense / 2)
            # self.lstm = nn.GRU(input_size=self.params.embed_dense, # input_shape=embed_dense=512
            #                    hidden_size=lstm_num,  # output_shape=lstm_num= 256 = 512 /2
            #                    num_layers=num_layers, # 2层双向RNN学习BERT 12 层隐藏层的权重
            #                    dropout=0.1,
            #                    batch_first=True,  # 第一维度是否为batch_size
            #                    bidirectional=True)  # 双向
            # RNN_model = self.lstm # https://pytorch.org/docs/master/generated/torch.nn.GRU.html#torch.nn.GRU
            self.lstm = ResidualWrapper4RNN(nn.Sequential(
                nn.GRU(self.params.embed_dense, lstm_num, num_layers, batch_first=True,
                       dropout=config.hidden_dropout_prob, bidirectional=True))) # nn.GRU输出2个内容： output 和 hidden
        # elif params.model_type == 'cnn':
        #     lstm_num = int(self.params.embed_dense / 2)
        #     self.lstm = ResidualWrapper(nn.Sequential(
        #         nn.Conv1d(in_channels=params.embed_dense, out_channels=lstm_num, kernel_size=)
        #         nn.MaxPool1d()
        #     ))
        elif params.model_type == 'transformer':
            # self.positional_embedding = nn.Embedding(self.max_len, self.embed_dense) # 学习sequence_output的位置关系
            # sequence_output.shape=(batch_size, seq_len, embed_size) 只有一个张量作为输入，所以可以不用考虑positional encoding
            self.lstm = ResidualWrapper4RTransformer(nn.Sequential(
                # TransformerPositionalEncoding(d_model=params.embed_dense, dropout=config.hidden_dropout_prob, max_len=params.sequence_length),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(self.params.embed_dense, nhead=8, dropout=config.hidden_dropout_prob), num_layers=6)
            ))

        # 全连接分类
        self.bert_cls = nn.Linear(config.hidden_size, params.cls_num) # (none, max_len, 768)

        mlp_middle_layer = int(self.params.embed_dense / 2)
        self.cls = nn.Sequential(nn.Linear(params.embed_dense, mlp_middle_layer),
                                 nn.ReLU(),
                                 nn.Linear(mlp_middle_layer, params.cls_num))  # for cls  512 --> 10 768个隐藏层变为10个分类
        # self.cls = ResidualWrapper(nn.Sequential(nn.Linear(params.embed_dense, 256),
        #                                          nn.ReLU(),
        #                                          nn.Linear(256, params.cls_num)))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # dropout
        if params.pretrainning_model == 'nezha':
            self.apply(self.init_bert_weights)
        else:
            self.init_weights()
        self.reset_params()  # nn.init.xavier_normal_(self.dym_weight)

    def reset_params(self):
        return nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs): # 神经网络学习参数：Attention
        layer_logits = []
        all_encoder_layers = outputs[1:]  # 00层有2条信息重复，所以有13层 取12层=[1:]
        for i, layer in enumerate(all_encoder_layers):
            # layer.shape = (N, 512, 768) = (batch_size, seq_len, bert_embed_size)
            # self.classifier(nn.Linear): (N, 512, 768) -> (, 512, 1) # 768维的向量压缩成1维, 方便后面做融合
            # layer_logits.shape=(N, 512, 1)
            # layer_logits.append(self.classifier(layer))
            layer_logits.append(self.classifier(layer)) # 768 -> 1
        layer_logits = torch.cat(layer_logits, dim=2) # layer_logits.shape = (N, 512, 12) 在dim=2维上加起来
        layer_dist = torch.nn.functional.softmax(layer_logits, dim=2) # dim=X 应该是embed_size维度 按行还是按列的问题 # (N, 512, 12)
        seq_out = torch.cat([torch.unsqueeze(x, dim=2) for x in all_encoder_layers], dim=2) # seq_out.shape=(N, 512, 12, 768)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, dim=2), seq_out) # pooled_output = (N, 512, 1, 768)
        pooled_output = torch.squeeze(pooled_output, dim=2) # pooled_output.shape=(N, 512, 768)
        ## 是否真的需要激活函数， 加上残差网络就不用纠结激活函数了： BERT 768维 -> MLP中 512
        word_embed = self.dense_final(pooled_output) # word_embed.shape=(N, 512, 512)
        dym_layer = word_embed
        return dym_layer

    # def weight_layers_attention(self, output):
    #     layer_logits = []
    #     all_encoder_layers = outputs[1:]  # 00层有2条信息重复，所以有13层 取12层=[1:]
    #     for i, layer in enumerate(all_encoder_layers):
    #         layer_logits.append(self.classifier(layer))

    def get_weight_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return: sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[1:], dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output



    def forward(self, input_ids, # 每个字通过字典转为序号
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                cls_label=None): # 标签

        # 特征提取：Nezha
        if self.params.pretrainning_model == 'nezha':
            # 得到BERT中每个隐藏层的权重 一层的shape = (batch_size, seq_len, embed_size) = (N, 512, 768)
            # BERT隐藏层的形状
            # encoded_layers.shape = (num_hidden_layers, batch_size, seq_len, embed_size) = (12, N, 512, 768)
            # ori_pooled_output.shape = (batch_size, embed_size) = (1, 768)
            encoded_layers, ori_pooled_output = self.bert(
                input_ids,     # (seq_len, batch_size) = (512, 1)
                attention_mask=attention_mask, # (batch_size, seq_len)=(N, 512)
                token_type_ids=token_type_ids, # (batch_size, seq_len)=(N, 512)
                output_all_encoded_layers=True,
            )  # encoded_layers, pooled_output
            sequence_output = encoded_layers[-1]  # 取最后一层隐藏层，(batch_size, seq_len, embed_size)=(N, 512, 768)
        else:
            sequence_output, ori_pooled_output, encoded_layers, att = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )  # sequence_output, pooled_output, (hidden_states), (attentions)

        if self.params.use_origin_bert == 'dym': # 12层信息融合成一个向量
            # BERT 12个隐藏层向量揉成一条： 768维的BERT_embedding 向量变为 512 维 embedding
            # (num_hidden_layers, batch_size, seq_len, embed_size) = (12, N, 512, 768) --> [batch_size, seq_len, 512]
            sequence_output = self.get_dym_layer(encoded_layers)  # [batch_size,seq_len,512] = [N, 512, 512]
            # 如果是BERT原始输出（最后一层隐藏层）则是[batch_size, seq_len, 768]
        elif self.params.use_origin_bert == 'weight':
            sequence_output = self.get_weight_layer(encoded_layers)
            # new code
            sequence_output = self.dense_final(sequence_output) #(batch_size, seq_len, 512)
        # new code
        else:
            sequence_output = self.dense_final(sequence_output) #(batch_size, seq_len, 512)

        # 文本分类模型-特征提取：融合后的向量喂给RNN+MLP
        if self.params.model_type == 'bilstm' or self.params.model_type == 'bigru': # [batch_size,seq_len, 512]
            # GRU输出两个内容：the whole sequence output + final_state(hidden), tuple中取第一维, (batch_size, 512, 512)
            # whole_sequence_output.shape = (batch_size, 512, 264)  final_state.shape=(batch_size, 264)
            # (N, 512, 512) =>
            sequence_output = self.lstm(sequence_output)#[0] # 如果不加residual connection 则要加下标[0] #(N, 512, 512)
        elif self.params.model_type == 'transformer':
            sequence_output = self.lstm(sequence_output) # (N, 512, 512)

        # 文本分类模型-MLP: classification
        "分类"""
        "" # Pooling 后再喂给MLP
        # print('sequence_output.shape', sequence_output.shape)
        if self.params.is_avg_pool == 'max':
            # (batch_size, seq_len, mlp_embed)=(1, 512, 512) --> (batch_size, mlp_embed, seq_len)=(1, 512, 512)
            pooled_output = F.max_pool1d(sequence_output.transpose(1,2), self.params.sequence_length) # (1, 512, 1)
            pooled_output = torch.squeeze(pooled_output, -1) # pooled_output.shape=(1, 512)
        elif self.params.is_avg_pool == 'mean':
            pooled_output = F.avg_pool1d(sequence_output.transpose(1,2), self.params.sequence_length)
            pooled_output = torch.squeeze(pooled_output, -1)
        elif self.params.is_avg_pool == 'dym': # 加权：初始化 1*2 的矩阵，看哪个效果更好，哪个效果好则往哪个权重偏移，
            # print(sequence_output.shape) # shape after resnet ([512, 512])    ori code (N, 512, 512)
            sequence_output = sequence_output.transpose(1, 2).contiguous()
            maxpooled_output = torch.nn.functional.max_pool1d(sequence_output, self.params.sequence_length) #(1, 512, 1)
            maxpooled_output = torch.squeeze(maxpooled_output, -1) #(N, 512)
            meanpooled_output = torch.nn.functional.avg_pool1d(sequence_output, self.params.sequence_length) # (N, 512, 1)
            meanpooled_output = torch.squeeze(meanpooled_output, -1)
            pooled_output = self.dym_pooling1d(meanpooled_output, maxpooled_output) # pooled_output.shape = (N, 512)
        elif self.params.is_avg_pool == 'weight':
            maxpooled_output = torch.nn.functional.max_pool1d(sequence_output.transpose(1,2),self.params.sequence_length)
            maxpooled_output = torch.squeeze(maxpooled_output, -1)
            meanpooled_output = torch.nn.functional.avg_pool1d(sequence_output.transpose(1,2), self.params.sequence_length)
            meanpooled_output = torch.squeeze(meanpooled_output, -1)
            pooled_output = self.weight_pooling1d(meanpooled_output, maxpooled_output)
        else:
            pooled_output = ori_pooled_output
            # new code
            pooled_output = self.dense_emb_size(pooled_output) # 768 -> 512
        # print('pooled_output.shape', pooled_output.shape)
        cls_output = self.dropout(pooled_output)  # (N, 512)
        classifier_logits = self.cls(cls_output)  # [bacth_size*]  #(batch_size, 10)


        if cls_label is not None:
            # BERT增加损失函数：原始BERT输出直接做分类后计算一次损失（也由于BERT的重要性高于fine_tune部分，其loss权重可以高于fine_tune部分的权重）
            ori_pooled_output = self.bert_cls(ori_pooled_output) #(none, 768) -> (none, cls_num)
            # temp = (ori_pooled_output, )
            # if len(cls_output) > 2:
            #     bert_cls = F.softmax(ori_pooled_output, dim=-1)
            # else:
            #     bert_cls = F.sigmoid(ori_pooled_output)
            # bert_loss = nn.CrossEntropyLoss()(bert_cls, cls_label)

            # 或者参考不均衡样本处理方法进行下采样，下采样是通用的思路 使用FocalLoss
            class_loss = nn.CrossEntropyLoss()(classifier_logits, cls_label.view(-1))# weight中设置不均衡的标签
            # class_loss = Multi_FocalLoss(alpha=0.25, gamma=2)(classifier_logits, cls_label)
            # class_loss =  #0.8 * bert_loss + 0.2 * class_loss
            outputs = ori_pooled_output + class_loss
            outputs = class_loss, classifier_logits, encoded_layers  # 后两项为知识蒸馏
        else:
            outputs = classifier_logits, encoded_layers  #

        return outputs

    def dym_pooling1d(self, avgpooled_out, maxpooled_out):
        pooled_output = [avgpooled_out, maxpooled_out]  # 0: (batch_size, 512)  1: (batch_size, 512)
        pool_logits = []
        for i, layer in enumerate(pooled_output):
            # new code
            pool_logits.append(self.dym_pool(layer))  # pool_logits.shape=(2, 1) 由(batch_size, 512) -> (batch_size, 1)
        pool_logits = torch.cat(pool_logits, dim=-1) # (batch_size, 2)
        pool_dist = torch.nn.functional.softmax(pool_logits, dim=-1) # dim=1   (batch_size, 2)
        pooled_out = torch.cat([torch.unsqueeze(x, dim=2) for x in pooled_output], dim=2) # (batch_size, 512, 2)
        pooled_out = torch.unsqueeze(pooled_out, dim=1)  # (batch_size, 1, 512, 2)
        pool_dist = torch.unsqueeze(pool_dist, dim=2) #(batch_size, 2, 1)
        pool_dist = torch.unsqueeze(pool_dist, dim=1) #(batch_size, 1, 2, 1)
        pooled_output = torch.matmul(pooled_out, pool_dist) #(batch_size, 1, 512, 1) = (N, 1, 512, 2) * (N, 1, 2, 1)
        pooled_output = torch.squeeze(pooled_output)  # (batch_size, 512) 删除2维为1的张量
        return pooled_output

    def weight_pooling1d(self, avgpooled_out, maxpooled_out):
        outputs = [avgpooled_out, maxpooled_out]
        # new code
        hidden_stack = torch.unsqueeze(torch.stack(outputs, dim=-1), dim=1)  # (batch_size, 1, hidden_size,2)
        # hidden_stack = torch.stack(outputs, dim=-1)
        # print('hidden_stack', hidden_stack)
        sequence_output = torch.sum(hidden_stack * self.pool_weight,
                                    dim=-1)  # (batch_size, seq_len, hidden_size[embedding_dim])
        # print('sequence_output.shape',sequence_output.shape)
        sequence_output = torch.squeeze(sequence_output)
        # print('sequence_output', sequence_output.shape)
        return sequence_output

class TransformerPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(TransformerPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).contiguous()
        self.register_buffer('pe', pe) # 封装为self.pe层

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerPositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        self.tok_embedding = nn.Embedding(d_model, params.embed_dense)
        self.pos_embedding = nn.Embedding(max_len, params.embed_dense)
    def forward(self, d_model, max_len):
        word_embedding = self.tok_embedding(d_model)
        position_embedding = self.pos_embedding(max_len)
        return word_embedding + position_embedding # = input_embedding

if __name__ == '__main__':
    Bert_config = 'E:/18.BERT_pretrained/nezha-cn-base/Bert_config'
    net = BertForCLS(config=Bert_config, params=config)
    print(net)
