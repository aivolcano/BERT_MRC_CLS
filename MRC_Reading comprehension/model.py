import torch.nn as nn
import torch
from config import Config
import torch.nn.functional as F

config = Config()
if config.pretrainning_model == 'nezha':
    from NEZHA.model_nezha import BertPreTrainedModel, NEZHAModel, BertPreTrainedModel
elif config.pretrainning_model == 'albert':
    from transformers import AlbertModel, BertPreTrainedModel
else:
    # bert,roberta
    from transformers import BertModel, BertPreTrainedModel

from down_layer import BiLSTM, IDCNN, RTransformer, TENER

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class ResidualWrapper4RNN(nn.Module):
    def __init__(self, model):
        super().__init__() # super(ResidualWrapper, self).__init__()
        self.model = model
        # self.alpha = nn.Parameter(torch.zeros((params.batch_size,1,1)))

    def forward(self, inputs, *args, **kwargs):
        # MLP中inputs.shape=(N, 512),  delta.shape=(batch_size, 10)
        delta = self.model(inputs, *args, **kwargs)[0]  # params.model_type=='bigru' or 'bilstm'
        return inputs + delta

class ResidualWrapper4RTransformer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs) # params.model_type == 'transformer'
        return inputs + delta

class BertForQA(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.params = params
        # nezha
        if params.pretrainning_model == 'nezha':
            self.bert = NEZHAModel(config)
        elif params.pretrainning_model == 'albert':
            self.bert = AlbertModel(config)
        else:
            self.bert = BertModel(config)

        # 动态权重
        self.classifier = nn.Linear(config.hidden_size, 1)  # for dym's dense
        self.dense_final = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.ReLU(True))  # 动态最后的维度
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)),
                                       requires_grad=True)
        self.pool_weight = nn.Parameter(torch.ones((2, 1, 1, 1)),
                                        requires_grad=True)
        # 结构
        self.num_labels = 768
        self.idcnn = ResidualWrapper4RTransformer(nn.Sequential(
            IDCNN(config, params, filters=params.filters, tag_size=self.num_labels, kernel_size=params.kernel_size)))

        self.bilstm = ResidualWrapper4RNN(nn.Sequential(
            BiLSTM(self.num_labels, embedding_size=config.hidden_size, hidden_size=params.lstm_hidden,
                             num_layers=params.num_layers, dropout=params.drop_prob, with_ln=True)
                             ))
        self.tener = ResidualWrapper4RTransformer(nn.Sequential(
            TENER(tag_size=self.num_labels, embed_size=config.hidden_size, dropout=params.drop_prob,
                           num_layers=params.num_layers, d_model=params.tener_hs, n_head=params.num_heads)))

        self.rtransformer = ResidualWrapper4RTransformer(nn.Sequential(
            RTransformer(tag_size=self.num_labels, dropout=params.drop_prob, d_model=config.hidden_size,
                                         ksize=params.k_size, h=params.rtrans_heads)))

        # 任务一：MRC（回归任务）
        self.start_outputs = nn.Linear(config.hidden_size, 1) # 缩放成起始点位置
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        # 任务二：是否包含答案（分类任务）
        self.task2 = nn.Linear(config.hidden_size, 1)

        if params.pretrainning_model == 'nezha':
            self.apply(self.init_bert_weights)
        else:
            self.init_weights()
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.dym_weight) # 初始化dym_weight的权重

    def get_dym_layer(self, outputs):
        layer_logits = []
        all_encoder_layers = outputs[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))
        layer_logits = torch.cat(layer_logits, 2)  # (batch_size, 512, 12)
        layer_dist = F.softmax(layer_logits, dim=-1)   # (batch_size, 512, 12)
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2) # (batch_size, 512, 12, 768)
        # temp = torch.unsqueeze(layer_dist, dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out) # (batch_size, 512, 1, 768)
        pooled_output = torch.squeeze(pooled_output, 2)
        word_embed = self.dense_final(pooled_output)
        dym_layer = word_embed
        return dym_layer

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

    # mask
    def mask_logits(self, logits, attention_mask):
        # padding位置的mask
        mask_list = attention_mask.float()
        logits = logits - (1.0 - mask_list) * 1e12  # mask位置得到一个很小的值
        return logits

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None,
            cls_label=None,
    ):
        # pretrain model
        # Nezha
        if config.pretrainning_model == 'nezha':
            encoded_layers, pooled_output = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_all_encoded_layers=True
            )  # encoded_layers, pooled_output
            sequence_output = encoded_layers[-1]
        else:
            sequence_output, pooled_output, encoded_layers = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )  # sequence_output, pooled_output, (hidden_states), (attentions)

        if self.params.fuse_bert == 'dym':
            sequence_output = self.get_dym_layer(encoded_layers)  # [batch_size,seq_len,768] = (batch_size, 512, 768)
        elif self.params.fuse_bert == 'weight':
            sequence_output = self.get_weight_layer(encoded_layers)  # [batch_size,seq_len,768]

        # middle
        # (seq_len, batch_size, hidden_size)
        if self.params.mid_struct == 'bilstm':
            feats = self.bilstm(sequence_output.transpose(1, 0), attention_mask.transpose(1, 0))
        elif self.params.mid_struct == 'idcnn':
            feats = self.idcnn(sequence_output).transpose(1, 0)
        elif self.params.mid_struct == 'tener':
            feats = self.tener(sequence_output, attention_mask).transpose(1, 0)
        elif self.params.mid_struct == 'rtransformer':
            # sequence_output.shape=(batch_size, 512, 768)  attention_mask.shape = (batch_size, 512)
            feats = self.rtransformer(sequence_output, attention_mask).transpose(1, 0)
        else:
            feats = sequence_output.transpose(0, 1)
            # print('mid_struct must in [bilstm idcnn tener rtransformer]')
        feats = feats.transpose(0, 1)  # [batch, seq_len, hidden_size]
        # MRC
        start_logits = self.start_outputs(feats).squeeze(-1)  # [batch, seq_len]
        end_logits = self.end_outputs(feats).squeeze(-1)  # [batch, seq_len]
        # Mask
        start_logits = self.mask_logits(start_logits, attention_mask)
        end_logits = self.mask_logits(end_logits, attention_mask)

        start_pre = torch.sigmoid(start_logits)  # batch x seq_len
        end_pre = torch.sigmoid(end_logits)

        # 任务二：是否有答案
        cls_logits = self.task2(pooled_output)  # batch_size,1
        cls_pre = torch.sigmoid(cls_logits)

        if start_positions is not None:
            # total scores
            start_loss = imbalanced_qa_loss(start_pre, start_positions, inbalance_rate=10)
            end_loss = imbalanced_qa_loss(end_pre, end_positions, inbalance_rate=10)
            Mrc_loss = start_loss + end_loss
            # loss 独立于模型之外放入cuda
            CLS_loss = nn.BCELoss()(cls_pre, cls_label.unsqueeze(-1).float())
            # 重点是定位正确，所以CLS_loss的比重大，而不是mrc_loss和cls_loss平分
            # l1正则化
            # l1_regularization = nn.L1Loss(reduction='sum')
            # loss = ... #standard cross-entropy loss
            # for param in model.parameters():
            #     loss += torch.sum(torch.abs(param))
            # loss.backward()
            return 0.01 * Mrc_loss + 0.99 * CLS_loss

        else:
            return start_pre, end_pre, cls_pre


def imbalanced_qa_loss(probs, labels, inbalance_rate=None):
    """
    @param logits: sigmoid数值， (batch_size, seq_len)
    @param labels: (batch_size,seq_len)
    @param num_labels: one-hot标签类别
    @param damping_ratio: 不平衡比>1，加大“1”样本的学习难度
    @return: loss
    """
    if inbalance_rate != None:
        weight = labels * (inbalance_rate - 1) + 1
        loss_func = nn.BCELoss(weight=weight.float()).to(device)
        loss = loss_func(probs, labels.float())
    else:
        loss_func = nn.BCELoss().to(device)
        loss = loss_func(probs, labels.float())
    return loss


if __name__ == '__main__':
    from NEZHA.model_nezha import BertConfig
    from config import Config
    import os

    params = Config()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.model_path, 'bert_config.json'))
    model = BertForQA(bert_config, params=params)
    for n, p in model.named_parameters():
        print(n)
