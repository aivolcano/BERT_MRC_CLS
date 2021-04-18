# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 22:22
# @Author  : chenyancan
# @Email   : ican22@foxmail.com
# @File    : train_fine_tune.py
# @Software: PyCharm

import os
import time
from tqdm import tqdm
import torch
from config import Config
from snippts import load_checkpoint
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import logging
from NEZHA.model_nezha import BertConfig
from NEZHA import nezha_utils
from model import BertForCLS
from utils import DataIterator
import numpy as np
from transformers import BertTokenizer, RobertaConfig, AlbertConfig
from optimization import BertAdam

torch.backends.cudnn.enabled = False # True 如果网络的输入数据维度或类型上变化不大，设置 True 可以增加运行效率
torch.backends.cudnn.benchmark = True # False

# gpu_id = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# result_data_dir = Config().result_file
# print('GPU ID: ', str(gpu_id))
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Batch Size: ', Config().batch_size)
print('Use original bert', Config().use_origin_bert)
print('Use avg pool', Config().is_avg_pool)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
device = 'cpu'
# # 固定每次结果
# seed = 156421
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# if n_gpu > 0:
#     torch.cuda.manual_seed_all(seed)  # 我的理解是：确保每次实验结果一致，不设置实验的情况下准确率这些指标会有波动，因为是随机


def list2ts2device(target_list):
    """把utils的数据写入到gpu"""
    target_ts = torch.from_numpy(np.array(target_list))
    target_ts = target_ts.to(torch.int64) # 否则报错
    return target_ts.to(device)


def train(train_iter, test_iter, config):
    """"""
    # Prepare model
    # Prepare model
    # reload weights from restore_file if specified
    if config.pretrainning_model == 'nezha':
        Bert_config = BertConfig.from_json_file(config.bert_config_file)
        model = BertForCLS(config=Bert_config, params=config) # lr=1e-4
        nezha_utils.torch_init_model(model, config.bert_file)
    elif config.pretrainning_model == 'albert':
        Bert_config = AlbertConfig.from_pretrained(config.model_path)
        model = BertForCLS.from_pretrained(config.model_path, config=Bert_config)
    else:
        Bert_config = RobertaConfig.from_pretrained(config.bert_config_file, output_hidden_states=True)
        Bert_config.output_hidden_states = True  # 获取每一层的输出
        Bert_config.output_attentions = True  # 获取每一层attention
        model = BertForCLS.from_pretrained(config=Bert_config, params=config,
                                                           pretrained_model_name_or_path=config.model_path)

    if config.restore_file is not None:
        logging.info("Restoring parameters from {}".format(config.restore_file))
        # 读取checkpoint
        model, optimizer = load_checkpoint(config.restore_file)
    model.to(device)

    """多卡训练"""
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    # optimizer
    # Prepare optimizer
    # fine-tuning
    # 取模型权重
    param_optimizer = list(model.named_parameters()) # hidden_size=768, nhead=12,
    # pretrain model param # torch.Size([768, 768]) torch.Size([3072, 768])
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n and 'head_weight' not in n]  # nezha的命名为bert
    # middle model param # torch.Size([21128, 768]) 768 [2, 768] ...
    param_middle = [(n, p) for n, p in param_optimizer if 'bert' not in n and 'head_weight' not in n]
    param_head=[(n, p) for n, p in param_optimizer if 'head_weight' in n]
    # crf param
    # 不进行衰减的权重
    no_decay = ['bias', 'LayerNorm', 'dym_weight', 'layer_norm']
    # 将权重分组
    optimizer_grouped_parameters = [
        # pretrain model param
        # 衰减
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': config.decay_rate, 'lr': config.embed_learning_rate # BERT的Learning rate
         },
        # 不衰减
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.embed_learning_rate
         },
        # middle model
        # 衰减
        {'params': [p for n, p in param_middle if not any(nd in n for nd in no_decay)],
         'weight_decay': config.decay_rate, 'lr': config.learning_rate
         },
        # 不衰减
        {'params': [p for n, p in param_middle if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.learning_rate
         },
        # head model
        # 衰减
        {'params': [p for n, p in param_head if not any(nd in n for nd in no_decay)],
         'weight_decay': config.decay_rate, 'lr': 1e-1
         },
        # 不衰减
        {'params': [p for n, p in param_head if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-1
         }
    ]
    # 数据总数是否整除epoch
    num_train_optimization_steps = train_iter.num_records // config.gradient_accumulation_steps * config.train_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=config.warmup_proportion,
                         schedule='warmup_linear', #"warmup_cosine" 'warmup_linear' 'warmup_constant'
                         t_total=num_train_optimization_steps, #1.4k
                         weight_decay=0.05) # L2正则系数

    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", config.batch_size)
    logger.info("  Num epochs = %d", config.train_epoch)
    logger.info("  Learning rate = %f", config.learning_rate)

    best_acc = 0.0 # 打擂法 与F1值比较
    cum_step = 0 # 保存模型路径使用
    timestamp = str(int(time.time()))
    if device != 'cpu': # 使用GPUid做为保存路径
        out_dir = os.path.abspath(os.path.join(config.save_model, "runs_" + str(gpu_id), timestamp))
    else: # cpu
        out_dir = os.path.abspath(os.path.join(config.save_model, 'run_' + str('cpu'), timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing to {}\n".format(out_dir))
    for i in range(config.train_epoch):
        model.train()
        for input_ids, input_mask, segment_ids, cls_list, seq_length in tqdm(train_iter):
            # 转成张量
            # input_ids = (1, 512) weights.t().shape = 768 * 10
            loss,_,_ = model(input_ids=list2ts2device(input_ids),
                             token_type_ids=list2ts2device(segment_ids),
                             attention_mask=list2ts2device(input_mask),
                             cls_label=list2ts2device(cls_list))
            # if n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu.
            # 梯度累加
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            if cum_step % 10 == 0:
                # format_str = 'step {}, loss {:.4f} lr {:.5f}' format_str.format()
                print('step {}, loss {:.4f} lr {:.5f}'.format(
                        cum_step, loss, config.learning_rate))
            if config.flooding:
                # 可以增加L1正则
                # regulation_loss_l1 += [torch.sum(abs(param)) for param in model.parameters()]
                # regularion_loss_l1 += [torch.sum(param ** 2) for param in model.parameters()] # L2正则
                # lamba 为正则系数
                loss = (loss - config.flooding).abs() + config.flooding #+ 0.05 * regulation_loss_l1  # 让loss趋于某个值收敛 foolding=0,意味着loss趋于0收敛
            loss.backward()  # 反向传播，得到正常的grad
            if (cum_step + 1) % config.gradient_accumulation_steps == 0:
                # performs updates using calculated gradients
                optimizer.step()
                model.zero_grad()

            cum_step += 1
        F1, P, R = set_test(model, test_iter)
        # lr_scheduler学习率递减 step
        print('dev set : step_{},F1_{},P_{},R_{}'.format(cum_step, F1, P, R))
        # 打擂法：通过F1值保存模型参数
        if F1 > best_acc:
            # Save a trained model
            best_acc = F1
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                os.path.join(out_dir, 'model_{:.4f}_{:.4f}_{:.4f}_{}.bin'.format(F1, P, R, str(cum_step))))
            torch.save(model_to_save, output_model_file)


def set_test(model, test_iter):
    if not test_iter.is_test:
        test_iter.is_test = True
    model.eval()
    with torch.no_grad():
        true_label = []
        pred_label = []
        for input_ids, input_mask, segment_ids, cls_label, seq_length in tqdm(
                test_iter):
            input_ids = list2ts2device(input_ids)
            input_mask = list2ts2device(input_mask)
            segment_ids = list2ts2device(segment_ids)
            y_preds,_ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            cls_pred = y_preds.detach().cpu()
            cls_probs = softmax(cls_pred.numpy())
            # cls_probs = torch.nn.Softmax(y_preds)
            cls_pre = np.argmax(cls_probs, axis=-1) # 返回对应维度的最大元素的位置（下标）
            # cls_pre = torch.argmax(cls_probs, dim=-1)
            true_label += list(cls_label)
            pred_label += list(cls_pre)
        # print(true_label)
        # print(pred_label)
        pred_label = pred_label.detach().cpu().numpy()
        true_label = true_label.detach().cpu().numpy()
        F1 = f1_score(true_label, pred_label, average='micro')
        R = recall_score(true_label, pred_label, average='micro')
        P = precision_score(true_label, pred_label, average='micro')
        logging.info(classification_report(true_label, pred_label))
        return F1, P, R


def softmax(x, axis=1):
    """
    自写函数定义softmax
    :param x:
    :param axis:
    :return:
    """
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case = True
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    train_iter = DataIterator(config.batch_size,
                              data_file=config.base_dir + 'train.csv',
                              use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    dev_iter = DataIterator(config.batch_size, data_file=config.base_dir + 'dev.csv',
                            use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    train(train_iter, dev_iter, config=config)