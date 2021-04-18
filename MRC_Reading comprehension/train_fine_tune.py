# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 22:22
# @Author  : chenyancan
# @Email   : ican22@foxmail.com
# @File    : train_fine_tune.py
# @Software: PyCharm

import os
import time
from tqdm import tqdm
import torch
from config import Config
import random
from snippts import load_checkpoint
from sklearn.metrics import f1_score, classification_report
import logging
from NEZHA.model_nezha import BertConfig
from NEZHA import nezha_utils
from model import BertForQA
from snippts import PGD, FGM  # 对抗
from utils import DataIterator
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, RobertaConfig, AlbertConfig
from optimization import BertAdam
from predict import refind_answer
from rouge import Rouge

# gpu_id = 1
# # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
result_data_dir = Config().result_file
# print('GPU ID: ', str(gpu_id))
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Batch Size: ', Config().batch_size)
print('Use original bert', Config().pretrainning_model)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# n_gpu = torch.cuda.device_count()

# 固定每次结果
seed = 156421
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# if n_gpu > 0:
#     torch.cuda.manual_seed_all(seed)  # 我的理解是：确保每次实验结果一致，不设置实验的情况下准确率这些指标会有波动，因为是随机


def list2ts2device(target_list):
    """把utils的数据写入到gpu"""
    target_ts = torch.from_numpy(np.array(target_list)).long()
    return target_ts.to(device)


def train(train_iter, test_iter, config):
    """"""
    # Prepare model
    # Prepare model
    # reload weights from restore_file if specified
    if config.pretrainning_model == 'nezha':
        Bert_config = BertConfig.from_json_file(config.bert_config_file)
        model = BertForQA(config=Bert_config, params=config)
        nezha_utils.torch_init_model(model, config.bert_file)
    elif config.pretrainning_model == 'albert':
        Bert_config = AlbertConfig.from_pretrained(config.model_path)
        model = BertForQA.from_pretrained(config.model_path, config=Bert_config)
    else:
        Bert_config = RobertaConfig.from_pretrained(config.bert_config_file, output_hidden_states=True)
        model = BertForQA.from_pretrained(config=Bert_config, params=config,
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
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n or 'electra' in n]  # nezha的命名为bert
    # middle model param
    param_middle = [(n, p) for n, p in param_optimizer if
                    not any([s in n for s in ('bert', 'crf', 'electra', 'albert')]) or 'dym_weight' in n]
    # crf param
    # 不进行衰减的权重
    no_decay = ['bias', 'LayerNorm', 'dym_weight', 'layer_norm']
    # 将权重分组
    optimizer_grouped_parameters = [
        # pretrain model param
        # 衰减
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': config.decay_rate, 'lr': config.embed_learning_rate
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
    ]
    num_train_optimization_steps = train_iter.num_records // config.gradient_accumulation_steps * config.train_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=config.warmup_proportion, schedule="warmup_linear",
                         t_total=num_train_optimization_steps)
    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", config.batch_size)
    logger.info("  Num epochs = %d", config.train_epoch)
    logger.info("  Learning rate = %f", config.learning_rate)

    cum_step = 0
    best_acc = 0.0
    timestamp = str(int(time.time()))
    if device != 'cpu':
        out_dir = os.path.abspath(
            os.path.join(config.save_model, "runs_" + str(gpu_id), timestamp))
    if device == 'cpu':
        out_dir = os.path.abspath(
            os.path.join(config.save_model, "runs_" + str('cpu_0'), timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing to {}\n".format(out_dir))
    num_parameters = sum(torch.numel(param) for param in model.parameters())
    print('total number of model parameters', num_parameters)

    for i in range(config.train_epoch):
        model.train()
        for input_ids_list, input_mask_list, segment_ids_list, start_list, end_list, uid_list, \
            answer_list, text_list, querylen_list, mapping_list, cls_list in tqdm(
            train_iter):
            # 转成张量
            loss = model(list2ts2device(input_ids_list), list2ts2device(input_mask_list),
                         list2ts2device(segment_ids_list),
                         list2ts2device(start_list), list2ts2device(end_list), list2ts2device(cls_list))
            # if n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu.
            # 梯度累加
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            if cum_step % 10 == 0:
                format_str = 'step {}, loss {:.4f} lr {:.5f}'
                print(
                    format_str.format(
                        cum_step, loss, config.learning_rate)
                )
            if config.flooding:
                loss = (loss - config.flooding).abs() + config.flooding  # 让loss趋于某个值收敛
            loss.backward()  # 反向传播，得到正常的grad
            if (cum_step + 1) % config.gradient_accumulation_steps == 0:
                # performs updates using calculated gradients
                optimizer.step()
                model.zero_grad()
            cum_step += 1
        acc = set_test(model, test_iter, epoch=i)
        # lr_scheduler学习率递减 step
        print('dev set : step_{},ACC_{}'.format(cum_step, acc))
        if acc > best_acc:
            # Save a trained model
            best_acc = acc
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                os.path.join(out_dir, 'model_{:.4f}_{}.bin'.format(acc, str(cum_step))))
            torch.save(model_to_save, output_model_file)


def set_test(model, test_iter, epoch):
    if not test_iter.is_test:
        test_iter.is_test = True
    model.eval()
    cls_pred = []
    start_pred = []
    end_pred = []
    cls_true = []
    start_true = []
    end_true = []
    rouge = Rouge() # 通过rouge控制答案句不被分在2个段落中
    all_uid_list, start_prob_list, end_prob_list, questionlen_list, allmaping_list, context_list, cls_prob_list = [], [], [], [], [], [], []
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, start_list, end_list, uid_list, \
                answer_list, text_list, querylen_list, mapping_list, cls_list in tqdm(test_iter):

            y_preds = model(list2ts2device(input_ids_list), list2ts2device(input_mask_list),
                            list2ts2device(segment_ids_list))
            y_maps = [torch.where(moid > 0.5, torch.ones_like(moid), torch.zeros_like(moid)) for moid in y_preds]
            pred_s, pred_e, pred_c = (p.int().detach().cpu().tolist() for p in y_maps)
            prob_s, prob_e, prob_c = (p.detach().cpu().tolist() for p in y_preds)

            cls_true.extend(cls_list)
            cls_pred.extend(pred_c)
            [start_true.extend(t) for t in start_list]
            [end_true.extend(t) for t in end_list]
            [start_pred.extend(t) for t in pred_s]
            [end_pred.extend(t) for t in pred_e]
            if (epoch + 1) % 5 == 0:
                all_uid_list.extend(uid_list)
                start_prob_list.extend(prob_s)
                end_prob_list.extend(prob_e)
                questionlen_list.extend(querylen_list)
                allmaping_list.extend(mapping_list)
                context_list.extend(text_list)
                cls_prob_list.extend(pred_c)
        start_acc = f1_score(start_true, start_pred)
        end_acc = f1_score(end_true, end_pred)
        print('Start F1:', f1_score(start_true, start_pred))
        print('End F1:', f1_score(end_true, end_pred))
        print('Cls ACC:', accuracy_score(cls_true, cls_pred))
        if (epoch + 1) % 5 == 0:
            dev_like_test = pd.read_csv(config.processed_data + 'dev_like_test.csv')[:10]
            true_answer = list(dev_like_test['answer'])
            pred_answer, C = refind_answer(dev_like_test, all_uid_list, start_prob_list, end_prob_list,
                                           questionlen_list,
                                           allmaping_list,
                                           context_list, cls_prob_list)
            # acc_answer = accuracy_score(answer_true, answer_pred)

            pred_answer = [' '.join(i) for i in pred_answer]
            true_answer = [' '.join(i) for i in true_answer]
            logging.info('Answer Rouge-L:', str(rouge.get_scores(pred_answer, true_answer, avg=True)['rouge-l']))
        return (start_acc + end_acc) / 2


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
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=False,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    train_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'train.csv',
                              config=config, tokenizer=tokenizer)
    dev_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'dev.csv',
                            config=config, tokenizer=tokenizer)
    train(train_iter, dev_iter, config=config)
