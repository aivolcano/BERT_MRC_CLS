# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 10:25
# @Author  : poppy  
# @Email   : szuleipengbin@foxmail.com
# @File    : snippts.py
# @Software: PyCharm
from config import Config
import os
import re
import torch.nn.init as init
import torch.nn as nn

config = Config()


def find_lcsubstr(s1, s2):
    """
    最长公共子串
    """
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]: # 如果相等，则加入现有的公共子串
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长子串及其长度

def bound2oktext(text, max_len, window_size=0):
    """
     截取回推法
    :param text: 过长的文本
    :param max_len: 限制的最大长度
    :window_size :滑动窗口size，防止截断文本
    :return:list-所有切分后的文本列表
    """
    new_test = []
    while text.__len__() >= max_len:
        cut_text = text[:max_len]
        cut_list = list(cut_text)
        cut_list.reverse()
        cur = ''.join(cut_list)
        if re.search('\？|\；|\！|\。', cur) == None:
            try:
                stop_index = re.search('|\，|\、|\：', cur).start()
            except:
                stop_index = 0
        else:
            stop_index = re.search('\？|\；|\！|\。', cur).start()
        cut_text = cut_text[:cut_text.__len__() - stop_index]
        new_test.append(cut_text)

        back_index = max_len - stop_index - window_size  # 滑动窗口size=
        if max_len - stop_index - window_size <= 0:
            back_index = max_len - window_size
        text = text[back_index:]
    res_text = new_test
    res_text.append(text)
    return res_text

def initial_parameter(net, initial_method=None):
    r"""A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model or a List of Pytorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            if len(m.weight.size()) > 1:
                init_method(m.weight.data)
            else:
                init.normal_(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    if isinstance(net, list):
        for n in net:
            n.apply(weights_init)
    else:
        net.apply(weights_init)
def split_text(text, maxlen, split_pat='default', greedy=True):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过maxlen；
             2）所有的子文本的合集要能覆盖原始文本。
    Arguments:
        text {str} -- 原始文本
        maxlen {int} -- 最大长度

    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式
                         非贪婪模式：在满足约束条件下，选择冗余度最小且交叉最为均匀的分割方式

    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表
    """
    if split_pat == 'default':
        #         \xa0|\u3000|\u2002|\u2003
        #         STOPS = (
        #             '\uFF01'  # Fullwidth exclamation mark
        #             '\uFF1F'  # Fullwidth question mark
        #             '\uFF61'  # Halfwidth ideographic full stop
        #             '\u3002'  # Ideographic full stop
        #         )
        STOPS = (
            '\xa0'  # 换行符
            '\u3000'  # 顶格
            '\u2002'  # 空格
            '\u2003'  # 2空格
        )
        split_pat = '([{}]”?)'.format(STOPS)

    if len(text) <= maxlen:
        return [text], [0]
    segs = re.split(split_pat, text)
    sentences = []
    for i in range(0, len(segs) - 1, 2):
        sentences.append(segs[i] + segs[i + 1])
    if segs[-1]:
        sentences.append(segs[-1])
    n_sentences = len(sentences)
    sent_lens = [len(s) for s in sentences]
    alls = []  # 所有满足约束条件的最长子片段
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= maxlen or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        if j == n_sentences - 1:
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            break

    if len(alls) == 1:
        return [text], [0]

    if greedy:  # 贪婪模式返回所有子文本
        sub_texts = [''.join([sentences[i] for i in sub]) for sub in alls]
        starts = [0] + [sum(sent_lens[:i]) for i in range(1, len(alls))]
        return sub_texts, starts
    else:  # 用动态规划求解满足要求的最优子片段集
        DG = {}  # 有向图
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist

        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)

        sub_texts, starts = [''.join([sentences[i] for i in alls[0]])], [0]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(''.join([sentences[i] for i in alls[k]]))
            starts.append(sum(sent_lens[: alls[k][0]]))
            if k == N - 1:
                break

        """newadd"""
        while True:
            sub_texts_ = []
            for text in sub_texts:
                if text.__len__() > maxlen:
                    _text = bound2oktext(text, maxlen, 128)
                    sub_texts_.extend(_text)
                else:
                    sub_texts_.append(text)
            #             raise ValueError
            flag = 0
            for _text in sub_texts_:
                if _text.__len__() > maxlen:
                    flag = 1
                    #                     raise ValueError
                    break
            if flag == 1:
                print('>>>>')
                sub_texts = sub_texts_
            else:
                break
    return sub_texts_, starts
def load_checkpoint(checkpoint, optimizer=True):
    """Loads entire model from file_path. If optimizer is True, loads
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        optimizer: (bool) resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['optim']
    return checkpoint['model']

class FGM():
    """
    对抗训练方法类
    """

    def __init__(self, model):

        self.model = model

        self.backup = {}

    """


    """

    def attack(self, epsilon=1., emb_name=config.embed_name):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name=config.embed_name):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


import torch


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name=config.embed_name, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name=config.embed_name):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(name)
                try:
                    self.grad_backup[name] = param.grad.clone()
                except:
                    self.grad_backup[name] = None
                    pass

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
