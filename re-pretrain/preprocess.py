# -*- coding: utf-8 -*-
# @Time    : 2021/1/12
# @Author  : chenyancan
# @Email   : ican22@foxmail.com
# @Software: PyCharm
import pandas as pd
import re
from tqdm import tqdm
from config import Config


def stop_words(x):
    try:
        x = x.strip()
    except:
        return ''
    x = re.sub('{IMG:.?.?.?}', '', x)
    x = re.sub('<!--IMG_\d+-->', '', x)
    x = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x)  # 过滤网址
    x = re.sub('<a[^>]*>', '', x).replace("</a>", "")  # 过滤a标签
    x = re.sub('<P[^>]*>', '', x).replace("</P>", "")  # 过滤P标签
    x = re.sub('<strong[^>]*>', ',', x).replace("</strong>", "")  # 过滤strong标签
    x = re.sub('<br>', ',', x)  # 过滤br标签
    # 过滤www开头的网址
    x = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x).replace("()", "")
    x = re.sub('\s', '', x)  # 过滤不可见字符
    x = re.sub('Ⅴ', 'V', x)

    # 删除奇怪标点
    for wbad in additional_chars:
        x = x.replace(wbad, '')
    return x


def split_text(text, maxlen, greedy=False):
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
    split_pat = '([。！？]”?)'  # 分隔符
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
        if j == n_sentences - 1:  # 加上最后一句长度大于max_len
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

    return sub_texts, starts


def generate_vocab(file_path, all_char_list):
    """
    # 生成词典映射
    :param file_path:字典保存路径
    :param all_char_list: 包含所有字的list
    :return:
    """
    w2i = {"[PAD]": 0, "[CLS]": 1, "[MASK]": 2, "[SEP]": 3}
    count = 4
    for char in tqdm(all_char_list):
        if char not in w2i.keys():
            w2i[char] = count
            count += 1
    w2i["[UNK]"] = len(w2i)
    with open(file_path + 'vocab.txt', 'w', encoding='utf-8') as fw:
        for key in tqdm(w2i.keys()):
            fw.write(key + '\n')
    print('字典长度为:', len(w2i))


if __name__ == '__main__':
    """some params"""
    split_len = 256  # 长文档切分成所需的预训练语料 句子全覆盖：保留更多信息
    config = Config()
    data_path = config.source_data_path
    # corpus_path = '/home/wangzhili/data/ccf_emotion/'
    corpus_path = './'
    # 加载数据
    unlabeled_df = pd.read_csv(data_path + 'nCoV_900k_train.unlabled.csv', encoding='utf_8_sig')
    # 清洗数据
    additional_chars = set()
    for t in list(unlabeled_df['微博中文内容']): # 取出需要预训练的语料
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(t)))
    print('文中出现的非中英文的数字符号：', additional_chars)
    # 一些需要保留的符号
    extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
    print('保留的标点:', extra_chars) # 用于切分句子
    additional_chars = additional_chars.difference(extra_chars)
    unlabeled_df['微博中文内容'] = unlabeled_df['微博中文内容'].apply(stop_words)

    # 生成bert预训练语料文档
    content_text = unlabeled_df['微博中文内容'].tolist()
    corpus_list = []
    all_char_list = []  # 字表
    for doc in tqdm(content_text):
        if len(doc) >= split_len:
            texts_list, _ = split_text(text=doc, maxlen=split_len, greedy=False)
            for text in texts_list:
                all_char_list.extend(text)
                corpus_list.append(text)
        else:
            corpus_list.append(doc)
            all_char_list.extend(doc)  # 加入每一个字
        corpus_list.append('')  # 不同文档的分隔符
    corpus_list = [corpus + '\n' for corpus in corpus_list]
    with open(corpus_path + '{}_corpus.txt'.format(split_len), 'w') as f:
        f.writelines(corpus_list)

    # # 序列任务从零开始训练时可能需要生成字典，但大部分情况下可以复用google_bert提供的字典。
    # #if your need new vocab ,run it!
    # generate_vocab(corpus_path,all_char_list)
    row_len_list = unlabeled_df['微博中文内容'].apply(len).tolist()
    count_64 = []
    count_128 = []
    count_256 = []
    count_384 = []
    count_512 = []
    count_1024 = []

    for len_l in tqdm(row_len_list):
        if len_l <= 64:
            count_64.append(len_l)
        elif 64 < len_l <= 128:
            count_128.append(len_l)
        elif 128 < len_l <= 256:
            count_256.append(len_l)
        elif 256 < len_l < 384:
            count_384.append(len_l)
        elif 384 <= len_l < 512:
            count_512.append(len_l)
        else:
            count_1024.append(len_l)

    row_len = len(row_len_list)
    print(len(count_64) / row_len)
    print(len(count_128) / row_len)
    print(len(count_256) / row_len)
    print(len(count_384) / row_len)
    print(len(count_512) / row_len)
    print(len(count_1024) / row_len)
