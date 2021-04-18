import numpy as np
# from transformers import BertTokenizer
import transformers
from tqdm import tqdm
from config import Config
import pandas as pd

import os # 使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def find_all(s, sub):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 0:
        return index_list
    else:
        return -1


def load_data(data_file):
    data_df = pd.read_csv(data_file)
    data_df.fillna('', inplace=True) # 填充空值
    # data_df['start'] = data_df['start'].apply(lambda x: eval(x))
    # 标记答案段落
    print(data_df['start'].apply(lambda x: '有答案' if x != -1 else '无答案').value_counts())
    lines = list(zip(list(data_df['q_id']), list(data_df['context']), list(data_df['query']),
                     list(data_df['answer']), list(data_df['score']), list(data_df['start'])))
    return lines


def get_examples(data_file):
    return create_example(load_data(data_file))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, q_id, text, query, answer, score, start):
        self.q_id = q_id # 段落ID
        self.text = uniform_perm(text)  #
        self.query = uniform_perm(query) #问题
        self.answer = uniform_perm(answer) # 答案
        self.score = score  # 召回分数
        self.start = start  # 答案开始的位置


def create_example(lines):
    examples = []
    for line in lines:
        q_id = line[0]
        text = line[1]
        query = line[2]
        answer = line[3]
        score = line[4]
        start = line[5]
        examples.append(
            InputExample(q_id=q_id, text=text, query=query, answer=answer, score=score, start=start))
    return examples


def uniform_perm(text):
    # bert字典无“” ’‘标点，容易丢失边界。
    token_list = ['“', '”', '‘', '’']
    for perm in token_list:
        text = text.replace(perm, '"')
    text.replace('—', '_')
    return text


class DataIterator(object):
    """
    数据迭代器
    """

    def __init__(self, batch_size, data_file, tokenizer, config, is_test=False):
        # 数据文件位置
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.config = config

        # 数据的个数
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.match_token = config.match_token # 匹配答案
        if not self.is_test:
            self.shuffle()
        STOPS = (
            '\uFF01'  # Fullwidth exclamation mark
            '\uFF1F'  # Fullwidth question mark
            '\uFF61'  # Halfwidth ideographic full stop
            '\u3002'  # Ideographic full stop
            '\u3002'
        )
        self.SPLIT_PAT = '([,，{}]”’‘“"!.。！?)'.format(STOPS)
        print("样本个数：", self.num_records)

    def convert_single_example(self, example_idx):
        tokenizer = self.tokenizer
        q_id = self.data[example_idx].q_id
        text = self.data[example_idx].text
        query = self.data[example_idx].query
        answer = self.data[example_idx].answer
        score = self.data[example_idx].score
        start_list = self.data[example_idx].start
        config = self.config
        ntokens = []
        segment_ids = []

        """得到input的token-----start-------"""
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # 得到问题的token
        """question_token"""
        q_tokens = tokenizer.tokenize(query)  #
        # 把问题的token加入至所有字的token中
        for i, token in enumerate(q_tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(1)
        """question_token"""
        query_len = len(ntokens)
        # 答案召唤匹配
        text_token = self.match_token._tokenize(text)
        mapping = self.match_token.rematch(text, text_token)
        if [] in mapping:
            print(text_token, text)
        # token后的start&&end
        start_pos, end_pos, cls = [0] * config.sequence_length, [0] * config.sequence_length, 0
        if start_list != -1:
            for start in [start_list]:
                """token后答案实际位置"""
                answer_token = tokenizer.tokenize(answer)
                pre_answer_len = len(tokenizer.tokenize(text[:start]))
                start_ = pre_answer_len + len(ntokens)
                end_ = start_ + len(answer_token) - 1
                if end_ <= config.sequence_length - 1:
                    start_pos[start_] = 1
                    end_pos[end_] = 1
            cls = 1
        for i, token in enumerate(text_token):
            ntokens.append(token)
            segment_ids.append(1)
        if ntokens.__len__() >= config.sequence_length - 1:
            ntokens = ntokens[:(config.sequence_length - 1)]
            segment_ids = segment_ids[:(config.sequence_length - 1)]
        ntokens.append("[SEP]")
        segment_ids.append(0)
        """得到input的token-------end--------"""
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * (len(input_ids))  # SEP也当作padding，mask
        while len(input_ids) < config.sequence_length:
            # 不足时补零
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            ntokens.append("**NULL**")
        assert len(input_ids) == config.sequence_length
        assert len(segment_ids) == config.sequence_length
        assert len(input_mask) == config.sequence_length
        """token2id ---end---"""
        return input_ids, input_mask, segment_ids, start_pos, end_pos, q_id, answer, text, query_len, mapping, cls, score

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        start_list = []
        end_list = []
        uid_list = []
        answer_list = []
        text_list = []
        querylen_list = []
        mapping_list = []
        cls_list = []
        score_list = []
        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            input_ids, input_mask, segment_ids, start_pos, end_pos, q_id, answer, text, query_len, mapping, cls, score = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            start_list.append(start_pos)
            end_list.append(end_pos)
            uid_list.append(q_id)
            answer_list.append(answer)
            text_list.append(text)
            querylen_list.append(query_len)
            mapping_list.append(mapping)
            cls_list.append(cls)
            score_list.append(score)
            num_tags += 1
            self.idx += 1
            if self.idx >= self.num_records:
                break

        return input_ids_list, input_mask_list, segment_ids_list, start_list, end_list, uid_list, \
               answer_list, text_list, querylen_list, mapping_list, cls_list


if __name__ == '__main__':
    config = Config()
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=False,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    dev_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'dev.csv',
                            config=config, tokenizer=tokenizer)
    for input_ids_list, input_mask_list, segment_ids_list, start_list, end_list, uid_list, \
        answer_list, text_list, querylen_list, mapping_list, cls_list in tqdm(dev_iter):
        print(answer_list)
