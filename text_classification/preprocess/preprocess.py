import pandas as pd
from config import Config

config = Config()

train_df = pd.read_csv(config.base_dir + 'train.csv', encoding='utf8')
dev_df = pd.read_csv(config.base_dir + 'dev.csv', encoding='utf8')

# 过长文本处理：固定句子长度
def cal_text_len(row):
    row_len = len(row)
    if row_len < 256:
        return 256
    elif row_len < 384:
        return 384
    elif row_len < 512:
        return 512
    else:
        return 1024


train_df['text_len'] = train_df['text'].apply(cal_text_len)
dev_df['text_len'] = dev_df['text'].apply(cal_text_len)
print(train_df['text_len'].value_counts())
print(dev_df['text_len'].value_counts())
print('-------------------')

# 提取核心内容： 标题 + 结论 = 一篇新闻文本
def merge_text(text):
    if len(text) < 512:
        return text
    else:
        return text[:128] + text[-382:]

# TextRank召回top-k个核心句子
# !pip install textrank4zh
from textrank4zh import TextRank4Sentence
def key_text(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    import_sentence = []
    for item in tr4s.get_key_sentences(num=3):  # sentence_num是生成关键句的个数
        # index是语句在文本中位置，weight表示权重
        # print('item.index, item.weight, item.sentence',item.index, item.weight, item.sentence)
        # self.import_sentence.append(str(item.index) + ' ' +item.sentence)
        import_sentence.append(item.sentence)
    # key_sentence = [i.sentence for i in tr4s.get_key_sentences(num=3)] # num生成关键句的个数
    # 核心内容是 标题 + top-3文本本体关键句
    key_sentences = import_sentence[0][2:] + import_sentence[1][2:] + import_sentence[2][2:]
    # 过长文本截断
    if len(key_sentences) < 512:
        return key_sentences
    else:
        return key_sentences[:512]

#  取文本段前128与后384作为整体的文本
if config.TextRank:
    train_df['sentence'] = train_df['text'].apply(key_text)
    dev_df['sentence'] = dev_df['text'].apply(key_text)
else:
    train_df['sentence'] = train_df['text'].apply(merge_text)
    dev_df['sentence'] = dev_df['text'].apply(merge_text)

train_df['text_len'] = train_df['sentence'].apply(cal_text_len)
dev_df['text_len'] = dev_df['sentence'].apply(cal_text_len)

print(train_df['text_len'].value_counts())
print(dev_df['text_len'].value_counts())

label_list = config.label_list # label_list = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']


def make_label(label):
    return label_list.index(label)


train_df['num_label'] = train_df['label'].apply(make_label)
dev_df['num_label'] = dev_df['label'].apply(make_label)

train_df[['text', 'sentence', 'label', 'num_label']].to_csv(config.base_dir + 'train.csv', encoding='utf-8')
dev_df[['text', 'sentence', 'label', 'num_label']].to_csv(config.base_dir + 'dev.csv', encoding='utf-8')

# if __name__ == '__main__':
