# -*- coding: utf-8 -*-
# @Description  : 关键词关键句提取
# @file         :新闻信息提取
# @Software     :PyCharm
# @Author       :chenyancan
# @Email        :ican22@foxmail.com


# !pip install bs4, requests, wordcloud, textrank4zh

import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize
from wordcloud import WordCloud
import jieba.posseg as pseg
import re
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pandas as pd

def remove_stop_words(f):
    stop_words = [line.strip('\n') for line in open('E:/8.BI名企课程/12.主题模型与文本表征/week3优秀作业（1）/stop_words.txt', encoding='utf-8').readlines()]
    # print('stop_words',stop_words)
    for stop_word in stop_words:
        f = f.replace(stop_word, '')
    return f

import jieba
filepath=''
def create_word_cloud(f,filename='wordcloud'):
    f = remove_stop_words(f)
    seg_list = jieba.cut(f)
    cut_text = ' '.join(seg_list)
    wc = WordCloud(max_words=100,
                    width=2000,
                    height=1200,
                    # font_path='C:\Windows\Fonts\STZHONGS.TTF',  
                    font_path='STZHONGS.TTF',  # 若是有中文的话，设置字体之后才会出现汉子，不然会出现方框
    )
    wordcloud = wc.generate(cut_text)
    wordcloud.to_file(filepath + filename + ".jpg")

# 自动阅读新闻
class ReadingNews:
    def __init__(self,url,channel='',sentence_num=3): # 生成几个关键句
        # 初始化关键信息
        self.title =''     # print(readingNews1.title)
        self.author = ''
        self.text = ''
        self.person = ''
        self.site = ''
        self.key_word = []
        self.import_sentence = []

        self.sentence_num = sentence_num
        html = requests.get(url, timeout=1000)
        content = html.content
        # print(content)
        # 通过content创建BeautifulSoup对象
        soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')

        if channel == 'weixin':
            #主题
            title = str(soup.find_all('h2', id='activity-name')[0])
            title = re.sub(r"[\n                    ]+", " ", title)
            title = re.findall(r"<h2.*?>(.*?)</h2>", title)
            self.title = title[0].replace(' ', '')
            #作者
            author = str(soup.find_all('a', id='js_name')[0])
            author = re.sub(r"[\n                    ]+", " ", author)
            author = re.findall(r'<a.*?>(.*?)</a>', author)
            self.author = author[0].replace(' ', '')

        text = soup.get_text()
        # print(text)
        words = pseg.lcut(text)

        # 人物合集（nr），地点合集（ns）
        self.person = {word for word, flag in words if flag == 'nr'}
        self.site = {word for word, flag in words if flag == 'ns'}
        # print('新闻中人物', self.person)
        # print('新闻中地点', self.site)
        self.text = re.sub('[^\u4e00-\u9fa5。，！：、]{3,}', '', text)
        self.text = re.sub('{IMG:.?.?.?}', '', self.text)
        self.text = re.sub('<!--IMG_\d+-->', '', self.text)
        self.text = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', self.text)  # 过滤网址
        self.text = re.sub('<a[^>]*>', '', self.text).replace("</a>", "")  # 过滤a标签
        self.text = re.sub('<P[^>]*>', '', self.text).replace("</P>", "")  # 过滤P标签
        self.text = re.sub('<strong[^>]*>', ',', self.text).replace("</strong>", "")  # 过滤strong标签
        self.text = re.sub('<br>', ',', self.text)  # 过滤br标签
        self.text = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', self.text).replace("()", "")  # 过滤www开头的网址
        self.text = re.sub('\s', '', self.text)   # 过滤不可见字符
        # print(self.text)
        self.cal_key_words()
        self.cal_key_sentence()
    def cal_key_words(self):
        # 输出关键词，设置文本小写，窗口为2
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=self.text, lower=True, window=2) # window=2类似word2vec中的window_size，表示n-gram
        # print('关键词：')
        for item in tr4w.get_keywords(num=20, word_min_len=1): # num=20表示提取关键词个数 word_min_len:获取最重要的num个长度大于等于word_min_len的关键词
            # print(item.word, item.weight)
            self.key_word.append(item.word)
    def cal_key_sentence(self):
        # 输出重要的句子
        tr4s = TextRank4Sentence()
        tr4s.analyze(text=self.text, lower=True, source='all_filters')
        # print('摘要：')
        # 重要性较高的3个句子
        for item in tr4s.get_key_sentences(num=self.sentence_num): # sentence_num是生成关键句的个数
            # index是语句在文本中位置，weight表示权重
            # print('item.index, item.weight, item.sentence',item.index, item.weight, item.sentence)
            # self.import_sentence.append(str(item.index) + ' ' +item.sentence)
            self.import_sentence.append(item.sentence)
        
        # print('self.import_sentence',self.import_sentence[0][2:] + self.import_sentence[1][2:] + self.import_sentence[2][2:])
        print('self.import_sentence', self.import_sentence)
    def create_word_cloud(self):
        #生成词云
        title = re.findall(r"[\u4e00-\u9fa5]+", self.title)
        title = ''.join([x for x in title])
        create_word_cloud(self.text, filename=title)
    def get_list(self):
        result = []
        result.append(self.title)
        result.append(self.author)
        result.append(self.person)
        result.append(self.site)
        result.append(self.key_word)
        result.append(self.import_sentence)
        result.append(self.text)
        return result

def get_news_info(urls):
    result_list = []
    for i,url in enumerate(urls):
        readingNews = ReadingNews(url, channel='weixin')
        l = readingNews.get_list()
        result_list.append(l)
    result_df = pd.DataFrame(result_list)
    result_df.columns = ['标题','作者','人物','地点','关键词','关键句','原文文本']
    return result_df


if __name__ == "__main__":
    #内容提取和词云
    readingNews1 = ReadingNews('https://mp.weixin.qq.com/s/mfubaYRypuMFSMxTBZNOJg', channel='weixin')
    print(readingNews1.key_word)
    print(readingNews1.import_sentence)
    # BERT的输入 = 所有关键句 + 所有关键词 + 新闻标题
    # print('all_key_text', readingNews1.title + readingNews1.import_sentence[0][2:] + readingNews1.import_sentence[1][2:] + readingNews1.import_sentence[2][2:])
    readingNews1.create_word_cloud()
    # print(readingNews1.title)
    # 要查询的网页(微信)
    urls = ['https://mp.weixin.qq.com/s/mfubaYRypuMFSMxTBZNOJg','https://mp.weixin.qq.com/s/GT2QIvz5Cy9Zv3fYCwMjQQ',
            'https://mp.weixin.qq.com/s/9bz7h84jgDBKKKJ_QLSxrA',]
    result = get_news_info(urls)
    print(result)