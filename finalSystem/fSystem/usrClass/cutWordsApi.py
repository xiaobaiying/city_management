# -*- coding: utf-8 -*-

import numpy as np
from django_ajax.decorators import ajax
from fSystem.usrClass import jieba
import pandas as pd
import math
import re
import os

url = "E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/"
class cutWordsApi:
    def data_process(self,path):
        # 加载停用词
        stopwords = {}.fromkeys([line.rstrip().decode('utf-8') for line in open(url + 'stopwords.txt', 'rb')])
        pattern = re.compile(r'\d+')
        path = url+path
        fd = open(path,"r", encoding='utf-8')
        # 数据处理，重复案件信息清理
        data = pd.DataFrame(fd.readlines(),columns=['description'])
        # data = data.drop_duplicates(['community number','street number','description'])
        data = data.drop_duplicates(['description'])
        # 去重后重新将index排序
        data = data.reset_index(drop=True)
        pattern = re.compile(r'\d+')
        length = len(data.index)
        if os.path.exists(url+"cutWords.txt"):
            os.remove(url+"cutWords.txt")
        fo = open(url+"cutWords.txt", "a+")
        for j in range(0, length):
            text = data.at[j, 'description']
            text = text.strip()
            text = text.replace('\n', ' ')
            # 精确模式分词
            key_list = list(jieba.cut(text, cut_all=False))
            s = ''
            for i in key_list:
                if i not in stopwords and not pattern.match(i):
                    # i = i.encode("utf-8")
                    s += i + ' '
            fo.write(s + "\n")
        fo.close()
        return {'result':'success'}


