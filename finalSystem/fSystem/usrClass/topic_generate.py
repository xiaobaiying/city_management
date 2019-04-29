import pickle
import codecs
import numpy as np
import os
import pandas as pd
import re
import random
from fSystem.usrClass.topicClasses import *
#dataPath = '/root/Pychar/rojects/NER/bi_LSTM_CRF_test/data/'
topicDataPath = 'E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/topicGenerate/'
#topicDataPath = '/root/PycharmProjects/NER/bi_LSTM_CRF_test/data/topicGenerate/'
#找到句子S 中标有Tag的词列表
class topic_generate:
    def getWordsByTag(self,S,Tag):
        BTag = 'B-' + Tag.strip()
        ITag = 'I-' + Tag.strip()
        ETag = 'E-' + Tag.strip()
        wordList = []
        words =[]
        tags=[]
        wts= S.split()
        for item in wts:
            wt = item.strip().split('/')
            if len(wt)==2:
                words.append(wt[0].strip())
                tags.append(wt[1].strip())
        word = ''
        if BTag in tags:
            for i in range(len(tags)):
                if tags[i] == BTag:
                    if len(word)>0 and word not in wordList:
                        wordList.append(word)
                    word = words[i]
                elif tags[i]==ITag or tags[i]==ETag:
                    word +=words[i]
                if i==len(tags)-1 and len(word)>0 and word not in wordList:
                    wordList.append(word)
        return wordList

    #按顺序得到句子中所有有标签的词
    def getTagedWords(self,S):

        wordList = []
        words = []
        tags = []
        wts = S.split()
        for item in wts:
            wt = item.strip().split('/')
            if len(wt) == 2:
                words.append(wt[0].strip())
                tags.append(wt[1].strip())
        word = ''
        Tag=''
        for i in range(len(tags)):
            if tags[i][0] =='B':
                if len(word) > 0 and word not in wordList:
                    wordList.append(word)
                Tag=tags[i][2:]
                word = words[i]
            elif tags[i] == 'I-'+Tag or tags[i] == 'E-'+Tag:
                word += words[i]
            if i == len(tags) - 1 and len(word) > 0 and word not in wordList:
                wordList.append(word)
                Tag=''
        return wordList

    #生成主题
    def getTopic(self,c,S):
        topic =''
        topicTags = {'0':'hasGoods','1':'damage','2':'consult','3':'visit','4':'visit_handle'}
        if topicTags[c] =='hasGoods':
            place= self.getWordsByTag(S,'PUBLIC')
            goods=self.getWordsByTag(S,'GOODS')
            ad = self.getWordsByTag(S,'AD_CER')
            org = self.getWordsByTag(S,'ORG')
            occupy = self.getWordsByTag(S,'OCCUPY')
            p=''
            g=''
            o=''

            if len(place)>0:
                p=place[0]
            if len(occupy)>0:
                o = occupy[0]
            if len(goods)>0:
                for item in goods:
                    g +=item+'、'
            if len(ad)>0:
                for item in ad:
                    g += item + '、'
            g = g.rstrip('、')
            if len(g)==0 and len(org)>0:
                g=org[-1]

            topic = hasGoods(p,o,g).getTopicSentence()
        elif topicTags[c] =='damage':
            public = self.getWordsByTag(S,'PUBLIC')
            goods = self.getWordsByTag(S,'GOODS')
            damage = self.getWordsByTag(S,'DAMAGE')
            occupy = self.getWordsByTag(S,'OCCUPY')
            p=''
            d = ''
            if len(public) > 0:
                for item in public:
                    p += item
            if len(goods) > 0:
                for item in goods:
                    p += item
            if len(damage) > 0:
                for item in damage:
                    d += item
            if len(damage)==0 and len(occupy)>0:
                for item in occupy:
                    d+=item


            topic = Damage(p, d).getTopicSentence()


        elif topicTags[c] =='consult':
            consult = self.getWordsByTag(S,'VISIT_HANDLE')
            certif = self.getWordsByTag(S,'AD_CER')
            con = ''
            cer = ''
            if len(consult)>0:
                for item in consult:
                    if item not in con:

                        con+=item
            if len(certif)>0:
                cer = certif[0]
            topic =Consult(con,cer).getTopicSentence()
        elif topicTags[c]=='visit':
            vs = self.getWordsByTag(S,'VISIT_HANDLE')
            org = self.getWordsByTag(S,'ORG')
            v=''
            o=''
            if len(vs)>0:
                for item in vs:
                    v+=item
            if len(org)>0:
                for item in org:
                    o +=item
            topic= Visit(v,o).getTopicSentence()
        elif topicTags[c] == 'visit_handle':
            vlist = self.getTagedWords(S)
            v=''
            for item in vlist:
                v+=item
            topic = v
        if len(topic)==0 or topic=='。':
            vlist = self.getTagedWords(S)
            v = ''
            for item in vlist:
                v += item
            topic = v
        return topic

    def computTheAccuracy(self,file):
        f=open(file,'r',encoding='utf-8')
        lines=f.readlines()
        if len(lines)%3>0:
            print("file format error!")
            return
        else:
            total={}
            right={}
            for i in range(2,len(lines),3):
                item=lines[i].split('\t')
                if len(item)==2 and item[0].strip().isdigit()and item[1].strip().isdigit():
                    if item[0] not in total.keys():
                        total[item[0]]=1
                    else:
                        total[item[0]]+=1
                    if item[0] not in right.keys():
                        right[item[0]]=0
                    if eval(item[1])==1:
                        right[item[0]]+=1
            s = sum(total.values())
            r=sum(right.values())


            for item in total.keys():
                print('The class '+item+' total has '+str(total[item])+' items, and there is '+str(right[item])
                      +' right items,the precesion is '+str(right[item]/total[item]))
            print('The total num is '+str(s)+',and the total right num is '+str(r)+',the total precesion is '+str(r/s))



    def topicGenerate(self,file,outFile):
        info=[]
        f= open(topicDataPath + 'inputData.txt',encoding='utf-8')
        fout = open(topicDataPath + 'sentenceAndTopics.txt','w',encoding='utf-8')
        line =f.readline()
        while True:
            if line:
                if '\t' in line:
                    cs= line.split('\t')
                    c=cs[0]
                    S=cs[-1]
                    topic = self.getTopic(c.strip(), S.strip())
                    if topic:
                        fout.write(line)
                        fout.write('Topic:'+topic+'\n')
                        info.append([line,topic])
            else:
                break
            line=f.readline()
        f.close()
        fout.close()
        return{"result":"success","info":info}

#main(topicDataPath+'inputData.txt',topicDataPath+'sentenceAndTopics.txt')
#computTheAccuracy(topicDataPath+'sentenceAndTopics.txt')