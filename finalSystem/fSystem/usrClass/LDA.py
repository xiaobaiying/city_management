'''
   LDA,主题结果输出
   '''

import os
import pickle
import numpy as np

model_dir =r'E:\技术和知识\面试准备\项目\城市管理案\finalSystem\data\lda.model'
url = "E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/"

class LDA:
    def get_data(self,fileName):
        fr = open(url +fileName,'r',encoding='utf-8')
        data_list = []
        i = 0
        for line in fr.readlines():
            word = line.strip().split()
            data_list.append(word)
            i += 1
        fr.close()
        return data_list


    def post_lda(self, topic_num,fileName):
        from gensim import corpora, models, matutils
        if os.path.exists(model_dir):
            lda = models.LdaModel.load(model_dir)
        else:

            data_list = self.get_data(fileName)

            print ("lda")
            dic = corpora.Dictionary(data_list)  # 构造词典
            corpus = [dic.doc2bow(text) for text in data_list]  # 每个text 对应的稀疏向量
            tfidf = models.TfidfModel(corpus)  # 统计tfidf

            print ("lda")
            corpus_tfidf = tfidf[corpus]  # 得到每个文本的tfidf向量，稀疏矩阵
            lda = models.LdaModel(corpus_tfidf, id2word=dic, num_topics=topic_num)
            # perplexity = lda.log_perplexity(corpus,8700)
            # print perplexity
            # lda.save("lda.model")

        corpus_lda = lda[corpus_tfidf]  # 每个文本对应的LDA向量，稀疏的，元素值是隶属与对应序数类的权重
        print ("lda")
        fro2 = open(url + "recentTopics.txt", "a",encoding='utf-8')

        # fro3.write("topic_num:"+str(topic_num)+"\n")
        result = ""
        for i in range(0, lda.num_topics):
            r = lda.print_topic(i)
            s = r + "\n"
            fro2.write(s)
            result += "topic" + str(i) + ":" + r + "\n"

        fro2.close()

        return {"result": "success", "LDA": result}
    """
    根据上一环节总结出的关键词，对本轮聚类数据进行处理，将含有上一轮关键词的案件信息保存到另一文件中
    """
    def SepCases(self,fileName,wordsCasesFile,wordsNotCasesFile,wordList):
       if not os.path.exists(url + fileName):
           print("please excute split.py first\n")
       else:
           # wordList
           words = wordList.split('\n')

           #本轮要删除的关键词案件信息

           frp = open(url+fileName,'r',encoding='utf-8')
           fonot = open(url + wordsNotCasesFile,"w",encoding='utf-8')
           fo=open(url+wordsCasesFile,'w',encoding='utf-8')
           for line in frp.readlines():
               line = line.strip()
               e = 0
               for item in words:
                   temp = item.split()
                   te = 1
                   for t in temp:
                       if t not in line:
                           te = 0 #这个词系列不在这行数据中，跳出循环，查找下一个词系列是否在line中
                           break
                   if te ==1:#te = 1 代表这个词系列在line中，这个line应该删除，所以赋值e = 1 跳出循环检查下一条line
                       e = 1
                       break

               if e == 1:
                   fo.write(line+'\n')
               else:
                   fonot.write(line + '\n')
           fo.close()
           frp.close()
           fonot.close()
           return {"result": "success"}
    def getTopicWords(self,fileName):
        f=open(url+fileName,'r',encoding='utf-8')
        lines = f.readlines()
        wordList = [item.split('+') for item in lines]
        for i in range(len(wordList)):
            wordList[i]=[item.split("*\"")[1].split("\"")[0] for item in wordList[i]]
        return wordList

    def calCoWords(self):
        if os.path.exists(url+"recentTopics.txt"):
            topicwordList=self.getTopicWords(url+"recentTopics.txt")
            f=open(url+"cutWords.txt",'r',encoding='utf-8')
            textList=f.readlines()

            dict={}
            for item in topicwordList:
                for i in range(len(item)):
                    for j in range(i+1,len(item)):
                        key=item[i]+'+'+item[j]
                        if key not in dict:
                            key=item[j]+'+'+item[i]
                        s=0
                        for text in textList:
                            if item[i] in text and item[j] in text:
                                s+=1
                        if key in dict:
                            dict[key]+=s
                        else:
                            dict[key]=s

            dict_list=sorted(dict.items(),key = lambda x:x[1],reverse = True)
            result=''
            k=0
            for item in dict_list:
                if item[1]>300:
                    continue
                result+=item[0]+":"+str(item[1])+'                '
                k+=1
                if item[1]<=5:
                    break
            return {"result":"success","cowords":result}


    def getKMeansFeatures(self,wordList):
        features=[]
        if os.path.exists(url+'vecDict.pkl'):
            f = open(url + 'vecDict.pkl', 'rb')
            vecDict = pickle.load(f)
            f.close()
            for i in range(len(wordList)):
                for item in wordList[i]:
                    if item in vecDict:
                        features.append(vecDict[item])
                    else:
                        features.append(np.zeros(200))

        else:
            print('There is no wordVectors!')
        return features
    def KMeans(self,fileName,n):
        from sklearn.cluster import KMeans
        from sklearn.externals import joblib

        wordList=self.getTopicWords(fileName)
        feature=self.getKMeansFeatures(wordList)

        # 调用kmeans类
        clf = KMeans(n_clusters=int(n))
        s = clf.fit(feature)
        # 保存模型
        joblib.dump(clf, url + 'km.pkl')

        # 载入保存的模型
        # clf = joblib.load('c:/km.pkl')

        # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
        '''
        for i in range(5, 30, 1):
            clf = KMeans(n_clusters=i)
            s = clf.fit(feature)
            print(i, clf.inertia_)
        '''
        dic={}
        k=0
        for i in range(len(wordList)):
            for item in wordList[i]:
                dic[item]=clf.labels_[k]
                k+=1

        sorted(dic.items(), key=lambda x: x[1])
        res=['类别'+str(i)+':'for i in range(int(n))]
        for key in dic.keys():
            res[dic[key]]+=key+' '
        res=[item +'</br>' for item in res]
        result=''
        for item in res:
            result+=item

        return{'result':result}



