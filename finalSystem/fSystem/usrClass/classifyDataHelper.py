import pickle
import codecs
import numpy as np
import os
import pandas as pd
import re
from fSystem.usrClass import data_helper

classifyDataPath = 'E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/classifyData/'
dataPath = 'E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/'
def replaceClassTag(file,outfile,oldtag,newtag):
    f=open(file,encoding='utf-8')
    lines=f.readlines()
    newlines=[str(newtag)+x[1:] if x[0]==str(oldtag) else x for x in lines]
    f.close()
    fout = open(outfile,'w',encoding='utf-8')
    fout.writelines(newlines)
    fout.close()
#当句子中只包含某些标签时，则被自动分类
def preClassification2(fileName,outFile,noTagFile):
    ftag = open(classifyDataPath + 'classifyOnlyTags.txt', encoding='utf-8')
    lines = ftag.readlines()
    tags = []
    cnum = []
    for i in range(len(lines)):
        tag_num = lines[i].split('\t')
        if len(tag_num) == 2:
            cnum.append(tag_num[1].strip())
            tags.append(tag_num[0].split('|'))
    ftag.close()
    fout = open(classifyDataPath + outFile, 'w', encoding='utf-8')
    fnoTag = open(classifyDataPath + noTagFile, 'w', encoding='utf-8')
    f = open(classifyDataPath + fileName, 'r', encoding='utf-8')
    line = f.readline()
    while True:
        if line:
            if len(line.split('\t')[0]) == 0:
                linetags=[]
                wordTags=line.strip().split()
                for item in wordTags:
                    t =item.split('/')[1]
                    if '-'in t:
                        t = t.split('-')[1]
                        linetags.append(t)
                c = 0
                for i in range(len(tags)):
                    c = 1
                    differenceSet1=[d for d in set(tags[i])-set(linetags)]
                    differenceSet2 = [d for d in set(linetags) - set(tags[i])]
                    if (len(differenceSet1)==0 or differenceSet1==['O']) and len(differenceSet2)==0:
                        c=1
                        break
                    else:
                        c=0
                if c==1:

                    line = cnum[i] + '\t' + line.strip()
                    fout.write(line + '\n')
                else:
                    fout.write('\t' + line.strip() + '\n')
                    fnoTag.write('\t' + line.strip() + '\n')
            else:
                fout.write(line.strip() + '\n')

            line = f.readline()
        else:
            break
    f.close()
    fout.close()
    fnoTag.close()




#预先自动分类函数
def preClassification(fileName,outFile,noTagFile):
    ftag = open(classifyDataPath+'classifyTag.txt',encoding = 'utf-8')
    lines = ftag.readlines()
    tags=[]
    cnum=[]
    for i in range(len(lines)):
        tag_num=lines[i].split('\t')
        if len(tag_num)==2:
            cnum.append(tag_num[1].strip())
            tags.append(tag_num[0].split())
    ftag.close()
    fout = open(classifyDataPath+outFile,'w',encoding='utf-8')
    fnoTag = open(classifyDataPath+noTagFile,'w',encoding='utf-8')
    f = open(classifyDataPath+fileName,'r',encoding='utf-8')
    line = f.readline()
    while True:
        if line:
            if len(line.split('\t')[0])==0:
                c=0
                for i in range(len(tags)):
                    c=0
                    for item in tags[i]:
                        c = 0
                        ws = item.split('|')

                        for t in ws:

                            if t in line:
                                c=1
                                continue
                            else:
                                c=0
                                break
                        if c == 0:
                            continue
                        else:
                            break
                    if c ==1:
                        line = cnum[i]+'\t'+line.strip()
                        fout.write(line+'\n')
                        break
                if c==0:
                    fout.write('\t'+line.strip()+'\n')
            else:
                fout.write(line.strip()+'\n')

            line = f.readline()
        else:
            break
    f.close()
    fout.close()
    fnoTag.close()
def getTagVec(fileName):
    tagVec ={}
    if os.path.exists(classifyDataPath+fileName):
        f = open(classifyDataPath+fileName,'rb')
        tagVec = pickle.load(f)
        f.close()
    else:
        if os.path.exists(classifyDataPath+'wordTag.txt'):
            ftag = open(classifyDataPath+'wordTag.txt')
            tags = ftag.readlines()
            for item in tags:
                if len(item)>0:
                    tagVec[item.strip()]=np.random.uniform(-0.6,0.6,10)
            outputTagVec = open(classifyDataPath + fileName, 'wb')
            outputTagVec.truncate()
            pickle.dump(tagVec, outputTagVec)
            outputTagVec.close()
        else:
            print("There is no wordTag.txt!")
            return
    return tagVec


def getClassifyModelEmbedding(wvfile):
    tagVec = getTagVec('tagVec.pkl')
    fw = open(wvfile,'rb')
    wordVec = pickle.load(fw)
    fw.close()
    embedding = []
    embeddingDic = {}
    i=0
    for wkey in wordVec:

        for tkey in tagVec:
            key= tkey+'_'+wkey
            vec =np.concatenate((tagVec[tkey],wordVec[wkey]),axis = 0)
            embeddingDic[key]=i
            embedding.append(vec)
            i+=1
    embedding = data_helper.embedding_regulation(embedding)
    outputEmbedding = open(classifyDataPath + 'embedding.pkl', 'wb')
    outputEmbedding.truncate()
    pickle.dump(embedding, outputEmbedding)
    outputEmbedding.close()

    outputEdict = open(classifyDataPath + 'embeddingDic.pkl', 'wb')
    outputEdict.truncate()
    pickle.dump(embeddingDic, outputEdict)
    outputEdict.close()
    return embedding
def getInput(fileName, fembeddingdic, maxLen, outputtag):
    finput = open(classifyDataPath + 'inputData.txt', 'w')
    funk = open(dataPath + 'unknown.txt', 'w')

    embeddingDic = {}

    if os.path.exists(classifyDataPath + fembeddingdic):
        f = open(classifyDataPath + fembeddingdic, 'rb')
        embeddingDic = pickle.load(f)
        f.close()
    else:
        print("There is no embeddingDic!")
        return

    f = open(classifyDataPath + fileName)
    input = []
    target = []
    data = (input, target)
    for line in f.readlines():
        sline = line.split('\t')
        if len(sline)==2:
            stag = []

            word_tags = sline[1].strip().split()
            le = 0
            for i in range(len(word_tags)):

                item = word_tags[i].split('/')
                if len(item) == 2:
                    key = item[1].split('-')[-1].strip()+'_'+item[0].strip()
                    if key not in embeddingDic.keys():
                        cs = []
                        tags = []
                        if i + 1 < len(word_tags):
                            cs, tags = data_helper.unkWordsProcess(item[0], item[1], word_tags[i + 1].split('/')[1])
                        else:
                            cs, tags = data_helper.unkWordsProcess(item[0], item[1], None)
                        for n in range(len(cs)):
                            key_new = tags[n].split('-')[-1].strip()+'_'+cs[n].strip()
                            if key_new not in embeddingDic.keys():
                                funk.write(cs[n] + '\n')
                                cs[n] = "<unk>"
                                key_new = tags[n].split('-')[-1].strip() + '_' + cs[n].strip()
                            if le < maxLen:
                                stag.append(embeddingDic[key_new])
                                finput.write(cs[n] + '/' + tags[n] + ' ')
                                le += 1

                    else:
                        if le < maxLen:
                            stag.append(embeddingDic[key])

                            finput.write(item[0] + '/' + item[1] + ' ')
                            le += 1
                    if le >= maxLen:
                        break
                else:
                    continue
            finput.write('\n')
            for j in range(le, maxLen):
                stag.append(embeddingDic['pad_<pad>'])

            input.append(stag)

            target.append(eval(sline[0].strip()))

    output = open(classifyDataPath + outputtag + 'data.pkl', 'wb')
    print(classifyDataPath + outputtag + 'data.pkl')
    output.truncate()
    pickle.dump(data, output)
    output.close()
    funk.close()
def load_test_data(fileName, max_len):

    f = open(classifyDataPath + fileName, 'rb')
    print('load data from %s', classifyDataPath + fileName)
    test_set = np.array(pickle.load(f))
    f.close()


    test_set_x, test_set_y = test_set

    test_set = (test_set_x, test_set_y)
    new_test_set_x = np.zeros([len(test_set[0]), max_len])

    new_test_set_y = np.zeros([len(test_set[0])])


    def padding_and_generate_mask(x, new_x, y, new_y):
        for i, (x, y) in enumerate(zip(x, y)):
            # whether to remove sentences with length larger than maxlen
            if len(x) != 28:
                print(i, len(x))
                print(x)
            new_x[i] = x
            new_y[i] = y

        new_set = (new_x, new_y)
        del new_x
        return new_set

    test_set = padding_and_generate_mask(test_set[0], new_test_set_x, test_set[1], new_test_set_y)

    return test_set
def load_data(fileName, max_len, valid_portion=0.2):

    f = open(classifyDataPath + fileName, 'rb')
    print('load data from %s', classifyDataPath + fileName)
    train_set = np.array(pickle.load(f))
    f.close()

    len_data = len(train_set[0])
    len_train = int(len_data * (1 - valid_portion))

    train_set_x, train_set_y = train_set


    valid_set_x = train_set_x[len_train:]
    valid_set_y = train_set_y[len_train:]

    train_set_x = train_set_x[:len_train]
    train_set_y = train_set_y[:len_train]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    new_train_set_x = np.zeros([len(train_set[0]), max_len])

    new_train_set_y = np.zeros(len(train_set[0]))

    new_valid_set_x = np.zeros([len(valid_set[0]), max_len])

    new_valid_set_y = np.zeros(len(train_set[0]))

    def padding_and_generate_mask(x, new_x, y, new_y):
        for i, (x, y) in enumerate(zip(x, y)):
            # whether to remove sentences with length larger than maxlen
            if len(x) != 28:
                print(i, len(x))
                print(x)
            new_x[i] = x
            new_y[i] = y

        new_set = (new_x, new_y)
        del new_x
        return new_set



    train_set = padding_and_generate_mask(train_set[0], new_train_set_x, train_set[1], new_train_set_y)
    valid_set = padding_and_generate_mask(valid_set[0], new_valid_set_x, valid_set[1], new_valid_set_y)

    return train_set, valid_set
#在序列标注的结果文件fileName中，对应写入classify的结果到outfile（仅用于测试，序列标注的句子和classify的句子应对应相同）
def getTestResult(prediction,y,fileName,outfile):
    f=open(fileName,encoding='utf-8')
    lines = f.readlines()
    f.close()
    fout=open(outfile,'w',encoding='utf-8')
    outlines=[]
    if len(y)==0:
        for i in range(len(prediction)):
            outlines.append(str(prediction[i])+'\t'+lines[i])
    else:
        for i in range(len(prediction)):
            outlines.append(str(prediction[i]) + '\t' + y[i]+'\t'+lines[i])
    fout.writelines(outlines)
    fout.close()
def getTheDataByClassifyTag(fileName,outputfile,tags):
    f=open(fileName,'r',encoding='utf-8')
    fout=open(outputfile,'w',encoding='utf-8')
    line=f.readline()
    while True:
        if line:
            if line[0].isdigit():
                if eval(line[0]) in tags:
                    fout.write(line.strip()+'\n')


        else:
            break
        line = f.readline()
    f.close()
    fout.close()


#preClassification2('noTagData2.txt','NewPretag.txt','noTagData.txt')
#preClassification('noTagData2.txt','NewPretag.txt','noTagData.txt')
#data_helper.getRandomList(classifyDataPath+'classifyTraindata.txt',classifyDataPath+'trainData.txt')
#replaceClassTag(classifyDataPath+'classifyTrain.txt',classifyDataPath+'classifyTrain1.txt',5,4)
#data_helper.getRandomList(classifyDataPath+'classifyTrain.txt',classifyDataPath+'RandomclassifyTrain.txt')
#getTheDataByClassifyTag(classifyDataPath+'RandomclassifyTrain.txt',classifyDataPath+'temp.txt',[2,3])




