import pickle
import codecs
import numpy as np
import os
import pandas as pd
import re
import random
#dataPath = '/root/PycharmProjects/NER/bi_LSTM_CRF_test/data/'
dataPath = 'E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/'
# classifyDataPath = 'D:/Work/Name Entity Recognition/bi_LSTM_CRF_test/data/relationClassify/'

#*******************************公用函数****************************************
#将不能获得词向量的词拆分成字以获取字向量
def unkWordsProcess(word,tag,NextTag):
    cs = [item for item in word]
    tags=[]
    l = len(word)
    if tag == 'O':
        tags = [tag for j in range(l)]
    else:
        tt = tag.split('-')
        if tt[0] =='B':
            tags.append(tag)
            for i in range(1,l-1):
                tags.append('I-'+tt[1])
            if NextTag!=None:
                if NextTag[0]!='B':
                    tags.append('I-' + tt[1])
                else:
                    tags.append('E-' + tt[1])
            else:
                tags.append('E-' + tt[1])

        elif tt[0] =='I':
            tags = [tag for j in range(l)]
        elif tt[0] =='S':
            tags.append('B-'+tt[1])
            for i in range(1,l-1):
                tags.append('I-'+tt[1])
            tags.append('E-'+tt[1])
        else:
            for i in range(0,l-1):
                tags.append('I-'+tt[1])
            tags.append(tag)
    return cs,tags
def load_vec(fileName):
    with codecs.open(fileName, 'r', "utf-8") as f:
        vecDict={}
        size  = 0
        vocab = []
        feature = []
        flag = 0
        for line in f:
            if not line:
                break
            if flag == 0:
                line = line.strip().split()
                _, size = int(line[0]), int(line[1])
                flag = 1
                continue
            line = line.strip().split()
            if not line:
                continue
            w = line[0]
            vec = [float(i) for i in line[1:]]
            if len(vec) != size:
                continue
            vec = np.array(vec)
            vecDict[w]= vec

            # print length,vec
            vocab.append(w)
            feature.append(vec)
        feature = np.array(feature)
        vecUnk = np.mean(feature,axis=0)
        vecDict['<unk>']=vecUnk
        vecDict['<pad>']=np.zeros(len(vecUnk))
        output = open(dataPath + 'vecDict.pkl', 'wb')
        output.truncate()
        pickle.dump(vecDict, output)
        output.close()

# 获得句子最大长度（95%）
def get_max_len(fileName):
    f = open(fileName)
    sList = f.readlines()
    sentenceNum = len(sList)
    listLength = [len(s.split(" ")) for s in sList]
    listLengthSorted = sorted(listLength)
    sentence_length = listLengthSorted[int(sentenceNum * 0.95)]
    print(sentence_length)
    return sentence_length
def getEmbedding():
    embeddingToWord = {}
    embedding = []
    embeddingDic = {}
    f = open(dataPath + 'vecDict.pkl', 'rb')
    vecDict = pickle.load(f)
    f.close()
    embedding.append(vecDict['<pad>'])
    embeddingDic['<pad>'] = 0
    embeddingToWord[0] = '<pad>'
    i = 1
    for key in vecDict.keys():
        if key == '<pad>':
            continue
        embedding.append(vecDict[key])
        embeddingDic[key] = i
        embeddingToWord[i] = key
        i += 1
    embedding = embedding_regulation(embedding)
    outputEmbedding = open(dataPath + 'embedding.pkl', 'wb')
    outputEmbedding.truncate()
    pickle.dump(embedding, outputEmbedding)
    outputEmbedding.close()

    outputEdict = open(dataPath + 'embeddingDic.pkl', 'wb')
    outputEdict.truncate()
    pickle.dump(embeddingDic, outputEdict)
    outputEdict.close()
    outputEtw = open(dataPath + 'embedding2Word.pkl', 'wb')
    outputEtw.truncate()
    pickle.dump(embeddingToWord, outputEtw)
    outputEtw.close()
    return embedding
'''

def getEmbedding():
    embeddingToWord = {}
    embedding = []
    embeddingDic = {}
    f = open(dataPath + 'vecDict.pkl', 'rb')
    vecDict = pickle.load(f)
    f.close()
    i = 0
    for key in vecDict.keys():
        embedding.append(vecDict[key])
        embeddingDic[key] = i
        embeddingToWord[i] = key
        i += 1
    embedding = embedding_regulation(embedding)
    outputEmbedding = open(dataPath + 'embedding.pkl', 'wb')
    outputEmbedding.truncate()
    pickle.dump(embedding, outputEmbedding)
    outputEmbedding.close()

    outputEdict = open(dataPath + 'embeddingDic.pkl', 'wb')
    outputEdict.truncate()
    pickle.dump(embeddingDic, outputEdict)
    outputEdict.close()
    outputEtw = open(dataPath + 'embedding2Word.pkl', 'wb')
    outputEtw.truncate()
    pickle.dump(embeddingToWord, outputEtw)
    outputEtw.close()
    return embedding
'''
def nextBatch(X, y, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    y_batch = list(y[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


def nextRandomBatch(X, y, batch_size=128):
    X_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch
#将embedding正则化
def embedding_regulation(embedding):
    embedding = np.array(embedding)
    mean = np.mean(embedding,axis = 0)
    embedding = [item-mean for item in embedding]
    Var = np.var(embedding,axis = 0)
    Vars=np.tile(Var,(len(embedding),1))
    embedding = embedding/Vars
    return embedding
#得到训练过的向量
def getTrainedVec(embedding,embeddingdicFile,outputFile):
    trainedVec = {}
    f = open(dataPath + embeddingdicFile, 'rb')
    embeddingDict = pickle.load(f)
    f.close()
    for key in embeddingDict.keys():
        trainedVec[key]=embedding[embeddingDict[key]]
    output = open(dataPath + outputFile, 'wb')
    output.truncate()
    pickle.dump(trainedVec, output)
    output.close()
    print("Trained Vec gotten!")
#得到file1和file2的差集(file1-file2),输出到outfile
def getDifferentSet(file1,file2,outfile):
    f1=open(file1,encoding='utf-8')
    f2=open(file2,encoding='utf-8')
    lines1=f1.readlines()
    lines2=f2.readlines()
    fout =open(outfile,'w',encoding='utf-8')
    lines1=set(lines1)
    lines2=set(lines2)
    lines = list(lines1.difference(lines2))
    fout.writelines(lines)
    f1.close()
    f2.close()
    fout.close()

#******************************实体标注所用函数*********************************
def getTagDic(fileName):
    tagDic={}
    f = open(dataPath+fileName,'r')
    k=0
    for line in f.readlines():
        line = line.strip()
        if len(line)>0:
            tagDic[line]=k
            k+=1
    output = open(dataPath + 'tagDic.pkl', 'wb')
    output.truncate()
    pickle.dump(tagDic, output)
    output.close()
    return tagDic

'''
句子格式：一句一行，标记格式为 词/标注 词/标注 ……
'''

def getInput(fileName, ftagDic, fembeddingdic, maxLen, outputtag):
    finput = open(dataPath + 'inputData.txt', 'w')
    funk = open(dataPath + 'unknown.txt', 'w')
    tagDic = {}
    embeddingDic = {}
    if os.path.exists(dataPath + ftagDic):
        f = open(dataPath + ftagDic, 'rb')
        tagDic = pickle.load(f)
        f.close()
    else:
        tagDic = getTagDic("tags.txt")
    if os.path.exists(dataPath + fembeddingdic):
        f = open(dataPath + fembeddingdic, 'rb')
        embeddingDic = pickle.load(f)
        f.close()
    else:
        print("There is no embeddingDic!")
        return

    f = open(dataPath + fileName)
    input = []
    target = []
    data = (input, target)
    for line in f.readlines():
        stag = []
        ttag = []
        word_tags = line.strip().split()
        le = 0
        for i in range(len(word_tags)):

            item = word_tags[i].split('/')
            if len(item) == 2:
                if item[0] not in embeddingDic.keys():
                    cs = []
                    tags = []
                    if i + 1 < len(word_tags):
                        cs, tags = unkWordsProcess(item[0], item[1], word_tags[i + 1].split('/')[1])
                    else:
                        cs, tags = unkWordsProcess(item[0], item[1], None)
                    for n in range(len(cs)):
                        if cs[n] not in embeddingDic.keys():
                            funk.write(item[0] + '\n')
                            cs[n] = "<unk>"
                        if le < maxLen:
                            stag.append(embeddingDic[cs[n]])
                            ttag.append(tagDic[tags[n]])
                            finput.write(cs[n] + '/' + tags[n] + ' ')
                            le += 1

                else:
                    if le < maxLen:
                        stag.append(embeddingDic[item[0]])
                        ttag.append(tagDic[item[1]])
                        finput.write(item[0] + '/' + item[1] + ' ')
                        le += 1
                if le >= maxLen:
                    break
            else:
                continue
        finput.write('\n')
        for j in range(le, maxLen):
            stag.append(embeddingDic['<pad>'])
            ttag.append(tagDic['pad'])
        input.append(stag)
        target.append(ttag)
    output = open(dataPath + outputtag + 'data.pkl', 'wb')
    print(dataPath + outputtag + 'data.pkl')
    output.truncate()
    pickle.dump(data, output)
    output.close()
    funk.close()
def load_test_data(fileName, max_len):

    f = open(dataPath + fileName, 'rb')
    print('load data from %s', dataPath + fileName)
    test_set = np.array(pickle.load(f))
    f.close()


    test_set_x, test_set_y = test_set

    test_set = (test_set_x, test_set_y)
    new_test_set_x = np.zeros([len(test_set[0]), max_len])

    new_test_set_y = np.zeros([len(test_set[0]), max_len])


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
def load_data(fileName, max_len, valid_portion=0.02):

    f = open(dataPath + fileName, 'rb')
    print('load data from %s', dataPath + fileName)
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

    new_train_set_y = np.zeros([len(train_set[0]), max_len])

    new_valid_set_x = np.zeros([len(valid_set[0]), max_len])

    new_valid_set_y = np.zeros([len(train_set[0]), max_len])

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

def loadMap(token2id_filepath):
    if not os.path.exists(dataPath + token2id_filepath):
        print("file not exist, building map")
        getTagDic('tagDic.txt')
    id2token = {}
    f = open(dataPath + token2id_filepath, 'rb')
    tagDic = pickle.load(f)
    f.close()
    for key in tagDic.keys():
        id2token[tagDic[key]] = key
    return tagDic, id2token

def getTestResult(X, predicts, y, fout):
    embedding2Word = {}
    _, id2tag = loadMap('tagDic.pkl')
    if os.path.exists(dataPath + 'embedding2Word.pkl'):
        few = open(dataPath + 'embedding2Word.pkl', 'rb')
        embedding2Word = pickle.load(few)
        few.close()
    else:
        print('There is no embedding2Word.pkl')
        return

    if len(y) > 0:

        for i in range(len(X)):
            line = ''
            for j in range(len(predicts[i])):
                line += embedding2Word[X[i][j]] + '/' + id2tag[predicts[i][j]] + '/' + id2tag[y[i][j]] + ' '
            line += '\n'
            fout.write(line)
    else:
        for i in range(len(X)):
            line = ''
            for j in range(len(predicts[i])):
                line += embedding2Word[X[i][j]] + '/' + id2tag[predicts[i][j]] + ' '
            line += '\n'
            fout.write(line)

def getTransition(y_train_batch, num_class):
    transition_batch = []
    for m in range(len(y_train_batch)):
        y = [num_class] + list(y_train_batch[m]) + [0]
        for t in range(len(y)):
            if t + 1 == len(y):
                continue
            i = y[t]
            j = y[t + 1]
            if i == 0:
                break
            transition_batch.append(i * (num_class + 1) + j)
    transition_batch = np.array(transition_batch)
    return transition_batch
#获得具有某些特征的标签
def getLableList(s, labels):
    matchList = []
    re_s = re.compile(r'^' + s)
    for item in labels:
        matchObject = re.match(re_s, item)
        if matchObject:
            matchList.append(item.strip())
    return matchList
#将file里的内容按行随机打乱，输出到outputFile
def getRandomList(file,outputFile):
    f = open(file,encoding='utf-8')
    list = f.readlines()
    f.close()
    random.shuffle(list)
    if outputFile!=None:
        output=open(outputFile,'w',encoding='utf-8')
        output.writelines(list)
        output.close()
    else:
        return list
#将分类train中，每行的类标签除去
def getRidOfTheClassifyTag(file,outputfile):
    f=open(file,encoding='utf-8')
    line = f.readline()
    fout = open(outputfile,'w',encoding='utf-8')
    while True:
        if line:
            tagAndSentence = line.split('\t')
            if len(tagAndSentence)==2:
                fout.write(tagAndSentence[1].strip()+'\n')
        else:
            break
        line = f.readline()
    f.close()
    fout.close()
#词向量文件瘦身（没用到的词就删去）
def vecFilter(vecFile,dataFile,outputFile):
    vecFiltered ={}
    fv=open(vecFile,'rb')
    vecDic =pickle.load(fv)
    fv.close()
    fd = open(dataFile,'r',encoding='utf-8')
    vecFiltered['<unk>']=vecDic['<unk>']
    vecFiltered['<pad>'] = vecDic['<pad>']
    for key in vecDic:
        if len(key) == 1:
            vecFiltered[key]=vecDic[key]
    line = fd.readline()
    while True:
        if line:
            line = line.strip().split()
            if len(line) >0:
                for item in line:
                    if item in vecDic.keys():
                        vecFiltered[item]=vecDic[item]
        else:
            break
        line = fd.readline()
    print(len(vecFiltered.keys()))
    fout = open(outputFile,'wb')
    fout.truncate()
    pickle.dump(vecFiltered, fout)
    fout.close()
    fd.close()
#vecFilter(dataPath+'trainedVec.pkl',dataPath+'post_key.txt',dataPath+'vecFiltered.pkl')
#getRidOfTheClassifyTag(classifyDataPath+'TEST.txt',dataPath+'TEST.txt')
#getRandomList(dataPath+'trainTagging.txt',dataPath+'trainTagging1.txt')
#getDifferentSet(dataPath+'trainTagging.txt',dataPath+'TEST.txt',dataPath+'trainTagging1.txt')
#load_vec(dataPath+'newsblogbbs.vec')