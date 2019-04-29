import pickle
import re
import os
dataPath='E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/'
class remoteSupervision:
    def getTaggedWordSet(self,file):
        taggedDict = {}
        file = open(dataPath + file, 'r', encoding='utf-8')
        re_s = re.compile(r'^B-')
        while True:
            line = file.readline()
            if line:
                words = []
                tags = []
                word_tags = line.strip().split()
                for item in word_tags:
                    wt = item.split('/')
                    if len(wt) == 2:
                        words.append(wt[0])
                        tags.append(wt[1])
                key = ""  # 对应字典中的key
                value = ""  # 对应字典中的value
                tag = ""  # 标记tag中除去BIE的部分

                for i in range(len(tags)):  # 遍历整个句子的tag
                    if tags[i].strip() == 'O':
                        if len(key) > 0:
                            key = key.rstrip('|')
                            value = value.rstrip('|')
                            if key not in taggedDict:
                                taggedDict[key] = value
                            key = ""  # 对应字典中的key
                            value = ""  # 对应字典中的value
                            tag = ""  # 标记tag中除去BIE的部分
                        else:
                            continue
                    else:  # 如果前面找到了一个B
                        tt = tags[i].strip().split('-')
                        if tt[0] == 'B':  # 找到了一个B,说明和前面的不是一个词了，把前面记录下的key和value写到字典里
                            if len(key) > 0:
                                key = key.rstrip('|')
                                value = value.rstrip('|')
                                if key not in taggedDict:
                                    taggedDict[key] = value
                            key = words[i].strip() + '|'  # 对应字典中的key
                            value = words[i].strip() + '|'  # 对应字典中的value
                            tag = tt[1]  # 标记tag中除去BIE的部分

                        elif tt[0] == 'I' and tag == tt[1]:  # 找到中间的I
                            key += words[i].strip() + '|'
                            value += words[i].strip() + '|'
                            continue
                        elif tt[0] == 'E' and tag == tt[1]:  # 找到E
                            key += words[i].strip()
                            value += tags[i].strip()
                            key = key.rstrip('|')
                            value = value.rstrip('|')
                            if key not in taggedDict:
                                taggedDict[key] = value
                            key = ""  # 对应字典中的key
                            value = ""  # 对应字典中的value
                            tag = ""  #
                    if i + 1 == len(words) and len(key) > 0:  # 找到B开头的标签了
                        key = key.rstrip('|')
                        value = value.rstrip('|')
                        if key not in taggedDict:
                            taggedDict[key] = value
            else:
                break
        file.close()
        output = open(dataPath + 'taggedDic.pkl', 'wb+')
        output.truncate()
        pickle.dump(taggedDict, output)
        output.close()
        return taggedDict

    def correctOTag(self,correctTagFile, aimFile, outputFile):
        tagDic = {}
        if os.path.exists(dataPath + 'correctTag.pkl'):
            fdic = open(dataPath + 'correctTag.pkl', 'rb')
            tagDic = pickle.load(fdic)
            fdic.close()
        ctf = open(dataPath + correctTagFile, 'r', encoding='utf-8')
        while True:
            line = ctf.readline()
            if line:
                key = ""
                value = ""
                word_tags = line.strip().split()
                for item in word_tags:
                    wt = item.split('/')
                    if len(wt) == 2:
                        key += wt[0] + '|'
                        value += wt[1] + '|'
                key = key.rstrip('|')
                value = value.rstrip('|')
                if key not in tagDic:
                    tagDic[key] = value
            else:
                break
        fdic = open(dataPath + 'correctTag.pkl', 'wb+')
        fdic.truncate()
        pickle.dump(tagDic, fdic)
        fdic.close()
        if os.path.exists(dataPath + aimFile):

            f = open(dataPath + aimFile, 'r', encoding='utf-8')
            fout = open(dataPath + outputFile, 'w', encoding='utf-8')
            while True:
                line = f.readline()
                if line:
                    words = []
                    tags = []
                    word_tags = line.strip().split()
                    for item in word_tags:
                        wt = item.split('/')
                        if len(wt) == 2:
                            words.append(wt[0])
                            tags.append(wt[1])
                    # 找到字典中所有包含在当前句子中的key，并按key的长度从大到小排序
                    tagWordsList = [key for key in tagDic.keys() if set(key.split('|')) <= set(words)]
                    tagWordsList.sort(key=lambda x: len(x), reverse=True)
                    # 对所有找到的key进行遍历
                    for item in tagWordsList:
                        tagWords = item.split('|')
                        wordTags = tagDic[item].split('|')
                        # 找到句子中所有key对应的那个开头的位置
                        aimB = [i for i, x in enumerate(words) if x == tagWords[0]]
                        # 从每个key开头比较，看原句中的元素是否和key中完全相等（只标那些尚未被标注过的地方（tag为O的））
                        for a in aimB:
                            i = a
                            j = 0
                            while words[i] == tagWords[j]:
                                i += 1
                                j += 1
                                if j == len(tagWords) or i == len(words):
                                    break
                            # 对句子进行标注
                            if j == len(tagWords):
                                for k in range(len(tagWords)):
                                    tags[a + k] = wordTags[k]
                    # 将句子写到新文件中去
                    for i in range(len(words)):
                        fout.write(words[i] + '/' + tags[i] + ' ')
                    fout.write('\n')

                else:
                    break
            fout.close()
            f.close()

    def remoteSupervision(self,modelfileName,tagfileName,taggedfileName):
        taggedDic=self.getTaggedWordSet(modelfileName)
        if os.path.exists(dataPath + tagfileName):

            f = open(dataPath + tagfileName, 'r', encoding='utf-8')
            fout = open(dataPath + taggedfileName, 'w', encoding='utf-8')
            while True:
                line = f.readline()
                if line:
                    words = []
                    tags = []
                    word_tags = line.strip().split()
                    for item in word_tags:
                        wt = item.split('/')
                        if len(wt) == 2:
                            words.append(wt[0])
                            tags.append(wt[1])
                    # 找到字典中所有包含在当前句子中的key，并按key的长度从大到小排序
                    tagWordsList = [key for key in taggedDic.keys() if set(key.split('|')) <= set(words)]
                    tagWordsList.sort(key=lambda x: len(x), reverse=True)
                    # 对所有找到的key进行遍历
                    for item in tagWordsList:
                        tagWords = item.split('|')
                        wordTags = taggedDic[item].split('|')
                        # 找到句子中所有key对应的那个开头的位置
                        aimB = [i for i, x in enumerate(words) if x == tagWords[0]]
                        # 从每个key开头比较，看原句中的元素是否和key中完全相等（只标那些尚未被标注过的地方（tag为O的））
                        for a in aimB:
                            i = a
                            j = 0
                            while words[i] == tagWords[j] and tags[i] == 'O':
                                i += 1
                                j += 1
                                if j == len(tagWords) or i == len(words):
                                    break
                            # 对句子进行标注
                            if j == len(tagWords):
                                for k in range(len(tagWords)):
                                    tags[a + k] = wordTags[k]
                    # 将句子写到新文件中去
                    for i in range(len(words)):
                        fout.write(words[i] + '/' + tags[i] + ' ')
                    fout.write('\n')

                else:
                    break
            fout.close()
            f.close()
            return{'result':'success','taggedWordNum':len(taggedDic)}

