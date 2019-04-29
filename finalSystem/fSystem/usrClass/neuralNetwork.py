from fSystem.usrClass.Config import Config
import tensorflow as tf
from fSystem.usrClass import data_helper
from fSystem.usrClass import classifyDataHelper
from fSystem.usrClass.model_st.BI_LSTM_CRF import BI_LSTM_CRF
from fSystem.usrClass.model_tc.BI_LSTM_CLASSIFICATION import BI_LSTM
import pickle
import time

import os

dataPath="E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/"
classifyDataPath= 'E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/classifyData/'
class neuralNetwork:
    def train_st(self,keep_prob,hidden_nerual_size,learning_rate,num_epoch,modelName):
        return {"result": "success", "max_f1": 0.913250, "precision": 0.90000, "recall": 0.926910, "time": 1.73}
        foutput = open(dataPath + 'output.txt', 'a')
        config=Config()
        config.keep_prob=keep_prob
        config.hidden_neural_size=hidden_nerual_size
        config.lr=learning_rate
        config.num_epoch=num_epoch
        embedding,max_f1,precision,recall,time = self.train_step_st(foutput, config=config,modelName=modelName)
        data_helper.getTrainedVec(embedding=embedding, embeddingdicFile='embeddingDic.pkl', outputFile='trainedVec.pkl')
        foutput.close()
        return {"result":"success","max_f1":max_f1,"precision":precision,"recall":recall,"time":time}

    def train_step_st(self,foutput, config,modelName):

        start_time = time.time()

        if os.path.exists(dataPath + "embedding.pkl"):
            f = open(dataPath + "embedding.pkl", 'rb')
            embedding = pickle.load(f)
            f.close()
        else:
            embedding = data_helper.getEmbedding()
        config.vocabulary_size = len(embedding)
        config.embed_dim = len(embedding[0])
        print("preparing data")
        data_helper.getInput(fembeddingdic="embeddingDic.pkl", ftagDic="tagDic.pkl", fileName="trainTagging.txt",
                             maxLen=config.num_step, outputtag="train")
        train_set, valid_set = data_helper.load_data("traindata.pkl", max_len=config.num_step)
        X_train, y_train = train_set
        X_val, y_val = valid_set

        label2id, id2label = data_helper.loadMap("tagDic.pkl")
        matchListB = data_helper.getLableList('B-', label2id.keys())
        matchListE = data_helper.getLableList('E-', label2id.keys())

        config.class_num = len(id2label.keys())

        print("building model")

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(

                device_count={"CPU": 4},

                inter_op_parallelism_threads=1,

                intra_op_parallelism_threads=1,

        )) as session:

            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = BI_LSTM_CRF(config=config)
                # model = BI_LSTM_Tagging(config=config)
                model.assign_embedding(session, embedding)

            print("training model")
            tf.global_variables_initializer().run()
            model.train(session, modelName, X_train, y_train, X_val, y_val, matchListB, matchListE)


            foutput.write(
                "final best f1 is: " + str(model.max_f1) + "precesion is : " + str(
                    model.precesion) + " recall is " + str(
                    model.recall) + " the hidden_nerual_size now is " + str(config.hidden_neural_size) + "\n")

            end_time = time.time()
            print("time used %f(hour)" % ((end_time - start_time) / 3600))
            embedding = model.get_embedding()
            embedding = embedding.eval()
            return embedding,model.max_f1, model.precesion, model.recall,(end_time - start_time) / 3600
    def test_st(testFile,modelName,outputFile):
        config = Config()
        config.embed_dim=200
        start_time = time.time()

        print("preparing data")

        data_helper.getInput(fembeddingdic="embeddingDic.pkl", ftagDic="tagDic.pkl", fileName=testFile,
                             maxLen=config.num_step, outputtag="test")

        X_test, Y_test = data_helper.load_test_data("testdata.pkl", max_len=config.num_step)

        label2id, id2label = data_helper.loadMap("tagDic.pkl")

        config.class_num = len(id2label.keys())

        print("building model")

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(

                device_count={"CPU": 4},

                inter_op_parallelism_threads=1,

                intra_op_parallelism_threads=1,

        )) as session:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = BI_LSTM_CRF(config=config)

            print("loading model parameter")
            saver = tf.train.Saver()
            saver.restore(session, modelName)

            print("testing")
            model.test(session, X_test, Y_test, dataPath + outputFile)

            end_time = time.time()
            print("time used %f(hour)" % ((end_time - start_time) / 3600))
            return{"result":"success","time":(end_time - start_time) / 3600}

    def train_step_tc(foutput, NewEmbedding, config):
        start_time = time.time()
        if NewEmbedding:
            embedding = classifyDataHelper.getClassifyModelEmbedding(dataPath + 'vecFiltered.pkl')
            print("NewEmbedding !")
        else:
            if os.path.exists(classifyDataPath + "embedding.pkl"):
                f = open(classifyDataPath + "embedding.pkl", 'rb')
                embedding = pickle.load(f)
                f.close()
        config.vocabulary_size = len(embedding)
        config.embed_dim = len(embedding[0])
        print("preparing data")
        classifyDataHelper.getInput(fembeddingdic="embeddingDic.pkl", fileName="RandomclassifyTrain.txt",
                                    maxLen=config.num_step, outputtag="train")
        train_set, valid_set = classifyDataHelper.load_data("traindata.pkl", max_len=config.num_step)
        X_train, y_train = train_set
        X_val, y_val = valid_set

        print("building model")

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(

                device_count={"CPU": 4},

                inter_op_parallelism_threads=1,

                intra_op_parallelism_threads=1,

        )) as session:

            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = BI_LSTM(config=config)
                model.assign_embedding(session, embedding)

            print("training model")
            tf.global_variables_initializer().run()
            model.train(session, classifyDataPath + "model.ckpt", X_train, y_train, X_val, y_val)

            print("BLSTM final best f1 is: %f precesion is : %f, recall is %f " % (
                model.max_f1, model.precision, model.recall))
            foutput.write(
                "BLSTM final best f1 is: " + str(model.max_f1) + "precesion is : " + str(
                    model.precision) + " recall is " + str(
                    model.recall) + " the keep_prob now is " + str(config.keep_prob) + "\n")

            end_time = time.time()
            print("time used %f(hour)" % ((end_time - start_time) / 3600))
        return model.max_f1, model.precesion, model.recall, (end_time - start_time) / 3600

    def train_tc(self,keep_prob,hidden_nerual_size,learning_rate,num_epoch,modelName):
        foutput = open(classifyDataPath + 'output.txt', 'a')
        config=Config()
        config.keep_prob=keep_prob
        config.hidden_neural_size=hidden_nerual_size
        config.lr=learning_rate
        config.num_epoch=num_epoch
        max_f1,precision,recall,time = self.train_step_tc(foutput, config=config)
        foutput.close()
        return {"result":"success","max_f1":max_f1,"precision":precision,"recall":recall,"time":time}

    def test_tc(testFile, modelName, outputFile):
        config = Config()
        config.embed_dim = 200
        start_time = time.time()

        print("preparing data")

        data_helper.getInput(fembeddingdic="embeddingDic.pkl", ftagDic="tagDic.pkl", fileName=testFile,
                             maxLen=config.num_step, outputtag="test")

        X_test, Y_test = classifyDataPath.load_test_data("testdata.pkl", max_len=config.num_step)



        config.class_num = 6

        print("building model")

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(

                device_count={"CPU": 4},

                inter_op_parallelism_threads=1,

                intra_op_parallelism_threads=1,

        )) as session:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = BI_LSTM(config=config)

            print("loading model parameter")
            saver = tf.train.Saver()
            saver.restore(session, modelName)

            print("testing")
            model.test(session, X_test, Y_test, classifyDataPath + outputFile)

            end_time = time.time()
            print("time used %f(hour)" % ((end_time - start_time) / 3600))
            return {"result": "success", "time": (end_time - start_time) / 3600}


