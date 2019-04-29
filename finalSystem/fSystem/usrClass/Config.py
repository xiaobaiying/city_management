import time
import tensorflow as tf
import pickle
import os

flags =tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('word_dim',100,'word_dim')

flags.DEFINE_integer('batch_size',128,'the batch_size of the training procedure')
flags.DEFINE_float('lr',0.002,'the learning rate')
flags.DEFINE_float('lr_decay',0.9,'the learning rate decay')
flags.DEFINE_integer('vocabulary_size',79543,'vocabulary_size')
flags.DEFINE_integer('emdedding_dim',52,'embedding dim')
flags.DEFINE_integer('hidden_neural_size',128,'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num',1,'LSTM hidden layer num')
flags.DEFINE_integer('max_len',28,'max_len of training sentence')
flags.DEFINE_float('init_scale',0.1,'init scale')
flags.DEFINE_integer('class_num',26,'class num')
flags.DEFINE_float('keep_prob',0.8,'dropout rate')
flags.DEFINE_integer('num_epoch',7,'num epoch')#训练轮数
flags.DEFINE_integer('max_decay_epoch',30,'num epoch')#decay 衰退
flags.DEFINE_integer('max_grad_norm',6,'max_grad_norm')
flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"runs")),'output directory')
flags.DEFINE_integer('check_point_every',10,'checkpoint every num epoch ')
dataPath =  'E:/技术和知识/面试准备/项目/城市管理案/finalSystem/data/'
class Config(object):

    hidden_neural_size=FLAGS.hidden_neural_size
    vocabulary_size=FLAGS.vocabulary_size
    embed_dim=FLAGS.emdedding_dim
    hidden_layer_num=FLAGS.hidden_layer_num
    class_num=FLAGS.class_num
    keep_prob=FLAGS.keep_prob
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    batch_size=FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm=FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    max_decay_epoch = FLAGS.max_decay_epoch
    out_dir=FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every
    word_dim = FLAGS.word_dim