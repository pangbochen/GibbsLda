# -*- coding: utf-8 -*-
__author__ = 'pangbochen'

import os
import numpy as np
import pandas as pd
import configparser
import random
import codecs
from collections import OrderedDict

# load config file lda_setting.config
conf = configparser.ConfigParser()
conf.read("lda_setting.config")

# load config the file path
TAG_filepath = 'filepath'

textFileName = conf.get(TAG_filepath, 'textfile')
wordidFileName = conf.get(TAG_filepath, 'wordidfile')
doctopicFileName = conf.get(TAG_filepath, 'thetafile')
wordtopicFileName = conf.get(TAG_filepath, 'phifile')
topicNFileName = conf.get(TAG_filepath, 'topNfiel')
topicassignFileName = conf.get(TAG_filepath, 'tassignfile')

# load config of lda model
TAG_model = 'lda_args'

K = int(conf.get(TAG_model, "K"))
alpha = float(conf.get(TAG_model, "alpha"))
beta = float(conf.get(TAG_model, "beta"))
iter_times = int(conf.get(TAG_model, "iter_times"))
topic_words_num = int(conf.get(TAG_model, "top_words_num"))

# LDA Model

class LDAModel:
    def __init__(self):
        # attributes of LDA model
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iter_times = iter_times
        self.topic_words_num = topic_words_num

        # attributes of the corpus
        self.documents_num = 0
        self.documents = []
        self.words_num = 0
        self.word_to_id = OrderedDict()

        # attributes of file name
        self.textFileName = textFileName
        self.wordidFileName = wordidFileName
        self.doctopicFileName = doctopicFileName
        self.wordtopicFileName = wordtopicFileName
        self.topicNFileName = topicNFileName
        self.topicassignFileName = topicassignFileName

        # vectors for computing using
        self.tmp = np.zeros(self.K)
        self.nd = np.zeros( (self.documents_num, self.K) , dtype="int")
        self.ndsum = np.zeros(self.documents_num, dtype="int")
        self.nw = np.zeros( (self.words_num, self.K) , dtype="int" )
        self.nwsum = np.zeros(self.K, dtype="int")
        #
        self.P = np.array( [ [0 for j in range(len(self.documents[i]))] for i in range(self.documents_num) ] )

        # init theta and phi
        self.theta = np.array([ [0.0 for j in range(self.K)] for i in range(self.documents_num)])
        self.phi = np.array([ [0.0 for j in range(self.words_num)] for i in range(self.K) ])


        # set random topic for each word
        for i in range(len(self.P)):
            self.ndsum[i] = len(self.documents[i])
            for j in range(len(self.documents[i])):
                random_topic = random.randint(0, self.K-1)
                self.P[i][j] = random_topic
                self.nw[self.documents[i][j]][random_topic] +=1
                self.nwsum[random_topic] += 1
                self.nd[i][random_topic] += 1

        # end of init

    def _compute_theta(self):
        for i in range(self.documents_num):
            self.theta[i] = (self.nd[i]+self.alpha)/(self.ndsum[i]+self.K*self.alpha)
    def _compute_phi(self):
        for i in range(self.K):
            self.phi[i]