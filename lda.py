# -*- coding: utf-8 -*-
__author__ = 'pangbochen'

import os
import numpy as np
import pandas as pd
import configparser
import random
import codecs
from collections import OrderedDict
from lda_config import ldaConfig
# load config file lda_setting.config
conf = configparser.ConfigParser()
conf.read("lda_setting.config")


# load config the file path
TAG_filepath = 'filepath'

textFileName = conf.get(TAG_filepath, 'textfile')
wordidFileName = conf.get(TAG_filepath, 'wordidfile')
doctopicFileName = conf.get(TAG_filepath, 'thetafile')
wordtopicFileName = conf.get(TAG_filepath, 'phifile')
topicNFileName = conf.get(TAG_filepath, 'topNfile')
topicassignFileName = conf.get(TAG_filepath, 'tassignfile')

# load config of lda model
TAG_model = 'lda_args'

K = int(conf.get(TAG_model, "K"))
alpha = float(conf.get(TAG_model, "alpha"))
beta = float(conf.get(TAG_model, "beta"))
iter_times = int(conf.get(TAG_model, "iter_times"))
topic_words_num = int(conf.get(TAG_model, "topic_words_num"))

# LDA Model

class LDAModel(object):
    def __init__(self, config):
        # attributes of LDA model
        self.K = config.K
        self.alpha = config.alpha
        self.beta = config.beta
        self.iter_times = config.iter_times
        self.topic_words_num = config.topic_words_num

        # attributes of file name
        self.textFileName = config.textFileName
        self.wordidFileName = config.wordidFileName
        self.doctopicFileName = config.doctopicFileName
        self.wordtopicFileName = config.wordtopicFileName
        self.topicNFileName = config.topicNFileName
        self.topicassignFileName = config.topicassignFileName

        print("finish configration")
        # attributes of the corpus
        self.documents_num = 0
        self.documents = []
        self.words_num = 0
        self.word_to_id = OrderedDict()
        self.id_to_word = OrderedDict()
        # get corpus
        print("start loading corpus")
        with codecs.open(self.textFileName, 'r', encoding='utf-8') as f:
            corpus = f.readlines()
        iter_index = 0
        for line in corpus:
            if len(line) > 0:
                tmp_words = line.strip().split()
                tmp_doc = []
                for word in tmp_words:
                    if word in self.word_to_id:
                        tmp_doc.append(self.word_to_id[word])
                    else:
                        self.word_to_id[word] = iter_index
                        self.id_to_word[iter_index] = word
                        tmp_doc.append(iter_index)
                        iter_index += 1
                self.documents.append(tmp_doc)
            else:
                pass
        self.documents_num = len(self.documents)
        self.words_num = len(self.word_to_id)

        print("finish loading corpus")

        # vectors for computing using
        self.tmp_p = np.zeros(self.K)
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
            self.phi[i] = (self.nw.T[i] + self.beta)/(self.nwsum[i]+self.words_num*self.beta)

    #
    def saveFile(self):
        # save theta : document -- topic
        with codecs.open(self.doctopicFileName, 'w', encoding='utf-8') as f:
            for i in range(self.documents_num):
                for j in range(self.K):
                    f.write(str(self.theta[i][j])+'\t')
                f.write('\n')

        # save phi : word -- topic
        with codecs.open(self.wordtopicFileName, 'w', encoding='utf-8') as f:
            for i in range(self.K):
                for j in range(self.words_num):
                    f.write(str(self.phi[i][j])+'\t')
                f.write('\n')

        # save topic n
        with codecs.open(self.topicNFileName, 'w', encoding='utf-8') as f:
            self.topic_words_num = min(self.topic_words_num, self.words_num)
            for i in range(self.K):
                f.write('Topic '+str(i)+":\n")
                topic_words = [(n, self.phi[i][n]) for n in range(self.words_num)]
                topic_words.sort(key=lambda i:i[1], reverse= True)
                for j in range(self.topic_words_num):
                    word = self.id_to_word[topic_words[j][0]]
                    f.write('\t'+word+'\t'*2+str(topic_words[j][1])+'\n')

        # save topic assigned
        with codecs.open(self.topicassignFileName, 'w', encoding='utf-8') as f:
            for i in range(self.documents_num):
                for j in range(len(self.documents[i])):
                    f.write(str(self.documents[i][j])+':'+str(self.P[i][j])+'\t')
                f.write('\n')

    def sampling(self, i, j):
        # init
        topic = self.P[i][j]
        word = self.documents[i][j]
        self.nw[word][topic] -= 1
        self.nd[i][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[i] -= 1
        #
        Vbeta = self.words_num*self.beta
        Kalpha = self.K * self.alpha

        self.tmp_p = (self.nw[word]+self.beta)/(self.nwsum+Vbeta) * \
                     (self.nd[i]+self.alpha)/(self.ndsum[i]+Kalpha)

        for k in range(1,self.K):
            self.tmp_p[k] += self.tmp_p[k-1]

        u = random.uniform(0, self.tmp_p[self.K-1])

        for topic in range(self.K):
            if self.tmp_p[topic]>u:
                break
        #
        self.nw[word][topic] += 1
        self.nd[i][topic] += 1
        self.nwsum[topic] += 1
        self.ndsum[i] += 1
        #
        return topic


    def generate(self):
        for t in range(self.iter_times):
            print("iteration time: " + str(t))
            if t % 10 == 0:
                print("iteration time: "+str(t))
            for i in range(self.documents_num):
                for j in range(len(self.documents[i])):
                    topic = self.sampling(i, j)
                    self.P[i][j] = topic
        #end of iteration
        print("end iter")
        print("compute document -- topic")
        self._compute_theta()
        print("compute word -- document")
        self._compute_phi()
        print("begin saving model")
        self.saveFile()


if __name__ == '__main__':
    ldaconfig = ldaConfig()
    lda = LDAModel(ldaconfig)
    lda.generate()