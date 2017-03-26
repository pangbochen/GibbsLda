# -*- coding: utf-8 -*-
__author__ = 'pangbochen'

import numpy as np
import random
import cumulative
import pandas
def GibbsLDA(doc_set, K, a, b, iter_number):
    '''
    :param doc_set:            documents
    :param K:               number of topics
    :param a:               a
    :param b:               b
    :param iter_number:     iteration times
    :return:                theta(doc->topic)
                             phi(topic->word)
                             tassign
    '''

    #word instance
    word_instances = []
    word_id_dict = {}
    word_cnt = 0
    for doc in doc_set:
        for word in doc:
            if word not in word_id_dict:
                word_id_dict[word] = word_cnt
                word_cnt += 1
    #
    V = len(word_id_dict)
    M = len(doc_set)

    nw = [[0 for _ in range(K)] for _ in range(V)]
    nwsum = [0 for _ in range(K)]
    nd = [[0 for _ in range(K)] for _ in range(M)]
    ndsum = [0 for _ in range(M)]
    z = [[]for _ in range[M]]
    #tmp file
    #Initial
    for m in range(len(doc_set)):
        doc = doc_set[m]
        z[m] = [0 for _ in range(len(doc))]

        for n in range(len(doc)):
            topic_index = random.randrange(K)
            z[m][n] = topic_index
            word_id = word_id_dict[doc[n]]
            nw[word_id][topic_index] += 1
            nwsum[topic_index] += 1
            nd[m][topic_index] += 1
            ndsum[m] += 1

    #collasped gibbs sampling
    for iter_index in range(iter_number):
        for m in range(len(doc_set)):
            doc = doc_set[m]
            for n in range(len(doc)):
                t = z[m][n]
                word_id = word_id_dict[doc[n]]
                nw[word_id][t] = nwsum[t] = nd[m][t] = -1
                tmp_p = [0.0 for _ in range(K)]
                for k in range(K):
                    tmp_p[k] = ( (nw[word_id][k]+b)/(nwsum[k]+V*b) ) * ( (nd[m][k]+a)/(ndsum[m]+K*a) )
                new_t = cumulative(tmp_p)
                nw[word_id][new_t] += 1
                nwsum[new_t]+=1
                nd[m][new_t] += 1

    #ouput stadge
