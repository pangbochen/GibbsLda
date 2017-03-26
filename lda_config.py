# -*- coding: utf-8 -*-
__author__ = 'pangbochen'

import configparser

class ldaConfig:
    def __init__(self):
        configFileName = "lda_setting.config"
        conf = configparser.ConfigParser()
        conf.read(configFileName)

        # load config the file path
        TAG_filepath = 'filepath'

        self.textFileName = conf.get(TAG_filepath, 'textfile')
        self.wordidFileName = conf.get(TAG_filepath, 'wordidfile')
        self.doctopicFileName = conf.get(TAG_filepath, 'thetafile')
        self.wordtopicFileName = conf.get(TAG_filepath, 'phifile')
        self.topicNFileName = conf.get(TAG_filepath, 'topNfile')
        self.topicassignFileName = conf.get(TAG_filepath, 'tassignfile')

        # load config of lda model
        TAG_model = 'lda_args'

        self.K = int(conf.get(TAG_model, "K"))
        self.alpha = float(conf.get(TAG_model, "alpha"))
        self.beta = float(conf.get(TAG_model, "beta"))
        self.iter_times = int(conf.get(TAG_model, "iter_times"))
        self.topic_words_num = int(conf.get(TAG_model, "topic_words_num"))

        print("load lda config from: "+configFileName)
