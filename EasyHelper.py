#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import os
import sys

class EasyConfig :
	self.mosesdecoder_path = "/opt/translation/moses/"
	self.irstlm_path = "/opt/translation/irstlm/"
	self.giza_path = "/opt/translation/mgizapp/"
	self.nmt_path = "/home/xwshi/tools/GroundHog/experiments/nmt/"
	self.easy_workspace = "/home/xwshi/easymoses_workspace2/"
	self.threads = "32"
	self.sentence_length = "80"

	def __init__(exp_id, workspace="", ):
		self.easy_experiment_id = exp_id
		workpath = os.path.join(self.easy_workspace, self.easy_experiment_id)
		if not os.path.exists(workpath):
			os.makedir(workpath)
			self.easy_corpus = os.path.join(workpath, "corpus")
			os.makedir(self.easy_corpus)
			self.easy_truecaser = os.path.join(workpath, "truecaser")
			os.makedir(self.easy_truecaser)
			self.easy_logs = workpath 
			self.easy_lm = os.path.join(workpath, "lm")
			os.makedir(easy_lm)
			self.easy_working = os.path.join(workpath, "train")
			os.makedir(easy_working)
			self.easy_train = os.path.join(workpath, "train")
			os.makedir(easy_train)
			self.easy_tuning = os.path.join(workpath, "tuning")
			os.makedir(easy_tuning)
			self.easy_overfitting = os.path.join(workpath, "overfitting")
			os.makedir(easy_evaluation)
			self.easy_evaluation = os.path.join(workpath, "evalutaion")
			os.makedir(easy_evaluation)
			self.easy_blm = os.path.join(workpath, "blm")
			os.makedir(easy_blm)
			self.easy_nplm = os.path.join(workpath, "nplm")
			os.makedir(easy_nplm)
			self.easy_nmt = os.path.join(workpath, "nmt")
			os.makedir(easy_nmt)
			self.easy_steps = os.path.join(workpath, "steps")
			os.makedir(easy_steps)
