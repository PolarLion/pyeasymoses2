#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import os
import sys
import time

class EasyConfig :
	def __init__(self, exp_group="test", exp_id="0"):
		self.mosesdecoder_path = "/opt/translation/moses/"
		self.irstlm_path = "/opt/translation/irstlm/"
		self.giza_path = "/opt/translation/mgizapp/"
		self.nmt_path = "/home/xwshi/tools/GroundHog/experiments/nmt/"
		self.easy_workspace = "/home/xwshi/easymoses_workspace2/"
		self.nplm_path = "/opt/translation/nplm/"

		self.easy_experiment_id = exp_id
		self.easy_workpath = os.path.join(self.easy_workspace, exp_group)
		if not os.path.exists(self.easy_workpath):
			os.mkdir(self.easy_workpath)
		self.easy_workpath = os.path.join(self.easy_workpath, self.easy_experiment_id)
		if not os.path.exists(self.easy_workpath):
			os.mkdir(self.easy_workpath)
		self.easy_corpus = os.path.join(self.easy_workpath, "corpus")
		if not os.path.exists(self.easy_corpus):
			os.mkdir(self.easy_corpus)

		self.easy_truecaser = os.path.join(self.easy_workpath, "truecaser")
		if not os.path.exists(self.easy_truecaser):
			os.mkdir(self.easy_truecaser)

		self.easy_logs = self.easy_workpath 

		self.easy_lm = os.path.join(self.easy_workpath, "lm")
		if not os.path.exists(self.easy_lm):
			os.mkdir(self.easy_lm)

		self.easy_train = os.path.join(self.easy_workpath, "train")
		if not os.path.exists(self.easy_train):
			os.mkdir(self.easy_train)

		self.easy_tuning = os.path.join(self.easy_workpath, "tuning")
		if not os.path.exists(self.easy_tuning):
			os.mkdir(self.easy_tuning)

		self.easy_overfitting = os.path.join(self.easy_workpath, "overfitting")
		if not os.path.exists(self.easy_overfitting):
			os.mkdir(self.easy_overfitting)

		self.easy_evaluation = os.path.join(self.easy_workpath, "evalutaion")
		if not os.path.exists(self.easy_evaluation):
			os.mkdir(self.easy_evaluation)

		self.easy_bnplm = os.path.join(self.easy_workpath, "bnplm")
		if not os.path.exists(self.easy_bnplm):
			os.mkdir(self.easy_bnplm)

		self.easy_nplm = os.path.join(self.easy_workpath, "nplm")
		if not os.path.exists(self.easy_nplm):
			os.mkdir(self.easy_nplm)

		self.easy_steps = os.path.join(self.easy_workpath, "steps")
		if not os.path.exists(self.easy_steps):
			os.mkdir(self.easy_steps)

		self.preparation()
		self.make_configfile()

	def preparation (self) :
    #outfile = open(os.path.join(self.easy_logs, "log.txt"),'w')
	  #outfile.write (str (time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time())) ))
	  #outfile.close ()
	  if not os.path.isfile (os.path.join(self.easy_steps, "step.txt")):
	    outfile = open (os.path.join(self.easy_steps, "step.txt"), 'wa')
	    outfile.write (str (time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time())) ) + "\n\n")
	    outfile.close ()

	def make_configfile(self):
		import shutil
		if not os.path.isfile(os.path.join(self.easy_workpath, "config.py")):
			shutil.copy("data/config.py", os.path.join(self.easy_workpath, "config.py"))


