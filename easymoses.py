#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import sys
import os
import re
import time
import EasyHelper
import utils
import easybleu

reload(sys)
sys.setdefaultencoding('utf8') 

# exp_group = "test"
# exp_id = "4"
exp_group = "nmt-wmtcb"
# exp_group = "smt-phrase-wmtcb"
exp_id = "0"

easy_config = EasyHelper.EasyConfig(exp_group, exp_id)

# import importlib
# exp_config = importlib.import_module(os.path.join(easy_config, "config"))
sys.path.append(easy_config.easy_workpath)
from config import info as exp_config

# print exp_config

###########################################
from smt import *

def smt_training_corpus_preparation (easy_config) :
  training_filename = utils.get_filename(exp_config["training_corpus"])
  write_step("start training_corpus_preparation", easy_config)
  # print "corpus preparation"
  tokenisation (easy_config, training_filename)
  truecaser (easy_config, training_filename)
  truecasing (easy_config, training_filename)
  limiting_sentence_length (easy_config, training_filename)
  write_step("finish training_corpus_preparation", easy_config)
  # print "finish corpus preparation"

def smt_language_model_training (easy_config) :
  training_filename = utils.get_filename(exp_config["training_corpus"])
  write_step("start language_model_training", easy_config)
  generate_sb (easy_config, training_filename)
  generate_lm (easy_config, training_filename)
  generate_arpa (easy_config, training_filename)
  generate_blm (easy_config, training_filename)
  write_step("finish language_model_training", easy_config)

def smt_translation_model_training(easy_config):
  training_filename = utils.get_filename(exp_config["training_corpus"])
  translation_model(easy_config, training_filename)

def smt_tuning (easy_config) :
  devfilename = utils.get_filename(exp_config["develop_corpus"])
  # print "tuning"
  tuning_tokenizer (easy_config, devfilename)
  tuning_truecase (easy_config, devfilename)
  tuning_process (easy_config, devfilename)
  # print "finish tuning"

######################   bnplm #############################################

def add_bnplm_feature(easy_config):
  if os.path.isfile(os.path.join(easy_config.easy_train, "model/moses.ini")):
    outfile = open(os.path.join(easy_config.easy_train, "model/moses.ini"), 'wa')
    easy_bnplm_feature = "BilingualNPLM "\
      + " order=" + exp_config["target-context"]\
      + " source_window=" + exp_config["source-context"]\
      +  "path= " + os.path.join(easy_config.easy_bnplm, "train.10K.model.nplm."+exp_config["bnplm_epochs"])\
      + " source_vocab=" + os.path.join(easy_config.easy_bnplm, "vocab.source")\
      + " target_vocab=" + os.path.join(easy_config.easy_bnplm, "vocab.target")
    outfile.write()
    outfile.close()

def bnplm (easy_config) :
  training_filename = utils.get_filename(exp_config["training_corpus"])
  extract_training (easy_config, training_filename)
  train_nplm (easy_config, training_filename)
  # averagebNullEmbedding (easy_config)

#########################  test  ###########################
def smt_testing (easy_config) :
  # t_start (easy_config)
  testfilename = utils.get_filename(exp_config["test_corpus"])
  t_tokenisation (easy_config, testfilename)
  t_truecasing (easy_config, testfilename)
  # t_filter_model_given_input (easy_config)
  run_test (easy_config, testfilename)
  view_result (easy_config, testfilename)
  # compare_resultt (easy_config, 0)

def smt_check_train(easy_config):
  training_filename = utils.get_filename(exp_config["training_corpus"])
  import commands
  lines = int(commands.getstatusoutput('wc -l ' + os.path.join(easy_config.easy_corpus, training_filename+'.clean.'+exp_config["source_id"]))[1].split(' ')[0])
  print "pairs ", lines
  # return 
  from nmt import overfitting_prepare
  base = 10
  if lines / 3000 > base:
    base = lines / 3000
  overfitting_prepare(easy_config, training_filename, base)
  test_on_train(easy_config)
#########################  nmt ############################
from nmt import *
#data preparation

def nmt_prepare(easy_config):
  # cpnmt(easy_config)
  training_filename = utils.get_filename(exp_config["training_corpus"])
  write_step("start nmt_prepare", easy_config)
  tokenisation (easy_config, training_filename)
  truecaser (easy_config, training_filename)
  truecasing (easy_config, training_filename)
  limiting_sentence_length (easy_config, training_filename)
  pkl(easy_config, training_filename)
  invert(easy_config, training_filename)
  hdf5(easy_config, training_filename)
  shuff(easy_config, training_filename)
  write_step("finish nmt_prepare", easy_config)
  
def nmt_train(easy_config):
  create_statefile(easy_config)
  command1 = "python " + easy_config.nmt_path + "train.py"\
    + " --proto=" + "prototype_search_state"\
    + " --state " + os.path.join(easy_config.easy_train, "state.py")\
    + " >& " + os.path.join(easy_config.easy_train, "out.txt")+" &"
  write_step (command1, easy_config)
  os.system(command1)

def nmt_test(easy_config):
  t_tokenisation(easy_config)
  t_truecasing(easy_config)
  command1 = "python " + easy_config.nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + os.path.join(easy_config.easy_train, "search_state.pkl")\
    + " --source " + os.path.join(easy_config.easy_evaluation, testfilename + ".true." + exp_config["source_id"])\
    + " --trans " + os.path.join(easy_config.easy_evaluation, testfilename + ".translated." + exp_config["target_id"])\
    + " " + os.path.join(easy_config.easy_train, "search_model.npz")\
    + " >& " + os.path.join(easy_config.easy_evaluation, "trans_out.txt") +" &"
  write_step(command1, easy_config)
  os.system(command1)

def nmt_dev(easy_config, devfilename):
  command1 = "python " + easy_config.nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + os.path.join(easy_config.easy_train, "search_state.pkl")\
    + " --source " + os.path.join(easy_config.easy_tuning, devfilename + ".true." + exp_config["source_id"])\
    + " --trans " + os.path.join(easy_config.easy_tuning, devfilename + ".translated." + exp_config["target_id"])\
    + " " + os.path.join(easy_config.easy_train, "search_model.npz")\
    + " >& " + os.path.join(easy_config.easy_tuning, "tunning_out.txt") +" &"
  write_step(command1, easy_config)
  os.system(command1)

def nmt_dev_res(easy_config):
  command2 = easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl"\
    + " -lc " + os.path.join(easy_config.easy_tuning, devfilename + ".true." + exp_config["target_id"])\
    + " < " + os.path.join(easy_config.easy_tuning, devfilename + ".translated." + exp_config["target_id"])
  write_step (command2, easy_config)
  os.system(command2)
  translation_result = open (os.path.join(easy_config.easy_tuning, "translation_result.txt"), 'w')
  translated = open (os.path.join(easy_config.easy_tuning, devfilename + ".translated." + exp_config["target_id"]), 'r')
  source = open (os.path.join(easy_config.easy_tuning, devfilename + ".true." + exp_config["source_id"]), 'r')
  target = open (os.path.join(easy_config.easy_tuning, devfilename + ".true." + exp_config["target_id"]), 'r')
  count = 0
  for tran_line in translated :
    source_line = source.readline ()
    if source_line : translation_result.write ("[#" + str(count) + "] " + source_line)
    else : 
      print "eeeeeeeeeerror  " + str (count)
      break  
    target_line = target.readline ()
    translation_result.write ("[" + str(easybleu.bleu (tran_line, target_line)) + "] " + tran_line)
    if target_line : 
      translation_result.write ("[ref] " + target_line) 
    else :
      print "errrrrrrrrrrror" + str (count)
      break
    count += 1
  translation_result.close()
  translated.close()
  source.close()
  target.close()

def nmt_check_overfitting(easy_config):
  training_filename = utils.get_filename(exp_config["training_corpus"])
  overfitting_prepare(easy_config, training_filename, 30)
  command1 = "python " + easy_config.nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + os.path.join(easy_config.easy_train, "search_state.pkl")\
    + " --source " + os.path.join(easy_config.easy_overfitting,"OF.true." + exp_config["source_id"])\
    + " --trans " + os.path.join(easy_config.easy_overfitting, "ontrain." + exp_config["target_id"])\
    + " " + os.path.join(easy_config.easy_train, "search_model.npz")
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl "\
    + " -lc " + easy_config.easy_overfitting + "OF.true." + exp_config["target_id"]\
    + " < " + easy_config.easy_overfitting + "ontrain." + exp_config["target_id"])
  write_step(command1, easy_config)
  os.system(command1) 
  write_step(command2, easy_config)
  os.system(command2)

def nmt_make_backup(easy_config):
  dirname = easy_config.easy_train + str(time.strftime('%Y_%m%d_%H%M',time.localtime(time.time())))
  command1 = "mkdir " + dirname
  if not os.path.exists(dirname):
    write_step(command1, easy_config)
    os.system(command1)
  command2 = "cp " + easy_config.easy_train + "*.* " + dirname
  write_step(command2, easy_config)
  os.system(command2)

def bleu_score(easy_config):
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_config.easy_evaluation + testfilename + ".true." + exp_config["target_id"] 
    + " < " + easy_config.easy_evaluation + testfilename + ".translated." + exp_config["target_id"]
    # + " < " + easy_config.easy_evaluation + testfilename + ".translated." + exp_config["target_id"] + ".9"
    )
  write_step (command2, easy_config)
  os.system(command2)
#########################  nmt ############################


def easymoses ():
  a = 0
  if "true" != exp_config["config"]:
    print "please edit your config.py file"
    exit()
  # smt_training_corpus_preparation (easy_config)
  # smt_language_model_training (easy_config)
  # smt_translation_model_training (easy_config)
  # smt_tuning (easy_config)
  # smt_testing (easy_config)
  # smt_check_train(easy_config)
  # cross_corpus("18", "nmt", "te", easy_config)
  # cross_corpus("17", "smt", "te", easy_config)
  # nplm (easy_config)
  # bnplm (easy_config)
  # nmt_prepare(easy_config)
  nmt_train(easy_config)
  # nmt_check_overfitting_1(easy_config)
  # nmt_check_overfitting_2(easy_config)
  # nmt_dev(easy_config)
  # nmt_dev_res(easy_config)
  # nmt_make_backup(easy_config)
  # nmt_test(easy_config)
  # bleu_score(easy_config)

if __name__ == "__main__" :
  print str (time.strftime('%Y-%m-%d %X',time.localtime(time.time())))
  if sys.argv[1] != exp_group:
    print "you input a wrong group id"
    print "you input a wrong group id"
    print "you input a wrong group id"
    exit()
  elif sys.argv[2] != exp_id:
    print "you input a wrong exp id"
    print "you input a wrong exp id"
    print "you input a wrong exp id" 
  easymoses ()
  sys.path.remove(easy_config.easy_workpath)


