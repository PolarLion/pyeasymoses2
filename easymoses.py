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
# exp_group = "smt-phrase-nnjm-wmt"
# exp_id = "9"
# exp_group = "nmt-alphabet-wmt"
# exp_id = "1"
exp_group = "nmt-wmtcb"
exp_id = "8"
# exp_group = "smt-phrase-wmtcb"
# exp_id = "180"
# exp_group = "fan-tuning"
# exp_id = "x"

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

def fan_tuning (easy_config):
  num = 3000
  for i in range(2701, num+1):
    devfilename = utils.get_filename(exp_config["develop_corpus"]+str(i))
    if not os.path.exists(os.path.join(easy_config.easy_tuning, str(i))):
      os.mkdir(os.path.join(easy_config.easy_tuning, str(i)))
    tuning_tokenizer (easy_config, str(i) + "/" + devfilename)
    tuning_truecase (easy_config, str(i) + "/" + devfilename)
    command1 = "nohup nice " + easy_config.mosesdecoder_path + "scripts/training/mert-moses.pl "\
      + "--decoder-flags=\"-threads "+exp_config["threads"]+"\""\
      + " -threads " + exp_config["threads"]\
      + " -maximum-iterations " + exp_config["tuning_max_iterations"]\
      + " -working-dir " + os.path.join(easy_config.easy_tuning, str(i))\
      + " " + os.path.join(easy_config.easy_tuning, str(i) + "/" + devfilename + ".true." + exp_config["source_id"])\
      + " " + os.path.join(easy_config.easy_tuning, str(i) + "/" + devfilename + ".true." + exp_config["target_id"])\
      + " " + easy_config.mosesdecoder_path + "bin/moses_chart " + os.path.join(easy_config.easy_train,"model/moses.ini ")\
      + " --mertdir " + easy_config.mosesdecoder_path + "bin/ &> " + os.path.join(easy_config.easy_tuning, str(i) + "/" + "mert.out") + " &"
    write_step (command1, easy_config)
    os.system(command1)

def fan_analyze(easy_config):
  devfilename = utils.get_filename(exp_config["develop_corpus"]+str(1))
  print devfilename
  standard_line = "0.0105348\t0.0651135\t0.0532412\t0.00603957\t0.0532839\t0.0809408\t0.122954\t0.271147\t0.0633011\t0.0520243\t0.0513287\t0.0239177\t0.024159\t1\t-0.122014"
  paths = os.listdir(easy_config.easy_tuning)
  outfile = open(os.path.join(easy_config.easy_tuning, "sentences.txt"),'w')
  count = 0
  for path in paths:
    if os.path.isfile(os.path.join(easy_config.easy_tuning, path)):continue
    # if os.path.isfile(os.path.join(easy_config.easy_tuning, path+"/run20.moses.ini")):
      # print path
    infile = open(os.path.join(easy_config.easy_tuning, path+"/"+devfilename + ".true." + exp_config["source_id"]), 'r')
    outfile.write(infile.readline())
    infile.close()
    dic = read_moses_ini(os.path.join(easy_config.easy_tuning, path))
    new_line = ""
    if dic :
      print path + '\t' + dic['LM0'] + '\t' + dic['TranslationModel00'] + '\t' + dic['bleu']
      for k in sorted(dic):
        if k == "bleu":continue
        # print k
        new_line += dic[k].strip() + '\t'
      outfile.write(new_line+'\n')
      # break
    else:
      count += 1
      outfile.write(standard_line + '\n')
  print count
  outfile.close()

def fan_decoder(easy_config, infilename):
  # testfilename = utils.get_filename(exp_config["test_corpus"])
  count = 0
  if not os.path.exists(os.path.join(easy_config.easy_evaluation, "fun")):
    os.mkdir(os.path.join(easy_config.easy_evaluation, "fun"))
  infile = open(infilename, 'r')
  state = 0
  for line in infile.readlines():
    if state == 0:
      outfile = open(os.path.join(easy_config.easy_evaluation, "fun/"+str(count) + '.' + exp_config["source_id"]), 'w')
      outfile.write(line)
      outfile.close()
      state = 1
    elif state == 1:
      weight_dic = weights2weightsdic(line)
      generate_moses_ini(os.path.join(easy_config.easy_evaluation, "fun/moses."+str(count)+".ini"), os.path.join(easy_config.easy_train,"model/moses.ini"), weight_dic)
      command1 = "nohup nice " + easy_config.mosesdecoder_path + "bin/moses "\
        + " -threads " + exp_config["threads"]\
        + " -f " + os.path.join(easy_config.easy_evaluation, "fun/moses."+str(count)+".ini")\
        + " < " + os.path.join(easy_config.easy_evaluation, "fun/"+str(count) + '.' + exp_config["source_id"])\
        + " > " + os.path.join(easy_config.easy_evaluation, "fun/"+str(count) + ".translated")\
        + " 2>" + os.path.join(easy_config.easy_evaluation, "fun/"+str(count) + ".out") + " "
      os.system(command1)
      count += 1
      state = 0
      # break

######################   bnplm #############################################

def add_bnplm_feature(easy_config):
  if os.path.isfile(os.path.join(easy_config.easy_train, "model/moses.ini")):
    outfile = open(os.path.join(easy_config.easy_train, "model/moses.ini"), 'a')
    easy_bnplm_feature = "[feature]\nBilingualNPLM "\
      + " order=" + exp_config["bnplm_target_context"]\
      + " source_window=" + exp_config["bnplm_source_context"]\
      + " path= " + os.path.join(easy_config.easy_bnplm, "train.10k.model.nplm."+exp_config["bnplm_epochs"])\
      + " source_vocab=" + os.path.join(easy_config.easy_bnplm, "vocab.source")\
      + " target_vocab=" + os.path.join(easy_config.easy_bnplm, "vocab.target")
    outfile.write(easy_bnplm_feature)
    write_step("add bnplm feature"+easy_bnplm_feature, easy_config)
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
  lines = int(commands.getstatusoutput('wc -l ' + os.path.join(easy_config.easy_corpus, training_filename+'.clean.'+exp_config["source_id"])) [1].split(' ')[0])
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
  # print exp_config["training_corpus"],training_filename
  # exit()
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
  # create_statefile(easy_config)
  command1 = "python " + easy_config.nmt_path + "train.py"\
    + " --proto=" + "prototype_search_state"\
    + " --state " + os.path.join(easy_config.easy_train, "state.py")\
    + " >& " + os.path.join(easy_config.easy_train, "out.txt")+" &"
  write_step (command1, easy_config)
  os.system(command1)

def nmt_test(easy_config):
  testfilename = utils.get_filename(exp_config["test_corpus"])
  t_tokenisation(easy_config, testfilename)
  t_truecasing(easy_config, testfilename)
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
  import commands
  lines = int(commands.getstatusoutput('wc -l ' + os.path.join(easy_config.easy_corpus, training_filename+'.clean.'+exp_config["source_id"]))[1].split(' ')[0])
  print "pairs ", lines
  # return 
  from nmt import overfitting_prepare
  base = 10
  if lines / 3000 > base:
    base = lines / 3000
  overfitting_prepare(easy_config, training_filename, base)
  command1 = "python " + easy_config.nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + os.path.join(easy_config.easy_train, "search_state.pkl")\
    + " --source " + os.path.join(easy_config.easy_overfitting,"OF.clean." + exp_config["source_id"])\
    + " --trans " + os.path.join(easy_config.easy_overfitting, "ontrain." + exp_config["target_id"])\
    + " " + os.path.join(easy_config.easy_train, "search_model.npz")
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl "\
    + " -lc " + easy_config.easy_overfitting + "/OF.clean." + exp_config["target_id"]\
    + " < " + easy_config.easy_overfitting + "/ontrain." + exp_config["target_id"])
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
  testfilename = utils.get_filename(exp_config["test_corpus"])
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + os.path.join(easy_config.easy_evaluation, testfilename + ".true." + exp_config["target_id"]) 
    + " < " + os.path.join(easy_config.easy_evaluation, testfilename + ".translated." + exp_config["target_id"])
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
  # bnplm (easy_config)
  # add_bnplm_feature(easy_config)
  # smt_tuning (easy_config)
  # fan_tuning(easy_config)
  # fan_analyze(easy_config)
  # fan_decoder(easy_config, "/home/xwshi/easymoses_workspace2/fan-tuning/x/tuning/sentences_weights.txt")
  # print read_moses_ini("/home/xwshi/easymoses_workspace2/fan-tuning/x/tuning/1/")
  # smt_testing (easy_config)
  # smt_check_train(easy_config)
  # cross_corpus("18", "nmt", "te", easy_config)
  # cross_corpus("17", "smt", "te", easy_config)
  # nplm (easy_config)
  
  # nmt_prepare(easy_config)
  nmt_train(easy_config)
  # nmt_check_overfitting(easy_config)
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
    exit()
  easymoses ()
  sys.path.remove(easy_config.easy_workpath)


