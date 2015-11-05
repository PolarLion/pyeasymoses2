#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import sys
import os
import re
import time
import easybleu
import easyconfig
import easycorpus


reload(sys)
sys.setdefaultencoding('utf8') 

cfg_info = easyconfig.CfgInfo()

easy_workspace="/home/xwshi/easymoses_workspace2/"
nmt_path = "/home/xwshi/tools/GroundHog/experiments/nmt/"


# easy_experiment_id = "k0"
# easy_experiment_id = 18
# easy_experiment_id = "nmt-CB-9"
# easy_experiment_id = "smt-CB-9"
# easy_experiment_id = "test"
# easy_experiment_id = "smt-NCB-3"
easy_experiment_id = "smt-WCB-36"
easy_corpus = ""
easy_truecaser = ""
easy_logs = "" 
easy_lm = ""
easy_working = ""
easy_train = ""
easy_tuning = ""
easy_overfitting = ""
easy_evaluation = ""
easy_blm = ""
easy_nplm = ""
easy_nmt = ""
easy_steps = ""

######################### preparation #####################
def preparation (cfg_info) :
  global easy_corpus
  global easy_truecaser
  global easy_logs
  global easy_steps
  global easy_lm
  global easy_working
  global easy_train
  global easy_tuning
  global easy_evaluation
  global easy_blm
  global easy_nplm
  global easy_nmt
  global easy_overfitting
  # read_state (cfg_info)

  print "experiment id: ", easy_experiment_id
  easy_steps = easy_workspace + "steps/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_steps) : os.system ("mkdir " + easy_steps)
  easy_corpus = easy_workspace + "corpus/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_corpus) : os.system ("mkdir " + easy_corpus)
  easy_truecaser = easy_workspace + "truecaser/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_truecaser) : os.system ("mkdir " + easy_truecaser)
  easy_logs = easy_workspace + "logs/" 
  if not os.path.exists (easy_logs) : os.system ("mkdir " + easy_logs)
  easy_lm = easy_workspace + "lm/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_lm) : os.system ("mkdir " + easy_lm)
  easy_working = easy_workspace + "working/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_working) : os.system ("mkdir " + easy_working)
  easy_train = easy_workspace + "train/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_train) : os.system ("mkdir " + easy_train)
  easy_tuning = easy_workspace + "tuning/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_tuning) : os.system ("mkdir " + easy_tuning)
  easy_evaluation = easy_workspace + "evaluation/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_evaluation) : os.system ("mkdir " + easy_evaluation)
  easy_blm = easy_workspace + "blm/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_blm) : os.system ("mkdir " + easy_blm)
  easy_nplm = easy_workspace + "nplm/" + str (easy_experiment_id) + "/"
  if not os.path.exists (easy_nplm) : os.system ("mkdir " + easy_nplm)
  easy_nmt = easy_workspace + "nmt/" + str(easy_experiment_id) + "/"
  if not os.path.exists (easy_nmt) : os.system ("mkdir " + easy_nmt)
  easy_overfitting = easy_workspace + "overfitting/" + str(easy_experiment_id) + "/"
  if not os.path.exists (easy_overfitting) : os.system ("mkdir " + easy_overfitting)

  outfile = open (easy_logs + str (easy_experiment_id) + ".log", 'w')
  outfile.write (str (time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time())) ))
  outfile.close ()

  if not os.path.exists (easy_steps + str (easy_experiment_id) + ".step"):
    outfile = open (easy_steps + str (easy_experiment_id) + ".step", 'w')
    outfile.write (str (time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time())) ) + "\n\n")
    outfile.close ()

def write_step (command) :
  outfile = open (easy_steps + str (easy_experiment_id) + ".step", 'a')
  outfile.write (str (time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time())) ) + "\n")
  outfile.write ("pid: " + str(os.getpid()))
  outfile.write (command + "\n")
  outfile.close ()

######################### corpus preparation  ###########################
def tokenisation (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + cfg_info.source_id 
    + " -threads " + cfg_info.threads 
    + " -no-escape 1 "
    + " < " + cfg_info.training_corpus_path + cfg_info.filename + "." + cfg_info.source_id + " > "
    + " " + easy_corpus + cfg_info.filename + ".tok." + cfg_info.source_id )
  command2 = (cfg_info.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + cfg_info.target_id 
    + " -threads " + cfg_info.threads
    + " -no-escape 1 "
    + " < " + cfg_info.training_corpus_path + cfg_info.filename + "." + cfg_info.target_id + " > "
    + " " + easy_corpus + cfg_info.filename + ".tok." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def truecaser (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/recaser/train-truecaser.perl --model " 
     + " " + easy_truecaser + "truecase-model." + cfg_info.source_id + " --corpus " 
    + " " + easy_corpus + cfg_info.filename + ".tok." + cfg_info.source_id)
  command2 = (cfg_info.mosesdecoder_path + "scripts/recaser/train-truecaser.perl --model " 
     + " " + easy_truecaser + "truecase-model." + cfg_info.target_id + " --corpus " 
    + " " + easy_corpus + cfg_info.filename + ".tok." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)    
  write_step (command2)
  os.system (command2)    

def truecasing (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + cfg_info.source_id 
    + " < " + easy_corpus + cfg_info.filename + ".tok." + cfg_info.source_id 
    + " > " + easy_corpus + cfg_info.filename + ".true." + cfg_info.source_id)
  command2 = (cfg_info.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + cfg_info.target_id 
    + " < " + easy_corpus + cfg_info.filename + ".tok." + cfg_info.target_id 
    + " > " + easy_corpus + cfg_info.filename + ".true." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def limiting_sentence_length (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/training/clean-corpus-n.perl "
    + " " + easy_corpus + cfg_info.filename + ".true " + cfg_info.source_id + " " + cfg_info.target_id
    +" " + easy_corpus + cfg_info.filename +".clean  1 "
    +cfg_info.sentence_length)
  write_step (command1)
  os.system (command1)
######################### corpus preparation  ###########################

#########################  language model traning #######################
def generate_sb (cfg_info) :
  command1 = (cfg_info.irstlm_path + "bin/add-start-end.sh < " 
    + " " + easy_corpus + cfg_info.filename + ".true." + cfg_info.target_id 
    + " > " + easy_lm + cfg_info.filename + ".sb." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)

def generate_lm (cfg_info) :
  command1 = ("export IRSTLM=" + cfg_info.irstlm_path + "; " + cfg_info.irstlm_path + "bin/build-lm.sh " 
    + " -i " + easy_lm + cfg_info.filename + ".sb." + cfg_info.target_id 
    + " -t ./tmp -p -s improved-kneser-ney -o " + easy_lm + cfg_info.filename + ".lm." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)

def generate_arpa (cfg_info) :
  command1 = (cfg_info.irstlm_path + "bin/compile-lm --text=yes " 
    + " " + easy_lm + cfg_info.filename + ".lm." + cfg_info.target_id + ".gz " 
    + " " + easy_lm + cfg_info.filename + ".arpa." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)

def generate_blm (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "bin/build_binary " 
    + " -i "
    + " " + easy_lm + cfg_info.filename + ".arpa." + cfg_info.target_id 
    + " " + easy_lm + cfg_info.filename + ".blm." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)
#########################  language model traning #######################

#########################  training ranslation system ###########################################
def training_translation_system (cfg_info) :
  command1 = ("nohup nice " + cfg_info.mosesdecoder_path + "scripts/training/train-model.perl " 
    + " -mgiza -mgiza-cpus 16 -cores 2 "
    + " -root-dir " + easy_train 
    + " -corpus " + " " + easy_corpus + cfg_info.filename + ".clean " 
    + " -f " + cfg_info.source_id + " -e " + cfg_info.target_id 
    + " -alignment grow-diag-final-and " 
    + " -reordering msd-bidirectional-fe -lm 0:3:" + easy_lm + cfg_info.filename + ".blm." + cfg_info.target_id + ":8 " 
    # + " -reordering msd-bidirectional-fe -lm 0:4:" + "" + ":8 " 
    + " -external-bin-dir " + cfg_info.giza_path + "bin " 
    + " >& " + easy_working + "training.out &")
  write_step (command1)
  os.system (command1)

def tuning_tokenizer (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + cfg_info.source_id 
    + " -threads " + cfg_info.threads
    + " -no-escape 1 "
    + " < " + cfg_info.training_corpus_path + cfg_info.devfilename + "." + cfg_info.source_id 
    + " > " + easy_tuning + cfg_info.devfilename + ".tok." + cfg_info.source_id)
  command2 = (cfg_info.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + cfg_info.target_id
    + " -threads " + cfg_info.threads
    + " -no-escape 1 "
    + " < " + cfg_info.training_corpus_path + cfg_info.devfilename + "." + cfg_info.target_id 
    + " > " + easy_tuning + cfg_info.devfilename + ".tok." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def tuning_truecase (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + cfg_info.source_id 
    + " < " + easy_tuning + cfg_info.devfilename + ".tok." + cfg_info.source_id 
    + " > " + easy_tuning + cfg_info.devfilename + ".true." + cfg_info.source_id)
  command2 = (cfg_info.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + cfg_info.target_id 
    + " < " + easy_tuning + cfg_info.devfilename + ".tok." + cfg_info.target_id 
    + " > " + easy_tuning + cfg_info.devfilename + ".true." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def tuning_process (cfg_info) :
  command1 = ("nohup nice " + cfg_info.mosesdecoder_path + "scripts/training/mert-moses.pl " 
    + "--decoder-flags=\"-threads 32\""
    + " -threads 32" #+ cfg_info.threads
    + " -working-dir " + easy_tuning 
    + " " + easy_tuning + cfg_info.devfilename + ".true." + cfg_info.source_id 
    + " " + easy_tuning + cfg_info.devfilename + ".true." + cfg_info.target_id 
    + " " + cfg_info.mosesdecoder_path + "bin/moses " + easy_train + "model/moses.ini " 
    + " --mertdir " + cfg_info.mosesdecoder_path + "bin/ &> " + easy_tuning + "mert.out &")
  write_step (command1)
  os.system (command1)

#########################  training translation system ###########################################

def corpus_preparation (cfg_info) :
  # print "corpus preparation"
  tokenisation (cfg_info)
  truecaser (cfg_info)
  truecasing (cfg_info)
  limiting_sentence_length (cfg_info)
  # print "finish corpus preparation"

def tuning (cfg_info) :
  # print "tuning"
  tuning_tokenizer (cfg_info)
  tuning_truecase (cfg_info)
  tuning_process (cfg_info)
  # print "finish tuning"

def language_model_training (cfg_info) :
  generate_sb (cfg_info)
  generate_lm (cfg_info)
  generate_arpa (cfg_info)
  generate_blm (cfg_info)

######################   bnplm #############################################
def extract_training (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/training/bilingual-lm/extract_training.py "
    + " --working-dir " + easy_blm
    + " --corpus " + easy_corpus + cfg_info.filename + ".clean " 
    + " --source-language " + cfg_info.source_id  
    + " --target-language " + cfg_info.target_id 
    + " --align " + easy_train + "/model/aligned.grow-diag-final-and " 
    + " --prune-target-vocab 20000 " 
    + " --prune-source-vocab 20000 " 
    + " --target-context 5 " 
    + " --source-context 2 ")
  write_step (command1)
  os.system (command1)

def train_nplm (cfg_info) : 
  command1 = (cfg_info.mosesdecoder_path + "scripts/training/bilingual-lm/train_nplm.py "
    + " --working-dir " + easy_blm 
    + " --corpus " + easy_corpus + cfg_info.filename + ".clean " 
    + " --nplm-home " + cfg_info.nplm_path 
    + " --ngram-size 10 " 
    + " --epochs 40 " 
    + " --learning-rate 0.7 "
    # + " --input_vocab_size 20000 " 
    # + " --output_vocab_size 20000 " 
    + " --hidden 512 "
    + " --input-embedding 150 "
    + " --output-embedding 150 " 
    + " --threads " + cfg_info.threads
    + " &> nplm.out &")
  write_step (command1)
  os.system (command1)

def averagebNullEmbedding (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/training/bilingual-lm/averageNullEmbedding.py " 
    + " -p " + cfg_info.nplm_path + "python " 
    + " -i " + easy_blm + "train.10k.model.nplm.40 "
    + " -o " + easy_blm + "blm.blm " 
    + " -t " + easy_blm + "CHT.Train.clean.ngrams ")
  write_step (command1)
  os.system (command1)

def bnplm (cfg_info) :
  # extract_training (cfg_info)
  # train_nplm (cfg_info)
  averagebNullEmbedding (cfg_info)

####################### testing #############################################

def t_tokenisation (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + cfg_info.source_id 
    + " -threads " + cfg_info.threads
    + " -no-escape 1 "
    + " < " + cfg_info.test_corpus_path + cfg_info.testfilename + "." + cfg_info.source_id 
    + " > " + easy_evaluation + cfg_info.testfilename + ".tok." + cfg_info.source_id)
  command2 = (cfg_info.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + cfg_info.target_id 
    + " -threads " + cfg_info.threads
    + " -no-escape 1 "
    + " < " + cfg_info.test_corpus_path + cfg_info.testfilename + "." + cfg_info.target_id 
    + " > " + easy_evaluation + cfg_info.testfilename + ".tok." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)
  
def t_truecasing (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
     + " " + easy_truecaser + "truecase-model." + cfg_info.source_id 
    + " < " + easy_evaluation + cfg_info.testfilename  + ".tok." + cfg_info.source_id
    # + " < " + easy_evaluation + cfg_info.testfilename  + ".translated." + cfg_info.target_id
    + " > " + easy_evaluation + cfg_info.testfilename  + ".true." + cfg_info.source_id)
    # + " > " + easy_evaluation + cfg_info.testfilename  + ".translated.true." + cfg_info.target_id)
  command2 = (cfg_info.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
     + " " + easy_truecaser + "truecase-model." + cfg_info.target_id 
    + " < " + easy_evaluation + cfg_info.testfilename  + ".tok." + cfg_info.target_id 
    + " > " + easy_evaluation + cfg_info.testfilename  + ".true." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)
 
def t_filter_model_given_input (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/training/filter-model-given-input.pl " 
    + " " + easy_evaluation + "filtered-" + cfg_info.testfilename 
    + " " + cfg_info.working_path + "moses.ini " 
    + " " + test_corpus_path + test_filename + ".true." + cfg_info.source_id 
    + " -Binarizer " + cfg_info.mosesdecoder_path + "bin/processPhraseTableMin")
  write_step (command1)
  os.system (command1)

def run_test (cfg_info) :
  command1 = ("nohup nice " + cfg_info.mosesdecoder_path + "bin/moses "
    + " -threads " + cfg_info.threads
    + " -f " + easy_tuning + "moses.ini "
    # + " -f /home/xwshi/easymoses_workspace/tuning/8/moses.ini "
    # + " -f " + easy_tuning + "run6.moses.ini"#"moses.ini " 
    #+ cfg_info.working_path + "filtered-" + test_filename + "/moses.ini " \
    #+ " -i " + cfg_info.working_path + "filtered-" + test_filename + "/input.115575 " \
    + " < " + easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.source_id 
    + " > " + easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id 
    + " 2> " + easy_evaluation + cfg_info.testfilename + ".out ")
  command2 = (cfg_info.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.target_id 
    + " < " + easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id
    # + " < " + easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id + ".9"
    )
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def view_result (cfg_info) :
  translation_result = open (easy_evaluation + "translation_result.txt", 'w')
  translated = open (easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id, 'r')
  # translated = open (easy_evaluation + "CHT.Test.translated.en.918", 'r')
  source = open (easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.source_id, 'r')
  target = open (easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.target_id, 'r')
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

def compare_resultt (cfg_info, exp_id):
  result1 = open (easy_evaluation + "translation_result.txt", 'r')
  result2 = open (easy_workspace + "evaluation/" + str (exp_id) + "/translation_result.txt", 'r')
  result_set = {}
  pattern = re.compile (r'\[\d.(\d+)\]')
  count = 0
  while result1:
      count += 1
      l1 = result1.readline ()
      if l1 == "": break
      #print "xx ", l1
      l2 = result2.readline ()
      #print "xxx ", l2
      item = []
      item.append (l1)
      l1 = result1.readline ()
      l2 = result2.readline ()
      match1 = pattern.match (l1)
      item.append (l1)
      match2 = pattern.match (l2)
      item.append (l2)
      score1 = -1.0
      score2 = -1.0
      if match1 and match2:
          score1 = float (match1.group ().strip (r'\[?\]?'))
          score2 = float (match2.group ().strip (r'\[?\]?'))
      #else :
          #print "error error error"
      l1 = result1.readline ()
      l2 = result2.readline ()
      item.append (l1)
      key = score1 - score2
      if key in result_set :
          result_set [key].append (item)
      else:
          result_set [key] = []
          result_set [key].append (item)
      # if count > 0 : break
  print count
  result1.close ()
  result2.close ()
  compare = open (easy_evaluation + "compare_to_" + str(exp_id) + ".txt", 'w')
  better_count = 0
  worse_count = 0
  equal_count = 0
  total = 0
  for lst in sorted (result_set.iteritems (), key=lambda d:d[0] , reverse = True):
      # print type (lst), lst
      for lstlst in lst[1]:
        if 0 < lst [0] : 
          better_count += 1
        elif 0 > lst[0] : 
          worse_count += 1
        else :
          equal_count += 1
        total += 1
        for line in lstlst :
          compare.write (line)
  compare.write ("better : " + str (better_count))
  compare.write ("  worse : " + str (worse_count))
  compare.write ("  same : " + str (equal_count))
  compare.write ("  total : " + str (total))
  print better_count, worse_count, equal_count, total
  compare.close ()

def testing (cfg_info) :
  # t_start (cfg_info)
  t_tokenisation (cfg_info)
  t_truecasing (cfg_info)
  #t_filter_model_given_input (cfg_info)
  run_test (cfg_info)
  view_result (cfg_info)
  compare_resultt (cfg_info, 0)
#########################  test  ###########################

def overfitting_prepare(cfg_info):
  sampling_base = 50
  easycorpus.sampling_file(easy_corpus+cfg_info.filename+".true."+cfg_info.source_id, 
    easy_overfitting+"OF.true."+cfg_info.source_id, sampling_base)
  easycorpus.sampling_file(easy_corpus+cfg_info.filename+".true."+cfg_info.target_id, 
    easy_overfitting+"OF.true."+cfg_info.target_id, sampling_base)
  write_step("overfitting_prepare")

#########################  nmt ############################

source_voc = "10000"
target_voc = "40000"

#data preparation
def pkl (cfg_info):
  command1 = "python " + nmt_path + "preprocess/preprocess.py "\
    + easy_corpus + cfg_info.filename  + ".true." + cfg_info.source_id\
    + " -d " + easy_corpus + "vocab." + cfg_info.source_id + ".pkl "\
    + " -v " + source_voc\
    + " -b " + easy_corpus + "binarized_text." + cfg_info.source_id + ".pkl"\
    + " -p " #+ easy_corpus + "*en.txt.gz"
  command2 = "python " + nmt_path + "preprocess/preprocess.py " \
    + easy_corpus + cfg_info.filename  + ".true." + cfg_info.target_id\
    + " -d " + easy_corpus + "vocab." + cfg_info.target_id + ".pkl "\
    + " -v " + target_voc\
    + " -b " + easy_corpus + "binarized_text." + cfg_info.target_id + ".pkl"\
    + " -p " #+ easy_corpus + "*en.txt.gz"
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def invert(cfg_info):
  print "-----------  invert ------------"
  command1 = "python " + nmt_path + "preprocess/invert-dict.py " \
    + " " + easy_corpus + "vocab." + cfg_info.source_id + ".pkl "\
    + " " + easy_corpus + "ivocab." + cfg_info.source_id + ".pkl "
  command2 = "python " + nmt_path + "preprocess/invert-dict.py " \
    + " " + easy_corpus + "vocab." + cfg_info.target_id + ".pkl "\
    + " " + easy_corpus + "ivocab." + cfg_info.target_id + ".pkl "
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def hdf5(cfg_info):
  command1 = "python " + nmt_path + "preprocess/convert-pkl2hdf5.py " \
    + " " + easy_corpus + "binarized_text." + cfg_info.source_id + ".pkl"\
    + " " + easy_corpus + "binarized_text." + cfg_info.source_id + ".h5"
  command2 = "python " + nmt_path + "preprocess/convert-pkl2hdf5.py " \
    + " " + easy_corpus + "binarized_text." + cfg_info.target_id + ".pkl"\
    + " " + easy_corpus + "binarized_text." + cfg_info.target_id + ".h5"
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def shuff(cfg_info):
  command1 = "python " + nmt_path + "preprocess/shuffle-hdf5.py " \
    + " " + easy_corpus + "binarized_text." + cfg_info.source_id + ".h5"\
    + " " + easy_corpus + "binarized_text." + cfg_info.target_id + ".h5"\
    + " " + easy_corpus + "binarized_text." + cfg_info.source_id + ".shuf.h5"\
    + " " + easy_corpus + "binarized_text." + cfg_info.target_id + ".shuf.h5"
  write_step (command1)
  os.system (command1)

def nmt_prepare(cfg_info):
  # cpnmt(cfg_info)
  pkl(cfg_info)
  invert(cfg_info)
  hdf5(cfg_info)
  shuff(cfg_info)

def nmt_train(cfg_info):
  # command1 = "python " + easy_nmt + "GroundHog/experiments/nmt/train.py "\
    # + " --proto=" + "prototype_search_state "
  # print easy_nmt, "======"
  command1 = "python " + nmt_path + "train.py"\
    + " --proto=" + "prototype_search_state"\
    + " --state " + easy_nmt + "state.py"\
    + " >& " + easy_nmt + "out.txt &"
  write_step (command1)
  os.system (command1)

def nmt_test(cfg_info):
  t_tokenisation(cfg_info)
  t_truecasing(cfg_info)
  command1 = "python " + nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_nmt + "search_state.pkl "\
    + " --source " + easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.source_id\
    + " --trans " + easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id\
    + " " + easy_nmt + "search_model.npz"\
    + " >& " + easy_evaluation + "trans_out.txt &"
  write_step(command1)
  os.system(command1)

def nmt_dev(cfg_info):
  command1 = "python " + nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_nmt + "search_state.pkl "\
    + " --source " + easy_tuning + cfg_info.devfilename + ".true." + cfg_info.source_id\
    + " --trans " + easy_tuning + cfg_info.devfilename + ".translated." + cfg_info.target_id\
    + " " + easy_nmt + "search_model.npz"\
    + " >& " + easy_tuning + "trans_out.txt &"
  write_step(command1)
  os.system(command1)

def nmt_dev_res(cfg_info):
  command2 = (cfg_info.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_tuning + cfg_info.devfilename + ".true." + cfg_info.target_id 
    + " < " + easy_tuning + cfg_info.devfilename + ".translated." + cfg_info.target_id
    # + " < " + easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id + ".9"
    )
  write_step (command2)
  os.system (command2)
  translation_result = open (easy_tuning + "translation_result.txt", 'w')
  translated = open (easy_tuning + cfg_info.devfilename + ".translated." + cfg_info.target_id, 'r')
  source = open (easy_tuning + cfg_info.devfilename + ".true." + cfg_info.source_id, 'r')
  target = open (easy_tuning + cfg_info.devfilename + ".true." + cfg_info.target_id, 'r')
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

def nmt_check_overfitting_1(cfg_info):
  command1 = "python " + nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_nmt + "search_state.pkl "\
    + " --source " + easy_overfitting + "OF.true." + cfg_info.source_id\
    + " --trans " + easy_overfitting + "ontrain." + cfg_info.target_id\
    + " " + easy_nmt + "search_model.npz"\
    + " >& " + easy_overfitting + "check_overfitting_out.txt &"
  write_step(command1)
  os.system(command1) 

def nmt_check_overfitting_2(cfg_info):
  command2 = (cfg_info.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_overfitting + "OF.true." + cfg_info.target_id 
    + " < " + easy_overfitting + "ontrain." + cfg_info.target_id
    # + " < " + easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id + ".9"
    )
  write_step (command2)
  os.system (command2)

def nmt_make_backup(cfg_info):
  dirname = easy_nmt + str(time.strftime('%Y_%m%d_%H%M',time.localtime(time.time())))
  command1 = "mkdir " + dirname
  if not os.path.exists(dirname):
    write_step(command1)
    os.system(command1)
  command2 = "cp " + easy_nmt + "*.* " + dirname
  write_step(command2)
  os.system(command2)


def bleu_score(cfg_info):
  command2 = (cfg_info.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.target_id 
    + " < " + easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id
    # + " < " + easy_evaluation + cfg_info.testfilename + ".translated." + cfg_info.target_id + ".9"
    )
  write_step (command2)
  os.system (command2)
#########################  nmt ############################


def easymoses ():
  preparation (cfg_info)
  corpus_preparation (cfg_info)
  language_model_training (cfg_info)
  training_translation_system (cfg_info)
  # tuning (cfg_info)
  # testing (cfg_info)
  # cross_corpus("18", "nmt", "te", cfg_info)
  # cross_corpus("17", "smt", "te", cfg_info)
  # nplm (cfg_info)
  # bnplm (cfg_info)
  # nmt_prepare(cfg_info)
  # nmt_train(cfg_info)
  # overfitting_prepare(cfg_info)
  # nmt_check_overfitting_1(cfg_info)
  # nmt_check_overfitting_2(cfg_info)
  # nmt_dev(cfg_info)
  # nmt_dev_res(cfg_info)
  # nmt_make_backup(cfg_info)
  # nmt_test(cfg_info)
  # bleu_score(cfg_info)

if __name__ == "__main__" :
  print str (time.strftime('%Y-%m-%d %X',time.localtime(time.time())))
  if sys.argv[1] != easy_experiment_id:
    print "you input a wrong experiment id"
    exit()
  easymoses ()


######################### Training NPLM #############################
def prepare_corpus (cfg_info) :
  command1 = (cfg_info.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + cfg_info.target_id 
    + " -threads " + cfg_info.threads
    + " -no-escape 1 "
    + " < " + cfg_info.training_corpus_path + cfg_info.filename + "." + cfg_info.target_id 
    + " > " + easy_nplm + cfg_info.filename + ".tok." + cfg_info.target_id)
  command2 = (cfg_info.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + cfg_info.target_id 
    + " < " + easy_nplm + cfg_info.filename + ".tok." + cfg_info.target_id 
    + " > " + easy_nplm + cfg_info.filename + ".true." + cfg_info.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def prepare_neural_language_model (cfg_info) :
  command1 = (cfg_info.nplm_path + "bin/prepareNeuralLM " 
    + " --train_text " + easy_nplm + cfg_info.filename  + ".true." + cfg_info.target_id
    + " --ngram_size 5 " 
    + " --vocab_size 20000 "  
    + " --write_words_file " + easy_nplm + "words " 
    + " --train_file " + easy_nplm + "train.ngrams " 
    + " --validation_size 500 "
    + " --validation_file " + easy_nplm + "validation.ngrams " 
    + " >& " + easy_nplm + "prepareout.out &")
  write_step (command1)
  os.system (command1)

def train_neural_network (cfg_info) :
  command1 = (cfg_info.nplm_path + "bin/trainNeuralNetwork " 
    + " --train_file " + easy_nplm + "train.ngrams " 
    + " --validation_file " + easy_nplm + "validation.ngrams " 
    + " --num_epochs 30 "
    + " --input_words_file " + easy_nplm + "words " 
    + " --model_prefix " + easy_nplm + "model " 
    + " --input_embedding_dimension 150 "  
    + " --num_hidden 0" 
    + " --output_embedding_dimension 750 "
     + " --num_threads "+ cfg_info.threads 
    + " >& " + easy_nplm + "nplmtrain.out &")
  write_step (command1)
  os.system (command1)

def nplm (cfg_info) :
  prepare_corpus (cfg_info)
  # prepare_neural_language_model (cfg_info)
  # train_neural_network (cfg_info)

def cross_corpus(id1, mt_type, tag, cfg_info):
  command1 = "python " + nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_workspace + "nmt/" + id1 + "/" + "search_state.pkl "\
    + " --source " + easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.source_id\
    + " --trans " + easy_evaluation + cfg_info.testfilename + ".translated." + id1 + "." + cfg_info.target_id\
    + " " + easy_workspace + "nmt/" + id1 + "/" + "search_model.npz"\
    + " >& " + easy_evaluation + id1 + "_out.txt &"
  command2 = "nohup nice " + cfg_info.mosesdecoder_path + "bin/moses "\
    + " -threads " + cfg_info.threads\
    + " -f " + easy_workspace + "tuning/" + id1 + "/"+  "moses.ini "\
    + " < " + easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.source_id\
    + " > " + easy_evaluation + cfg_info.testfilename + ".translated." + id1 + "." + cfg_info.target_id\
    + " 2> " + easy_evaluation + id1 + "_out.txt"
  command3 = cfg_info.mosesdecoder_path + "scripts/generic/multi-bleu.perl "\
    + " -lc " + easy_evaluation + cfg_info.testfilename + ".true." + cfg_info.target_id\
    + " < " + easy_evaluation + cfg_info.testfilename + ".translated." + id1 + "." + cfg_info.target_id
  if tag == "tr" and mt_type == "nmt":
    write_step(command1)
    os.system(command1)
  elif tag == "tr" and mt_type == "smt":
    write_step(command2)
    os.system(command2)
  if tag == "te":
    write_step(command3)
    os.system(command3)