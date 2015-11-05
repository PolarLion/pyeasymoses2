#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import sys
import os
import re
import time
import easybleu
import EasyHelper
import easycorpus


reload(sys)
sys.setdefaultencoding('utf8') 


exp_group = "test"
exp_id = "0"

easy_config = EasyHelper.EasyConfig("test", "0")


def write_step (command) :
  outfile = open (os.path.join(easy_config.easy_steps, "step.txt"), 'a')
  outfile.write (str (time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time())) ) + "\n")
  outfile.write ("pid: " + str(os.getpid()))
  outfile.write (command + "\n")
  outfile.close ()

######################### corpus preparation  ###########################
def tokenisation (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + easy_config.source_id 
    + " -threads " + easy_config.threads 
    + " -no-escape 1 "
    + " < " + easy_config.training_corpus_path + easy_config.filename + "." + easy_config.source_id + " > "
    + " " + easy_corpus + easy_config.filename + ".tok." + easy_config.source_id )
  command2 = (easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + easy_config.target_id 
    + " -threads " + easy_config.threads
    + " -no-escape 1 "
    + " < " + easy_config.training_corpus_path + easy_config.filename + "." + easy_config.target_id + " > "
    + " " + easy_corpus + easy_config.filename + ".tok." + easy_config.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def truecaser (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/recaser/train-truecaser.perl --model " 
     + " " + easy_truecaser + "truecase-model." + easy_config.source_id + " --corpus " 
    + " " + easy_corpus + easy_config.filename + ".tok." + easy_config.source_id)
  command2 = (easy_config.mosesdecoder_path + "scripts/recaser/train-truecaser.perl --model " 
     + " " + easy_truecaser + "truecase-model." + easy_config.target_id + " --corpus " 
    + " " + easy_corpus + easy_config.filename + ".tok." + easy_config.target_id)
  write_step (command1)
  os.system (command1)    
  write_step (command2)
  os.system (command2)    

def truecasing (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + easy_config.source_id 
    + " < " + easy_corpus + easy_config.filename + ".tok." + easy_config.source_id 
    + " > " + easy_corpus + easy_config.filename + ".true." + easy_config.source_id)
  command2 = (easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + easy_config.target_id 
    + " < " + easy_corpus + easy_config.filename + ".tok." + easy_config.target_id 
    + " > " + easy_corpus + easy_config.filename + ".true." + easy_config.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def limiting_sentence_length (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/training/clean-corpus-n.perl "
    + " " + easy_corpus + easy_config.filename + ".true " + easy_config.source_id + " " + easy_config.target_id
    +" " + easy_corpus + easy_config.filename +".clean  1 "
    +easy_config.sentence_length)
  write_step (command1)
  os.system (command1)
######################### corpus preparation  ###########################

#########################  language model traning #######################
def generate_sb (easy_config) :
  command1 = (easy_config.irstlm_path + "bin/add-start-end.sh < " 
    + " " + easy_corpus + easy_config.filename + ".true." + easy_config.target_id 
    + " > " + easy_lm + easy_config.filename + ".sb." + easy_config.target_id)
  write_step (command1)
  os.system (command1)

def generate_lm (easy_config) :
  command1 = ("export IRSTLM=" + easy_config.irstlm_path + "; " + easy_config.irstlm_path + "bin/build-lm.sh " 
    + " -i " + easy_lm + easy_config.filename + ".sb." + easy_config.target_id 
    + " -t ./tmp -p -s improved-kneser-ney -o " + easy_lm + easy_config.filename + ".lm." + easy_config.target_id)
  write_step (command1)
  os.system (command1)

def generate_arpa (easy_config) :
  command1 = (easy_config.irstlm_path + "bin/compile-lm --text=yes " 
    + " " + easy_lm + easy_config.filename + ".lm." + easy_config.target_id + ".gz " 
    + " " + easy_lm + easy_config.filename + ".arpa." + easy_config.target_id)
  write_step (command1)
  os.system (command1)

def generate_blm (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "bin/build_binary " 
    + " -i "
    + " " + easy_lm + easy_config.filename + ".arpa." + easy_config.target_id 
    + " " + easy_lm + easy_config.filename + ".blm." + easy_config.target_id)
  write_step (command1)
  os.system (command1)
#########################  language model traning #######################

#########################  training ranslation system ###########################################
def training_translation_system (easy_config) :
  command1 = ("nohup nice " + easy_config.mosesdecoder_path + "scripts/training/train-model.perl " 
    + " -mgiza -mgiza-cpus 16 -cores 2 "
    + " -root-dir " + easy_train 
    + " -corpus " + " " + easy_corpus + easy_config.filename + ".clean " 
    + " -f " + easy_config.source_id + " -e " + easy_config.target_id 
    + " -alignment grow-diag-final-and " 
    + " -reordering msd-bidirectional-fe -lm 0:3:" + easy_lm + easy_config.filename + ".blm." + easy_config.target_id + ":8 " 
    # + " -reordering msd-bidirectional-fe -lm 0:4:" + "" + ":8 " 
    + " -external-bin-dir " + easy_config.giza_path + "bin " 
    + " >& " + easy_working + "training.out &")
  write_step (command1)
  os.system (command1)

def tuning_tokenizer (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + easy_config.source_id 
    + " -threads " + easy_config.threads
    + " -no-escape 1 "
    + " < " + easy_config.training_corpus_path + easy_config.devfilename + "." + easy_config.source_id 
    + " > " + easy_tuning + easy_config.devfilename + ".tok." + easy_config.source_id)
  command2 = (easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + easy_config.target_id
    + " -threads " + easy_config.threads
    + " -no-escape 1 "
    + " < " + easy_config.training_corpus_path + easy_config.devfilename + "." + easy_config.target_id 
    + " > " + easy_tuning + easy_config.devfilename + ".tok." + easy_config.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def tuning_truecase (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + easy_config.source_id 
    + " < " + easy_tuning + easy_config.devfilename + ".tok." + easy_config.source_id 
    + " > " + easy_tuning + easy_config.devfilename + ".true." + easy_config.source_id)
  command2 = (easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + easy_config.target_id 
    + " < " + easy_tuning + easy_config.devfilename + ".tok." + easy_config.target_id 
    + " > " + easy_tuning + easy_config.devfilename + ".true." + easy_config.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def tuning_process (easy_config) :
  command1 = ("nohup nice " + easy_config.mosesdecoder_path + "scripts/training/mert-moses.pl " 
    + "--decoder-flags=\"-threads 32\""
    + " -threads 32" #+ easy_config.threads
    + " -working-dir " + easy_tuning 
    + " " + easy_tuning + easy_config.devfilename + ".true." + easy_config.source_id 
    + " " + easy_tuning + easy_config.devfilename + ".true." + easy_config.target_id 
    + " " + easy_config.mosesdecoder_path + "bin/moses " + easy_train + "model/moses.ini " 
    + " --mertdir " + easy_config.mosesdecoder_path + "bin/ &> " + easy_tuning + "mert.out &")
  write_step (command1)
  os.system (command1)

#########################  training translation system ###########################################

def corpus_preparation (easy_config) :
  # print "corpus preparation"
  tokenisation (easy_config)
  truecaser (easy_config)
  truecasing (easy_config)
  limiting_sentence_length (easy_config)
  # print "finish corpus preparation"

def tuning (easy_config) :
  # print "tuning"
  tuning_tokenizer (easy_config)
  tuning_truecase (easy_config)
  tuning_process (easy_config)
  # print "finish tuning"

def language_model_training (easy_config) :
  generate_sb (easy_config)
  generate_lm (easy_config)
  generate_arpa (easy_config)
  generate_blm (easy_config)

######################   bnplm #############################################
def extract_training (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/training/bilingual-lm/extract_training.py "
    + " --working-dir " + easy_blm
    + " --corpus " + easy_corpus + easy_config.filename + ".clean " 
    + " --source-language " + easy_config.source_id  
    + " --target-language " + easy_config.target_id 
    + " --align " + easy_train + "/model/aligned.grow-diag-final-and " 
    + " --prune-target-vocab 20000 " 
    + " --prune-source-vocab 20000 " 
    + " --target-context 5 " 
    + " --source-context 2 ")
  write_step (command1)
  os.system (command1)

def train_nplm (easy_config) : 
  command1 = (easy_config.mosesdecoder_path + "scripts/training/bilingual-lm/train_nplm.py "
    + " --working-dir " + easy_blm 
    + " --corpus " + easy_corpus + easy_config.filename + ".clean " 
    + " --nplm-home " + easy_config.nplm_path 
    + " --ngram-size 10 " 
    + " --epochs 40 " 
    + " --learning-rate 0.7 "
    # + " --input_vocab_size 20000 " 
    # + " --output_vocab_size 20000 " 
    + " --hidden 512 "
    + " --input-embedding 150 "
    + " --output-embedding 150 " 
    + " --threads " + easy_config.threads
    + " &> nplm.out &")
  write_step (command1)
  os.system (command1)

def averagebNullEmbedding (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/training/bilingual-lm/averageNullEmbedding.py " 
    + " -p " + easy_config.nplm_path + "python " 
    + " -i " + easy_blm + "train.10k.model.nplm.40 "
    + " -o " + easy_blm + "blm.blm " 
    + " -t " + easy_blm + "CHT.Train.clean.ngrams ")
  write_step (command1)
  os.system (command1)

def bnplm (easy_config) :
  # extract_training (easy_config)
  # train_nplm (easy_config)
  averagebNullEmbedding (easy_config)

####################### testing #############################################

def t_tokenisation (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + easy_config.source_id 
    + " -threads " + easy_config.threads
    + " -no-escape 1 "
    + " < " + easy_config.test_corpus_path + easy_config.testfilename + "." + easy_config.source_id 
    + " > " + easy_evaluation + easy_config.testfilename + ".tok." + easy_config.source_id)
  command2 = (easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + easy_config.target_id 
    + " -threads " + easy_config.threads
    + " -no-escape 1 "
    + " < " + easy_config.test_corpus_path + easy_config.testfilename + "." + easy_config.target_id 
    + " > " + easy_evaluation + easy_config.testfilename + ".tok." + easy_config.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)
  
def t_truecasing (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
     + " " + easy_truecaser + "truecase-model." + easy_config.source_id 
    + " < " + easy_evaluation + easy_config.testfilename  + ".tok." + easy_config.source_id
    # + " < " + easy_evaluation + easy_config.testfilename  + ".translated." + easy_config.target_id
    + " > " + easy_evaluation + easy_config.testfilename  + ".true." + easy_config.source_id)
    # + " > " + easy_evaluation + easy_config.testfilename  + ".translated.true." + easy_config.target_id)
  command2 = (easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
     + " " + easy_truecaser + "truecase-model." + easy_config.target_id 
    + " < " + easy_evaluation + easy_config.testfilename  + ".tok." + easy_config.target_id 
    + " > " + easy_evaluation + easy_config.testfilename  + ".true." + easy_config.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)
 
def t_filter_model_given_input (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/training/filter-model-given-input.pl " 
    + " " + easy_evaluation + "filtered-" + easy_config.testfilename 
    + " " + easy_config.working_path + "moses.ini " 
    + " " + test_corpus_path + test_filename + ".true." + easy_config.source_id 
    + " -Binarizer " + easy_config.mosesdecoder_path + "bin/processPhraseTableMin")
  write_step (command1)
  os.system (command1)

def run_test (easy_config) :
  command1 = ("nohup nice " + easy_config.mosesdecoder_path + "bin/moses "
    + " -threads " + easy_config.threads
    + " -f " + easy_tuning + "moses.ini "
    # + " -f /home/xwshi/easymoses_workspace/tuning/8/moses.ini "
    # + " -f " + easy_tuning + "run6.moses.ini"#"moses.ini " 
    #+ easy_config.working_path + "filtered-" + test_filename + "/moses.ini " \
    #+ " -i " + easy_config.working_path + "filtered-" + test_filename + "/input.115575 " \
    + " < " + easy_evaluation + easy_config.testfilename + ".true." + easy_config.source_id 
    + " > " + easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id 
    + " 2> " + easy_evaluation + easy_config.testfilename + ".out ")
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_evaluation + easy_config.testfilename + ".true." + easy_config.target_id 
    + " < " + easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id
    # + " < " + easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id + ".9"
    )
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def view_result (easy_config) :
  translation_result = open (easy_evaluation + "translation_result.txt", 'w')
  translated = open (easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id, 'r')
  # translated = open (easy_evaluation + "CHT.Test.translated.en.918", 'r')
  source = open (easy_evaluation + easy_config.testfilename + ".true." + easy_config.source_id, 'r')
  target = open (easy_evaluation + easy_config.testfilename + ".true." + easy_config.target_id, 'r')
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

def compare_resultt (easy_config, exp_id):
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

def testing (easy_config) :
  # t_start (easy_config)
  t_tokenisation (easy_config)
  t_truecasing (easy_config)
  #t_filter_model_given_input (easy_config)
  run_test (easy_config)
  view_result (easy_config)
  compare_resultt (easy_config, 0)
#########################  test  ###########################

def overfitting_prepare(easy_config):
  sampling_base = 50
  easycorpus.sampling_file(easy_corpus+easy_config.filename+".true."+easy_config.source_id, 
    easy_overfitting+"OF.true."+easy_config.source_id, sampling_base)
  easycorpus.sampling_file(easy_corpus+easy_config.filename+".true."+easy_config.target_id, 
    easy_overfitting+"OF.true."+easy_config.target_id, sampling_base)
  write_step("overfitting_prepare")

#########################  nmt ############################

source_voc = "10000"
target_voc = "40000"

#data preparation
def pkl (easy_config):
  command1 = "python " + nmt_path + "preprocess/preprocess.py "\
    + easy_corpus + easy_config.filename  + ".true." + easy_config.source_id\
    + " -d " + easy_corpus + "vocab." + easy_config.source_id + ".pkl "\
    + " -v " + source_voc\
    + " -b " + easy_corpus + "binarized_text." + easy_config.source_id + ".pkl"\
    + " -p " #+ easy_corpus + "*en.txt.gz"
  command2 = "python " + nmt_path + "preprocess/preprocess.py " \
    + easy_corpus + easy_config.filename  + ".true." + easy_config.target_id\
    + " -d " + easy_corpus + "vocab." + easy_config.target_id + ".pkl "\
    + " -v " + target_voc\
    + " -b " + easy_corpus + "binarized_text." + easy_config.target_id + ".pkl"\
    + " -p " #+ easy_corpus + "*en.txt.gz"
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def invert(easy_config):
  print "-----------  invert ------------"
  command1 = "python " + nmt_path + "preprocess/invert-dict.py " \
    + " " + easy_corpus + "vocab." + easy_config.source_id + ".pkl "\
    + " " + easy_corpus + "ivocab." + easy_config.source_id + ".pkl "
  command2 = "python " + nmt_path + "preprocess/invert-dict.py " \
    + " " + easy_corpus + "vocab." + easy_config.target_id + ".pkl "\
    + " " + easy_corpus + "ivocab." + easy_config.target_id + ".pkl "
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def hdf5(easy_config):
  command1 = "python " + nmt_path + "preprocess/convert-pkl2hdf5.py " \
    + " " + easy_corpus + "binarized_text." + easy_config.source_id + ".pkl"\
    + " " + easy_corpus + "binarized_text." + easy_config.source_id + ".h5"
  command2 = "python " + nmt_path + "preprocess/convert-pkl2hdf5.py " \
    + " " + easy_corpus + "binarized_text." + easy_config.target_id + ".pkl"\
    + " " + easy_corpus + "binarized_text." + easy_config.target_id + ".h5"
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def shuff(easy_config):
  command1 = "python " + nmt_path + "preprocess/shuffle-hdf5.py " \
    + " " + easy_corpus + "binarized_text." + easy_config.source_id + ".h5"\
    + " " + easy_corpus + "binarized_text." + easy_config.target_id + ".h5"\
    + " " + easy_corpus + "binarized_text." + easy_config.source_id + ".shuf.h5"\
    + " " + easy_corpus + "binarized_text." + easy_config.target_id + ".shuf.h5"
  write_step (command1)
  os.system (command1)

def nmt_prepare(easy_config):
  # cpnmt(easy_config)
  pkl(easy_config)
  invert(easy_config)
  hdf5(easy_config)
  shuff(easy_config)

def nmt_train(easy_config):
  # command1 = "python " + easy_nmt + "GroundHog/experiments/nmt/train.py "\
    # + " --proto=" + "prototype_search_state "
  # print easy_nmt, "======"
  command1 = "python " + nmt_path + "train.py"\
    + " --proto=" + "prototype_search_state"\
    + " --state " + easy_nmt + "state.py"\
    + " >& " + easy_nmt + "out.txt &"
  write_step (command1)
  os.system (command1)

def nmt_test(easy_config):
  t_tokenisation(easy_config)
  t_truecasing(easy_config)
  command1 = "python " + nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_nmt + "search_state.pkl "\
    + " --source " + easy_evaluation + easy_config.testfilename + ".true." + easy_config.source_id\
    + " --trans " + easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id\
    + " " + easy_nmt + "search_model.npz"\
    + " >& " + easy_evaluation + "trans_out.txt &"
  write_step(command1)
  os.system(command1)

def nmt_dev(easy_config):
  command1 = "python " + nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_nmt + "search_state.pkl "\
    + " --source " + easy_tuning + easy_config.devfilename + ".true." + easy_config.source_id\
    + " --trans " + easy_tuning + easy_config.devfilename + ".translated." + easy_config.target_id\
    + " " + easy_nmt + "search_model.npz"\
    + " >& " + easy_tuning + "trans_out.txt &"
  write_step(command1)
  os.system(command1)

def nmt_dev_res(easy_config):
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_tuning + easy_config.devfilename + ".true." + easy_config.target_id 
    + " < " + easy_tuning + easy_config.devfilename + ".translated." + easy_config.target_id
    # + " < " + easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id + ".9"
    )
  write_step (command2)
  os.system (command2)
  translation_result = open (easy_tuning + "translation_result.txt", 'w')
  translated = open (easy_tuning + easy_config.devfilename + ".translated." + easy_config.target_id, 'r')
  source = open (easy_tuning + easy_config.devfilename + ".true." + easy_config.source_id, 'r')
  target = open (easy_tuning + easy_config.devfilename + ".true." + easy_config.target_id, 'r')
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

def nmt_check_overfitting_1(easy_config):
  command1 = "python " + nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_nmt + "search_state.pkl "\
    + " --source " + easy_overfitting + "OF.true." + easy_config.source_id\
    + " --trans " + easy_overfitting + "ontrain." + easy_config.target_id\
    + " " + easy_nmt + "search_model.npz"\
    + " >& " + easy_overfitting + "check_overfitting_out.txt &"
  write_step(command1)
  os.system(command1) 

def nmt_check_overfitting_2(easy_config):
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_overfitting + "OF.true." + easy_config.target_id 
    + " < " + easy_overfitting + "ontrain." + easy_config.target_id
    # + " < " + easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id + ".9"
    )
  write_step (command2)
  os.system (command2)

def nmt_make_backup(easy_config):
  dirname = easy_nmt + str(time.strftime('%Y_%m%d_%H%M',time.localtime(time.time())))
  command1 = "mkdir " + dirname
  if not os.path.exists(dirname):
    write_step(command1)
    os.system(command1)
  command2 = "cp " + easy_nmt + "*.* " + dirname
  write_step(command2)
  os.system(command2)


def bleu_score(easy_config):
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + easy_evaluation + easy_config.testfilename + ".true." + easy_config.target_id 
    + " < " + easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id
    # + " < " + easy_evaluation + easy_config.testfilename + ".translated." + easy_config.target_id + ".9"
    )
  write_step (command2)
  os.system (command2)
#########################  nmt ############################


def easymoses ():
  a = 0
  # corpus_preparation (easy_config)
  # language_model_training (easy_config)
  # training_translation_system (easy_config)
  # tuning (easy_config)
  # testing (easy_config)
  # cross_corpus("18", "nmt", "te", easy_config)
  # cross_corpus("17", "smt", "te", easy_config)
  # nplm (easy_config)
  # bnplm (easy_config)
  # nmt_prepare(easy_config)
  # nmt_train(easy_config)
  # overfitting_prepare(easy_config)
  # nmt_check_overfitting_1(easy_config)
  # nmt_check_overfitting_2(easy_config)
  # nmt_dev(easy_config)
  # nmt_dev_res(easy_config)
  # nmt_make_backup(easy_config)
  # nmt_test(easy_config)
  # bleu_score(easy_config)

if __name__ == "__main__" :
  print str (time.strftime('%Y-%m-%d %X',time.localtime(time.time())))
  if sys.argv[1] != exp_id:
    print "you input a wrong experiment id"
    exit()
  easymoses ()


######################### Training NPLM #############################
def prepare_corpus (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + easy_config.target_id 
    + " -threads " + easy_config.threads
    + " -no-escape 1 "
    + " < " + easy_config.training_corpus_path + easy_config.filename + "." + easy_config.target_id 
    + " > " + easy_nplm + easy_config.filename + ".tok." + easy_config.target_id)
  command2 = (easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_truecaser + "truecase-model." + easy_config.target_id 
    + " < " + easy_nplm + easy_config.filename + ".tok." + easy_config.target_id 
    + " > " + easy_nplm + easy_config.filename + ".true." + easy_config.target_id)
  write_step (command1)
  os.system (command1)
  write_step (command2)
  os.system (command2)

def prepare_neural_language_model (easy_config) :
  command1 = (easy_config.nplm_path + "bin/prepareNeuralLM " 
    + " --train_text " + easy_nplm + easy_config.filename  + ".true." + easy_config.target_id
    + " --ngram_size 5 " 
    + " --vocab_size 20000 "  
    + " --write_words_file " + easy_nplm + "words " 
    + " --train_file " + easy_nplm + "train.ngrams " 
    + " --validation_size 500 "
    + " --validation_file " + easy_nplm + "validation.ngrams " 
    + " >& " + easy_nplm + "prepareout.out &")
  write_step (command1)
  os.system (command1)

def train_neural_network (easy_config) :
  command1 = (easy_config.nplm_path + "bin/trainNeuralNetwork " 
    + " --train_file " + easy_nplm + "train.ngrams " 
    + " --validation_file " + easy_nplm + "validation.ngrams " 
    + " --num_epochs 30 "
    + " --input_words_file " + easy_nplm + "words " 
    + " --model_prefix " + easy_nplm + "model " 
    + " --input_embedding_dimension 150 "  
    + " --num_hidden 0" 
    + " --output_embedding_dimension 750 "
     + " --num_threads "+ easy_config.threads 
    + " >& " + easy_nplm + "nplmtrain.out &")
  write_step (command1)
  os.system (command1)

def nplm (easy_config) :
  prepare_corpus (easy_config)
  # prepare_neural_language_model (easy_config)
  # train_neural_network (easy_config)

def cross_corpus(id1, mt_type, tag, easy_config):
  command1 = "python " + nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_workspace + "nmt/" + id1 + "/" + "search_state.pkl "\
    + " --source " + easy_evaluation + easy_config.testfilename + ".true." + easy_config.source_id\
    + " --trans " + easy_evaluation + easy_config.testfilename + ".translated." + id1 + "." + easy_config.target_id\
    + " " + easy_workspace + "nmt/" + id1 + "/" + "search_model.npz"\
    + " >& " + easy_evaluation + id1 + "_out.txt &"
  command2 = "nohup nice " + easy_config.mosesdecoder_path + "bin/moses "\
    + " -threads " + easy_config.threads\
    + " -f " + easy_workspace + "tuning/" + id1 + "/"+  "moses.ini "\
    + " < " + easy_evaluation + easy_config.testfilename + ".true." + easy_config.source_id\
    + " > " + easy_evaluation + easy_config.testfilename + ".translated." + id1 + "." + easy_config.target_id\
    + " 2> " + easy_evaluation + id1 + "_out.txt"
  command3 = easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl "\
    + " -lc " + easy_evaluation + easy_config.testfilename + ".true." + easy_config.target_id\
    + " < " + easy_evaluation + easy_config.testfilename + ".translated." + id1 + "." + easy_config.target_id
  if tag == "tr" and mt_type == "nmt":
    write_step(command1)
    os.system(command1)
  elif tag == "tr" and mt_type == "smt":
    write_step(command2)
    os.system(command2)
  if tag == "te":
    write_step(command3)
    os.system(command3)