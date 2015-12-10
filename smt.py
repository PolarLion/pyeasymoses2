import os
import easybleu
from utils import write_step
from config import info as exp_config

######################### corpus preparation  ###########################

def tokenisation (easy_config, training_filename) :
  command1 = easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + exp_config["source_id"]\
    + " -threads " + exp_config["threads"]\
    + " -no-escape 1 "\
    + " < " + exp_config["training_corpus"] + training_filename + "." + exp_config["source_id"] + " > "\
    + " " + os.path.join(easy_config.easy_corpus, training_filename + ".tok." + exp_config["source_id"])
  command2 = easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + exp_config["target_id"]\
    + " -threads " + exp_config["threads"]\
    + " -no-escape 1 "\
    + " < " + exp_config["training_corpus"] + training_filename + "." + exp_config["target_id"] + " > "\
    + " " + os.path.join(easy_config.easy_corpus, training_filename + ".tok." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)

def truecaser (easy_config, training_filename) :
  command1 = easy_config.mosesdecoder_path + "scripts/recaser/train-truecaser.perl --model "\
    + " " + os.path.join(easy_config.easy_truecaser, "truecase-model." + exp_config["source_id"]) + " --corpus "\
    + " " + os.path.join(easy_config.easy_corpus, training_filename + ".tok." + exp_config["source_id"])
  command2 = easy_config.mosesdecoder_path + "scripts/recaser/train-truecaser.perl --model "\
    + " " + os.path.join(easy_config.easy_truecaser, "truecase-model." + exp_config["target_id"]) + " --corpus "\
    + " " + os.path.join(easy_config.easy_corpus, training_filename + ".tok." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)    
  write_step (command2, easy_config)
  os.system(command2)    

def truecasing (easy_config, training_filename) :
  command1 = easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model "\
    + " " + os.path.join(easy_config.easy_truecaser, "truecase-model." + exp_config["source_id"])\
    + " < " + os.path.join(easy_config.easy_corpus, training_filename + ".tok." + exp_config["source_id"] )\
    + " > " + os.path.join(easy_config.easy_corpus, training_filename + ".true." + exp_config["source_id"])
  command2 = easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model"\
    + " " + os.path.join(easy_config.easy_truecaser, "truecase-model." + exp_config["target_id"])\
    + " < " + os.path.join(easy_config.easy_corpus, training_filename + ".tok." + exp_config["target_id"] )\
    + " > " + os.path.join(easy_config.easy_corpus, training_filename + ".true." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)

def limiting_sentence_length (easy_config, training_filename) :
  command1 = easy_config.mosesdecoder_path + "scripts/training/clean-corpus-n.perl "\
    + " " + os.path.join(easy_config.easy_corpus, training_filename + ".true " + exp_config["source_id"] + " " + exp_config["target_id"])\
    + " " + os.path.join(easy_config.easy_corpus, training_filename +".clean  1 "\
    + exp_config["sentence_length"])
  write_step (command1, easy_config)
  os.system(command1)
######################### corpus preparation  ###########################


#########################  language model traning #######################
def generate_sb (easy_config, training_filename) :
  command1 = easy_config.irstlm_path + "bin/add-start-end.sh < "\
    + " " + os.path.join(easy_config.easy_corpus, training_filename + ".true." + exp_config["target_id"])\
    + " > " + os.path.join(easy_config.easy_lm, training_filename + ".sb." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)

def generate_lm (easy_config, training_filename) :
  command1 = ("export IRSTLM=" + easy_config.irstlm_path + "; " + easy_config.irstlm_path + "bin/build-lm.sh"\
    + " -n 5"\
    + " -i " + os.path.join(easy_config.easy_lm, training_filename + ".sb." + exp_config["target_id"])\
    + " -t ./tmp -p -s improved-kneser-ney -o " + os.path.join(easy_config.easy_lm, training_filename + ".lm." + exp_config["target_id"]))
  write_step (command1, easy_config)
  os.system(command1)

def generate_arpa (easy_config, training_filename) :
  command1 = easy_config.irstlm_path + "bin/compile-lm --text=yes "\
    + " " + os.path.join(easy_config.easy_lm, training_filename + ".lm." + exp_config["target_id"] + ".gz")\
    + " " + os.path.join(easy_config.easy_lm, training_filename + ".arpa." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)

def generate_blm (easy_config, training_filename) :
  command1 = easy_config.mosesdecoder_path + "bin/build_binary -i "\
    + " " + os.path.join(easy_config.easy_lm, training_filename + ".arpa." + exp_config["target_id"])\
    + " " + os.path.join(easy_config.easy_lm, training_filename + ".blm." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
#########################  language model traning #######################

#########################  training ranslation system 

def translation_model (easy_config, training_filename) :
  command1 = "nohup nice " + easy_config.mosesdecoder_path + "scripts/training/train-model.perl "\
    + " -mgiza -mgiza-cpus 16 -cores 2 "\
    + " -root-dir " + easy_config.easy_train\
    + " -corpus " + " " + os.path.join(easy_config.easy_corpus, training_filename + ".clean")\
    + " -f " + exp_config["source_id"] + " -e " + exp_config["target_id"]\
    + " -alignment grow-diag-final-and "\
    + " " + exp_config["phrase"]\
    + " msd-bidirectional-fe -lm 0:"+exp_config["n-gram"]+":"\
    + os.path.join(easy_config.easy_lm, training_filename + ".blm." + exp_config["target_id"]) + ":8 "\
    + " -external-bin-dir " + easy_config.giza_path + "bin"\
    + " >& " + os.path.join(easy_config.easy_train, "training.out") + " &"
  write_step (command1, easy_config)
  os.system(command1)

###########################################

def tuning_tokenizer (easy_config, devfilename) :
  command1 = easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + exp_config["source_id"]\
    + " -threads " + exp_config["threads"]\
    + " -no-escape 1 "\
    + " < " + exp_config["develop_corpus"] + devfilename + "." + exp_config["source_id"]\
    + " > " + os.path.join(easy_config.easy_tuning, devfilename + ".tok." + exp_config["source_id"])
  command2 = easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + exp_config["target_id"]\
    + " -threads " + exp_config["threads"]\
    + " -no-escape 1 "\
    + " < " + exp_config["develop_corpus"] + devfilename + "." + exp_config["target_id"]\
    + " > " + os.path.join(easy_config.easy_tuning, devfilename + ".tok." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)

def tuning_truecase (easy_config, devfilename) :
  command1 = easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model"\
    + " " + os.path.join(easy_config.easy_truecaser, "truecase-model." + exp_config["source_id"])\
    + " < " + os.path.join(easy_config.easy_tuning, devfilename + ".tok." + exp_config["source_id"])\
    + " > " + os.path.join(easy_config.easy_tuning, devfilename + ".true." + exp_config["source_id"])
  command2 = easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model"\
    + " " + os.path.join(easy_config.easy_truecaser, "truecase-model." + exp_config["target_id"])\
    + " < " + os.path.join(easy_config.easy_tuning, devfilename + ".tok." + exp_config["target_id"])\
    + " > " + os.path.join(easy_config.easy_tuning, devfilename + ".true." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)

def tuning_process (easy_config, devfilename) :
  command1 = "nohup nice " + easy_config.mosesdecoder_path + "scripts/training/mert-moses.pl "\
    + "--decoder-flags=\"-threads "+exp_config["threads"]+"\""\
    + " -threads " + exp_config["threads"]\
    + " -maximum-iterations " + exp_config["tuning_max_iterations"]\
    + " -working-dir " + easy_config.easy_tuning\
    + " " + os.path.join(easy_config.easy_tuning, devfilename + ".true." + exp_config["source_id"])\
    + " " + os.path.join(easy_config.easy_tuning, devfilename + ".true." + exp_config["target_id"])\
    + " " + easy_config.mosesdecoder_path + "bin/moses_chart " + os.path.join(easy_config.easy_train,"model/moses.ini ")\
    + " --mertdir " + easy_config.mosesdecoder_path + "bin/ &> " + os.path.join(easy_config.easy_tuning, "mert.out") + " &"
  write_step (command1, easy_config)
  os.system(command1)

def read_moses_ini(inpath):
  import re
  files = os.listdir(inpath)
  ini_dict = {}
  if not os.path.isfile(os.path.join(inpath, 'moses.ini')):
    return ini_dict
  ini_file = open(os.path.join(inpath, 'moses.ini'), 'r')
  for line in ini_file.readlines():
    mt = re.match(r'# BLEU (\d\.?\d*)(.*)', line)
    if mt :
      # #print mt.groups()[0]
      if mt.groups()[0] == "0":
        # print inpath
        return ini_dict
      ini_dict['bleu'] = mt.groups()[0]
      
    lr0 = re.match(r'LexicalReordering0= (.*)', line)
    if lr0:
      #print lr0.groups()[0]
      count = 0
      for w in lr0.groups()[0].split(' '):
        ini_dict['LexicalReordering0'+str(count)] = w
        count += 1
      
    d0 = re.match(r'Distortion0= (.+)', line)
    if d0:
      #print d0.groups()[0]
      ini_dict['Distortion0']=d0.groups()[0]
      
    LM0 = re.match(r'LM0= (.+)', line)
    if LM0:
      # print LM0.groups()[0]
      ini_dict['LM0']=LM0.groups()[0]
      
    WordPenalty0 = re.match(r'WordPenalty0= (.+)', line)
    if WordPenalty0:
      #print WordPenalty0.groups()[0]
      ini_dict['WordPenalty0']=WordPenalty0.groups()[0]
      
    PhrasePenalty0 = re.match(r'PhrasePenalty0= (.+)', line)
    if PhrasePenalty0:
      #print PhrasePenalty0.groups()[0]
      ini_dict['PhrasePenalty0']=PhrasePenalty0.groups()[0]
      
    TranslationModel0 = re.match(r'TranslationModel0= (.*)', line)
    if TranslationModel0:
      #print TranslationModel0.groups()[0]
      count = 0
      for w in TranslationModel0.groups()[0].split(' '):
        ini_dict['TranslationModel0'+str(count)] = w
        count += 1
      
    UnknownWordPenalty0 = re.match(r'UnknownWordPenalty0= (.+)', line)
    if UnknownWordPenalty0:
      #print UnknownWordPenalty0.groups()[0]
      ini_dict['UnknownWordPenalty0']=UnknownWordPenalty0.groups()[0]
      
  return ini_dict
    # else:
      # print line
  # exit()

def weights2weightsdic(weights):
  weights_dict = {}
  # print weights
  Distortion0, LM0, LexicalReordering00, LexicalReordering01, LexicalReordering02, LexicalReordering03, LexicalReordering04, LexicalReordering05, PhrasePenalty0, TranslationModel00, TranslationModel01, TranslationModel02, TranslationModel03, UnknownWordPenalty0, WordPenalty0 = weights.strip().split('\t')
  weights_dict["Distortion0"]=str(Distortion0)
  weights_dict["LM0"]=LM0
  weights_dict["LexicalReordering0"] = str(LexicalReordering00) + ' ' + str(LexicalReordering01) + ' ' + str(LexicalReordering02) + ' ' + str(LexicalReordering03) + ' ' + str(LexicalReordering04) + ' ' + str(LexicalReordering05)
  weights_dict["PhrasePenalty0"]=str(PhrasePenalty0)
  weights_dict["TranslationModel0"]=str(TranslationModel00)+' '+str(TranslationModel01)+' '+str(TranslationModel02)+' '+str(TranslationModel03)
  weights_dict["UnknownWordPenalty0"]=str(UnknownWordPenalty0)
  weights_dict["WordPenalty0"]=str(WordPenalty0)
  return weights_dict

def generate_weight_setting(w_id, weights_dict):
  weights = ""
  weights += "id="+str(w_id) + '\n'
  weights += "UnknownWordPenalty0= " + weights_dict["UnknownWordPenalty0"] + '\n'
  weights += "WordPenalty0= " + weights_dict["WordPenalty0"] + '\n'
  weights += "PhrasePenalty0= " + weights_dict["PhrasePenalty0"] + '\n'
  weights += "TranslationModel0= " + weights_dict["TranslationModel0"] + '\n'
  weights += "LexicalReordering0= " + weights_dict["LexicalReordering0"] + '\n'
  weights += "Distortion0= " + weights_dict["Distortion0"] + '\n'
  weights += "LM0= " + weights_dict["LM0"] + '\n'
  return weights
      
#########################  training translation system 

####################### testing #############################################

def t_tokenisation (easy_config, testfilename) :
  command1 = easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + exp_config["source_id"]\
    + " -threads " + exp_config["threads"]\
    + " -no-escape 1 "\
    + " < " + exp_config["test_corpus"] + testfilename + "." + exp_config["source_id"]\
    + " > " + os.path.join(easy_config.easy_evaluation, testfilename + ".tok." + exp_config["source_id"])
  command2 = easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + exp_config["target_id"]\
    + " -threads " + exp_config["threads"]\
    + " -no-escape 1 "\
    + " < " + exp_config["test_corpus"] + testfilename + "." + exp_config["target_id"]\
    + " > " + os.path.join(easy_config.easy_evaluation, testfilename + ".tok." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)
  
def t_truecasing (easy_config, testfilename) :
  command1 = easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model"\
     + " " + os.path.join(easy_config.easy_truecaser, "truecase-model." + exp_config["source_id"])\
    + " < " + os.path.join(easy_config.easy_evaluation, testfilename  + ".tok." + exp_config["source_id"])\
    + " > " + os.path.join(easy_config.easy_evaluation, testfilename  + ".true." + exp_config["source_id"])\
    # + " > " + easy_config.easy_evaluation + testfilename  + ".translated.true." + exp_config["target_id"])
  command2 = easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model"\
     + " " + os.path.join(easy_config.easy_truecaser, "truecase-model." + exp_config["target_id"])\
    + " < " + os.path.join(easy_config.easy_evaluation, testfilename  + ".tok." + exp_config["target_id"])\
    + " > " + os.path.join(easy_config.easy_evaluation, testfilename  + ".true." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)
 
def t_filter_model_given_input (easy_config, testfilename) :
  command1 = (easy_config.mosesdecoder_path + "scripts/training/filter-model-given-input.pl " 
    + " " + easy_config.easy_evaluation + "filtered-" + testfilename 
    + " " + easy_config.working_path + "moses.ini " 
    + " " + test_corpus_path + test_filename + ".true." + exp_config["source_id"] 
    + " -Binarizer " + easy_config.mosesdecoder_path + "bin/processPhraseTableMin")
  write_step (command1, easy_config)
  os.system(command1)

def run_test (easy_config, testfilename) :
  command1 = "nohup nice " + easy_config.mosesdecoder_path + "bin/moses "\
    + " -threads 1"\
    + " -f " + os.path.join(easy_config.easy_tuning, "moses.ini ")\
    + " < " + os.path.join(easy_config.easy_evaluation, testfilename + ".true." + exp_config["source_id"])\
    + " > " + os.path.join(easy_config.easy_evaluation, testfilename + ".translated." + exp_config["target_id"])\
    + " 2> " + os.path.join(easy_config.easy_evaluation, testfilename + ".out") + " "
  command2 = easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl "\
    + " -lc " + os.path.join(easy_config.easy_evaluation, testfilename + ".true." + exp_config["target_id"])\
    + " < " + os.path.join(easy_config.easy_evaluation, testfilename + ".translated." + exp_config["target_id"])
    # + " < " + easy_config.easy_evaluation + testfilename + ".translated." + exp_config["target_id"] + ".9"
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)

def view_result (easy_config, testfilename) :
  translation_result = open (os.path.join(easy_config.easy_evaluation, "translation_result.txt"), 'w')
  translated = open (os.path.join(easy_config.easy_evaluation, testfilename + ".translated." + exp_config["target_id"]), 'r')
  # translated = open (easy_config.easy_evaluation + "CHT.Test.translated.en.918", 'r')
  source = open (os.path.join(easy_config.easy_evaluation, testfilename + ".true." + exp_config["source_id"]), 'r')
  target = open (os.path.join(easy_config.easy_evaluation, testfilename + ".true." + exp_config["target_id"]), 'r')
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

def test_on_train (easy_config) :
  command1 = "nohup nice " + easy_config.mosesdecoder_path + "bin/moses "\
    + " -threads " + exp_config["threads"]\
    + " -f " + os.path.join(easy_config.easy_tuning, "moses.ini ")\
    + " < " + os.path.join(easy_config.easy_overfitting, "OF.clean." + exp_config["source_id"])\
    + " > " + os.path.join(easy_config.easy_overfitting, "OF." + ".translated." + exp_config["target_id"])\
    + " 2> " + os.path.join(easy_config.easy_overfitting, "OF.clean." + ".out") + " "
  command2 = easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl "\
    + " -lc " + os.path.join(easy_config.easy_overfitting, "OF.clean." + exp_config["target_id"])\
    + " < " + os.path.join(easy_config.easy_overfitting, "OF." + ".translated." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)

######################### Training NPLM #############################
def prepare_corpus (easy_config) :
  command1 = (easy_config.mosesdecoder_path + "scripts/tokenizer/tokenizer.perl -l " + exp_config["target_id"] 
    + " -threads " + exp_config["threads"]
    + " -no-escape 1 "
    + " < " + exp_config["training_corpus"] + training_filename + "." + exp_config["target_id"]
    + " > " + easy_config.easy_nplm + training_filename + ".tok." + exp_config["target_id"])
  command2 = (easy_config.mosesdecoder_path + "scripts/recaser/truecase.perl --model " 
    + " " + easy_config.easy_truecaser + "truecase-model." + exp_config["target_id"] 
    + " < " + easy_config.easy_nplm + training_filename + ".tok." + exp_config["target_id"] 
    + " > " + easy_config.easy_nplm + training_filename + ".true." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)

def prepare_neural_language_model (easy_config) :
  command1 = (easy_config.nplm_path + "bin/prepareNeuralLM " 
    + " --train_text " + easy_config.easy_nplm + training_filename  + ".true." + exp_config["target_id"]
    + " --ngram_size 5 " 
    + " --vocab_size 20000 "  
    + " --write_words_file " + easy_config.easy_nplm + "words " 
    + " --train_file " + easy_config.easy_nplm + "train.ngrams " 
    + " --validation_size 500 "
    + " --validation_file " + easy_config.easy_nplm + "validation.ngrams " 
    + " >& " + easy_config.easy_nplm + "prepareout.out &")
  write_step (command1, easy_config)
  os.system(command1)

def train_neural_network (easy_config) :
  command1 = (easy_config.nplm_path + "bin/trainNeuralNetwork " 
    + " --train_file " + easy_config.easy_nplm + "train.ngrams " 
    + " --validation_file " + easy_config.easy_nplm + "validation.ngrams " 
    + " --num_epochs 30 "
    + " --input_words_file " + easy_config.easy_nplm + "words " 
    + " --model_prefix " + easy_config.easy_nplm + "model " 
    + " --input_embedding_dimension 150 "  
    + " --num_hidden 0" 
    + " --output_embedding_dimension 750 "
     + " --num_threads "+ exp_config["threads"] 
    + " >& " + easy_config.easy_nplm + "nplmtrain.out &")
  write_step (command1, easy_config)
  os.system(command1)

def nplm (easy_config) :
  prepare_corpus (easy_config)
  # prepare_neural_language_model (easy_config)
  # train_neural_network (easy_config)

def cross_corpus(id1, mt_type, tag, easy_config):
  command1 = "python " + easy_config.nmt_path + "sample.py"\
    + " --beam-search "\
    + " --beam-size 12"\
    + " --state " + easy_config.easy_workspace + "nmt/" + id1 + "/" + "search_state.pkl "\
    + " --source " + easy_config.easy_evaluation + testfilename + ".true." + exp_config["source_id"]\
    + " --trans " + easy_config.easy_evaluation + testfilename + ".translated." + id1 + "." + exp_config["target_id"]\
    + " " + easy_config.easy_workspace + "nmt/" + id1 + "/" + "search_model.npz"\
    + " >& " + easy_config.easy_evaluation + id1 + "_out.txt &"
  command2 = "nohup nice " + easy_config.mosesdecoder_path + "bin/moses "\
    + " -threads " + exp_config["threads"]\
    + " -f " + easy_config.easy_workspace + "tuning/" + id1 + "/"+  "moses.ini "\
    + " < " + easy_config.easy_evaluation + testfilename + ".true." + exp_config["source_id"]\
    + " > " + easy_config.easy_evaluation + testfilename + ".translated." + id1 + "." + exp_config["target_id"]\
    + " 2> " + easy_config.easy_evaluation + id1 + "_out.txt"
  command3 = easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl "\
    + " -lc " + easy_config.easy_evaluation + testfilename + ".true." + exp_config["target_id"]\
    + " < " + easy_config.easy_evaluation + testfilename + ".translated." + id1 + "." + exp_config["target_id"]
  if tag == "tr" and mt_type == "nmt":
    write_step(command1, easy_config)
    os.system(command1)
  elif tag == "tr" and mt_type == "smt":
    write_step(command2, easy_config)
    os.system(command2)
  if Tag == "te":
    write_step(comMand3)
    os.system(command3)



######################## bnplm ######################################

def extract_training (easy_config, training_filename) :
  command1 = easy_config.mosesdecoder_path + "scripts/training/bilingual-lm/extract_training.py"\
    + " --working-dir " + easy_config.easy_bnplm\
    + " --corpus " + os.path.join(easy_config.easy_corpus, training_filename + ".clean")\
    + " --source-language " + exp_config["source_id"]\
    + " --target-language " + exp_config["target_id"]\
    + " --align " + os.path.join(easy_config.easy_train, "model/aligned.grow-diag-final-and")\
    + " --prune-target-vocab " + exp_config["target_vocb"]\
    + " --prune-source-vocab " + exp_config["source_vocb"]\
    + " --target-context " + exp_config["bnplm_target_context"]\
    + " --source-context " + exp_config["bnplm_source_context"]
  write_step (command1, easy_config)
  os.system(command1)

def train_nplm (easy_config, training_filename) : 
  command1 = easy_config.mosesdecoder_path + "scripts/training/bilingual-lm/train_nplm.py"\
    + " --working-dir " + easy_config.easy_bnplm\
    + " --corpus " + os.path.join(easy_config.easy_corpus, training_filename + ".clean ")\
    + " --nplm-home " + easy_config.nplm_path\
    + " --ngram-size " + exp_config["bnplm_ngram"]\
    + " --epochs " + exp_config["bnplm_epochs"]\
    + " --learning-rate " + exp_config["bnplm_learning_rate"]\
    + " --hidden "+ exp_config["bnplm_hidden"]\
    + " --input-embedding "+ exp_config["bnplm_input_embedding"]\
    + " --output-embedding " + exp_config["bnplm_hidden"]\
    + " --threads " + exp_config["threads"]\
    + " &> " + os.path.join(easy_config.easy_bnplm, "nplm.out") + " &"
  write_step (command1, easy_config)
  os.system(command1)