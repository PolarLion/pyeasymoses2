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


exp_group = "fan-tuning"
exp_id = "x"
# exp_id = "base"

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

def fan_tuning (easy_config):
  num = 100
  for i in range(2, num):
    # devfilename = utils.get_filename(exp_config["develop_corpus"]+str(i))
    # devfilename = utils.get_filename(exp_config["develop_corpus"])
    devfilename = "C_B.Dev"
    # if not os.path.exists(os.path.join(easy_config.easy_tuning, str(i))):
      # os.mkdir(os.path.join(easy_config.easy_tuning, str(i)))
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
  weight_dic = {}
  standard_line = "0.0105348\t0.0651135\t0.0532412\t0.00603957\t0.0532839\t0.0809408\t0.122954\t0.271147\t0.0633011\t0.0520243\t0.0513287\t0.0239177\t0.024159\t1\t-0.122014"
  paths = os.listdir(easy_config.easy_tuning)
  outfile = open(os.path.join(easy_config.easy_tuning, "cluster_weights.txt"),'w')
  outfile_re = open(os.path.join(easy_config.easy_tuning, "cluster_weights_reference.txt"),'w')
  # count_line = 0
  count0 = 0
  count = 0
  for path in paths:
    if os.path.isfile(os.path.join(easy_config.easy_tuning, path)):continue
    # if os.path.isfile(os.path.join(easy_config.easy_tuning, path+"/run20.moses.ini")):
      # print path
    
    # outfile.write(infile.readline())
    # outfile.write(infile.readline())
    dic = read_moses_ini(os.path.join(easy_config.easy_tuning, path))
    new_line = ""
    if dic :
      print path + '\t' + dic['LM0'] + '\t' + dic['TranslationModel00'] + '\t' + dic['bleu']
      for k in sorted(dic):
        if k == "bleu":continue
        # print k
        new_line += dic[k].strip() + '\t'
      # outfile.write(new_line+'\n')
      count += 1
      # break
      if not weight_dic.has_key(int(path)) :
        weight_dic[int(path)] = new_line
    else:
      count0 += 1
      new_line = standard_line
      # outfile.write(standard_line + '\n')
    
    infile = open(os.path.join(easy_config.easy_tuning, path+"/"+devfilename + ".true." + exp_config["source_id"]), 'r')
    for line in infile.readlines():
      outfile.write(line)
      outfile.write(new_line+'\n')
    infile.close()
    infile = open(os.path.join(easy_config.easy_tuning, path+"/"+devfilename + ".true." + exp_config["target_id"]), 'r')
    for line in infile.readlines():
      outfile_re.write(line)
    infile.close()
  outfile.close()  
  outfile_re.close()
  print count0, count, len(weight_dic), weight_dic.keys()
  return weight_dic
  
def fan_clustering(easy_config):
  # devfilename = utils.get_filename(exp_config["develop_corpus"])
  # print devfilename
  devfilename = "C_B.Dev"
  # exit()
  model_name = "cluster"
  embedding_file = open("/home/xwshi/easymoses_workspace2/fan-tuning/x/C_B.Dev.true.zh.encdoc_1000_embedding", 'r')
  embeddings = embedding_file.readlines()
  embedding_file.close()
  import re
  import numpy as np
  tokenized_sentences = [re.split(r'[\s\t]',sent.strip()) for sent in embeddings]
  X_train = np.asarray([[np.float32(w) for w in sent] for sent in tokenized_sentences])
  # print X_train
  # from sklearn.cluster import KMeans
  # model = KMeans(n_clusters=100)
  # model.fit(X_train)
  # print model.predict(X_train[11])
  import pickle
  # s = pickle.dumps(model)
  # open(model_name,'w').write(s)
  s = open(model_name,'r').read()
  model2 = pickle.loads(s)
  print model2.predict(X_train[11])
  source_lines = open(os.path.join(exp_config["develop_corpus"], devfilename +'.'+ exp_config["source_id"]), 'r').readlines()
  target_lines = open(os.path.join(exp_config["develop_corpus"], devfilename +'.'+ exp_config["target_id"]), 'r').readlines()
  line_dict = {}
  for i in range(0, 100):
    if not os.path.isdir(os.path.join(easy_config.easy_tuning, str(i))):
      os.mkdir(os.path.join(easy_config.easy_tuning, str(i)))
  for i in range(0, len(X_train)):
    k = model2.predict(X_train[i])
    if len(k) != 1:
      print "!!!!!!!!!!!!!!!!!!!!!!", len(k)
    if line_dict.has_key(k[0]):
      line_dict[k[0]].append(i)
    else:
      line_dict[k[0]] = []
      line_dict[k[0]].append(i)
  for k in line_dict.keys():
    s_outfile = open(os.path.join(easy_config.easy_tuning, str(k) + "/" + devfilename + "." + exp_config["source_id"]), 'w')
    t_outfile = open(os.path.join(easy_config.easy_tuning, str(k) + "/" + devfilename + "." + exp_config["target_id"]), 'w')
    for v in line_dict[k]:
      s_outfile.write(source_lines[v])
      t_outfile.write(target_lines[v])
    s_outfile.close()
    t_outfile.close()

def fan_clustering_test(easy_config):
  model_name = "cluster100"
  embedding_file = open("/home/xwshi/easymoses_workspace2/fan-tuning/x/C_B.Test.true.zh.encdoc_1000_embedding", 'r')
  # embedding_file = open("/home/xwshi/easymoses_workspace2/fan-tuning/x/C_B.Dev.true.zh.encdoc_1000_embedding", 'r')
  embeddings = embedding_file.readlines()
  embedding_file.close()
  import re
  import numpy as np
  tokenized_sentences = [re.split(r'[\s\t]',sent.strip()) for sent in embeddings]
  X_test = np.asarray([[np.float32(w) for w in sent] for sent in tokenized_sentences])
  import pickle
  # s = pickle.dumps(model)
  # open(model_name,'w').write(s)
  s = open(model_name,'r').read()
  model2 = pickle.loads(s)
  print model2.predict(X_test[1])
  weight_dic = fan_analyze(easy_config)
  # source_lines = open("/home/xwshi/easymoses_workspace2/fan-tuning/x/evalutaion/C_B.Test.true.zh", 'r').readlines()
  source_lines = open("/home/xwshi/easymoses_workspace2/fan-tuning/x/evalutaion/C_B.Dev.true.zh", 'r').readlines()
  count_score = 0
  score_threshold = 10
  standard_line = "0.0105348\t0.0651135\t0.0532412\t0.00603957\t0.0532839\t0.0809408\t0.122954\t0.271147\t0.0633011\t0.0520243\t0.0513287\t0.0239177\t0.024159\t1\t-0.122014"
  # outfile = open("/home/xwshi/easymoses_workspace2/fan-tuning/x/evalutaion/test_"+model_name+'_'+str(score_threshold), 'w')
  outfile = open("/home/xwshi/easymoses_workspace2/fan-tuning/x/evalutaion/dev_"+model_name+'_'+str(score_threshold), 'w')
  for i in range(0, len(source_lines)):
    sys.stdout.write("\rload %d"%(i+1))
    sys.stdout.flush()
    outfile.write(source_lines[i])
    cluster_id = model2.predict(X_test[i])[0]
    if model2.score(X_test[i]) > -score_threshold :
      print model2.score(X_test[i])
      if weight_dic.has_key(cluster_id):
        count_score += 1
        outfile.write(weight_dic[cluster_id].strip()+'\n')
      else:
        outfile.write(standard_line+'\n')
    else:
      outfile.write(standard_line+'\n')
    # print str(cluster_id)+'\t'+source_lines[i].strip()
  print "  score ", count_score
  outfile.close()

def fan_decoder(easy_config, filename):
  testfilename = utils.get_filename(exp_config["test_corpus"])
  # testfilename = "C_B.Dev"
  # testfilename = "cluster_weights.dev"
  print testfilename
  old_moses_ini = open(os.path.join(easy_config.easy_train,"model/moses.ini"),'r')
  new_moses_ini = open(os.path.join(easy_config.easy_evaluation, filename+"_moses.ini"), 'w')
  for line in old_moses_ini.readlines():
    if line.strip() != "[weight]":
      new_moses_ini.write(line.strip() + '\n')
    else:
      new_moses_ini.write("[alternate-weight-setting]\n")
      break
  old_moses_ini.close()
  infile = open(os.path.join(easy_config.easy_evaluation, filename), 'r')
  testfile = open(os.path.join(easy_config.easy_evaluation, filename+'.'+exp_config['source_id']), 'w')
  count = 0
  state = 0
  for line in infile.readlines():
    if state == 0:
      testfile.write("<seg weight-setting="+str(count)+">"+line.strip()+"</seg>\n")
      # outfile.close()
      state = 1
    elif state == 1:
      weight_dic = weights2weightsdic(line.strip())
      new_moses_ini.write(generate_weight_setting(count, weight_dic))
      count += 1
      state = 0
  infile.close()
  testfile.close()
  new_moses_ini.close()
  command1=easy_config.mosesdecoder_path+"bin/moses "\
    + " -threads 1"\
    + " -alternate-weight-setting"\
    + " -f " + os.path.join(easy_config.easy_evaluation, filename+"_moses.ini ")\
    + " -i " + os.path.join(easy_config.easy_evaluation, filename + "." + exp_config["source_id"])\
    + " > " + os.path.join(easy_config.easy_evaluation, filename + ".translated." + exp_config["target_id"])\
    + " 2> " + os.path.join(easy_config.easy_evaluation, filename + ".out") + " "
  command2 = easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl "\
    + " -lc " + os.path.join(easy_config.easy_evaluation, testfilename + ".true." + exp_config["target_id"])\
    + " < " + os.path.join(easy_config.easy_evaluation, filename + ".translated." + exp_config["target_id"])
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)
  
def train_encdec(easy_config):
  # training_filename = utils.get_filename(exp_config["training_corpus"])
  command1 = "python " + easy_config.nmt_path + "preprocess/preprocess.py "\
    + os.path.join(easy_config.easy_corpus, "Encdec.Train.zh")\
    + " -d " + os.path.join(easy_config.easy_corpus, "vocab." + exp_config["source_id"] + ".pkl")\
    + " -v " + exp_config["source_vocb"]\
    + " -b " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".pkl")\
    + " -p "
  # write_step (command1, easy_config)
  # os.system(command1)
  # print "-----------  invert ------------"
  command2 = "python " + easy_config.nmt_path + "preprocess/invert-dict.py " \
    + " " + os.path.join(easy_config.easy_corpus, "vocab." + exp_config["source_id"] + ".pkl")\
    + " " + os.path.join(easy_config.easy_corpus, "ivocab." + exp_config["source_id"] + ".pkl")
  # write_step (command2, easy_config)
  # os.system(command2)
  command3 = "python " + easy_config.nmt_path + "preprocess/convert-pkl2hdf5.py " \
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".pkl")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".h5")
  # write_step (command3, easy_config)
  # os.system(command3)
  command4 = "python " + easy_config.nmt_path + "preprocess/shuffle-hdf5.py " \
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".h5")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".h5")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".shuf.h5")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["target_id"] + ".shuf.h5")
  # write_step (command4, easy_config)
  # os.system(command4)
  hidden_dim = "500"
  command5 = "python " + easy_config.nmt_path + "train.py"\
    + " --proto=" + "prototype_encdec_state "\
    + " --state " + os.path.join(easy_config.easy_nplm, "state_"+hidden_dim+".py")\
    + " >& " + os.path.join(easy_config.easy_nplm, "out"+hidden_dim+".txt")+" &"
  write_step (command5, easy_config)
  os.system(command5)

def bleu_score(easy_config):
  testfilename = utils.get_filename(exp_config["test_corpus"])
  command2 = (easy_config.mosesdecoder_path + "scripts/generic/multi-bleu.perl " 
    + " -lc " + os.path.join(easy_config.easy_evaluation, testfilename + ".true." + exp_config["target_id"]) 
    + " < " + os.path.join(easy_config.easy_evaluation, testfilename + ".translated." + exp_config["target_id"])
    # + " < " + easy_config.easy_evaluation + testfilename + ".translated." + exp_config["target_id"] + ".9"
    )
  write_step (command2, easy_config)
  os.system(command2)




def easymoses ():
  a = 0
  if "true" != exp_config["config"]:
    print "please edit your config.py file"
    exit()
  # bnplm (easy_config)
  # add_bnplm_feature(easy_config)
  # smt_tuning (easy_config)
  fan_clustering(easy_config)
  # fan_tuning(easy_config)
  # fan_analyze(easy_config)
  # print fan_analyze(easy_config)
  # fan_clustering_test(easy_config)
  # fan_decoder(easy_config, "encdec_embedding_encdec100_25_test")
  # fan_decoder(easy_config, "test_cluster100_6")
  # fan_decoder(easy_config, "/home/xwshi/easymoses_workspace2/fan-tuning/x/evalutaion/C_B.Test.zh.new")
  # train_encdec(easy_config)




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


