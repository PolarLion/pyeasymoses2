import os
from utils import write_step
from config import info as exp_config

def pkl (easy_config, training_filename):
  command1 = "python " + easy_config.nmt_path + "preprocess/preprocess.py "\
    + os.path.join(easy_config.easy_corpus, training_filename  + ".clean." + exp_config["source_id"])\
    + " -d " + os.path.join(easy_config.easy_corpus, "vocab." + exp_config["source_id"] + ".pkl")\
    + " -v " + exp_config["source_vocb"]\
    + " -b " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".pkl")\
    + " -p " #+ os.path.join(easy_config.easy_corpus, "*"+exp_config["source_id"]+".txt.gz"
  command2 = "python " + easy_config.nmt_path + "preprocess/preprocess.py " \
    + os.path.join(easy_config.easy_corpus, training_filename  + ".clean." + exp_config["target_id"])\
    + " -d " + os.path.join(easy_config.easy_corpus, "vocab." + exp_config["target_id"] + ".pkl")\
    + " -v " + exp_config["target_vocb"]\
    + " -b " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["target_id"] + ".pkl")\
    + " -p " #+ os.path.join(easy_config.easy_corpus, "*"+exp_config["source_id"]+".txt.gz"
  write_step (command1, easy_config)
  os.system (command1)
  write_step (command2, easy_config)
  os.system (command2)

def invert(easy_config, training_filename):
  print "-----------  invert ------------"
  command1 = "python " + easy_config.nmt_path + "preprocess/invert-dict.py " \
    + " " + os.path.join(easy_config.easy_corpus, "vocab." + exp_config["source_id"] + ".pkl")\
    + " " + os.path.join(easy_config.easy_corpus, "ivocab." + exp_config["source_id"] + ".pkl")
  command2 = "python " + easy_config.nmt_path + "preprocess/invert-dict.py " \
    + " " + os.path.join(easy_config.easy_corpus, "vocab." + exp_config["target_id"] + ".pkl")\
    + " " + os.path.join(easy_config.easy_corpus, "ivocab." + exp_config["target_id"] + ".pkl")
  write_step (command1, easy_config)
  os.system (command1)
  write_step (command2, easy_config)
  os.system (command2)

def hdf5(easy_config, training_filename):
  command1 = "python " + easy_config.nmt_path + "preprocess/convert-pkl2hdf5.py " \
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".pkl")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".h5")
  command2 = "python " + easy_config.nmt_path + "preprocess/convert-pkl2hdf5.py " \
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["target_id"] + ".pkl")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["target_id"] + ".h5")
  write_step (command1, easy_config)
  os.system(command1)
  write_step (command2, easy_config)
  os.system(command2)

def shuff(easy_config, training_filename):
  command1 = "python " + easy_config.nmt_path + "preprocess/shuffle-hdf5.py " \
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".h5")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["target_id"] + ".h5")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["source_id"] + ".shuf.h5")\
    + " " + os.path.join(easy_config.easy_corpus, "binarized_text." + exp_config["target_id"] + ".shuf.h5")
  write_step (command1, easy_config)
  os.system(command1)

def create_statefile(easy_config):
  text = "dict("\
    + "\nnull_sym_source = " + exp_config["source_vocb"]\
    + ",\n null_sym_target = " + exp_config["target_vocb"]\
    + ",\n n_sym_source = " + str(int(exp_config["source_vocb"])+1)\
    + ",\n n_sym_target = " + str(int(exp_config["target_vocb"])+1)\
    + ",\n loopIters = " + exp_config["nmt_loopiters"]\
    + ",\n source = [\"" + os.path.join(easy_config.easy_corpus,"binarized_text."+exp_config["source_id"]+".shuf.h5") + "\"]"\
    + ",\n indx_word = \"" + os.path.join(easy_config.easy_corpus, "ivocab."+exp_config["target_id"]+".pkl") +"\""\
    + ",\n word_indx = \"" + os.path.join(easy_config.easy_corpus,"vocab."+exp_config["target_id"]+".pkl") +"\""\
    + ",\n target = [\"" + os.path.join(easy_config.easy_corpus,"binarized_text."+exp_config["source_id"]+".shuf.h5") +"\"]"\
    + ",\n indx_word_target = \"" + os.path.join(easy_config.easy_corpus, "ivocab."+exp_config["source_id"]+".pkl")+"\""\
    + ",\n word_indx_trgt = \"" + os.path.join(easy_config.easy_corpus, "vocab."+exp_config["source_id"]+".pkl")+"\""\
    + ",\n prefix = \'" + easy_config.easy_train + "/search_\'"\
    + "\n)"
  if not os.path.isfile(os.path.join(easy_config.easy_train, "state.py")):
    outfile =open(os.path.join(easy_config.easy_train, "state.py"), 'w')
    outfile.write(text+'\n')
    outfile.close()


def overfitting_prepare(easy_config, training_filename, sampling_base = 30):
  easycorpus.sampling_file(easy_config.easy_corpus+training_filename+".true."+exp_config["source_id"], 
    easy_config.easy_overfitting+"OF.true."+exp_config["source_id"], sampling_base)
  easycorpus.sampling_file(easy_config.easy_corpus+training_filename+".true."+exp_config["target_id"], 
    easy_config.easy_overfitting+"OF.true."+exp_config["target_id"], sampling_base)
  write_step("overfitting_prepare")

