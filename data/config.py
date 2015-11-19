
info ={
    "config":"false",
    "threads":"16",
    "sentence_length":"100",
    "training_corpus":"",
    "develop_corpus":"",
    "test_corpus":"",
    "source_id":"zh",
    "target_id":"en",
    "source_vocb":"10000",
    "target_vocb":"30000",
    #moses
    "n-gram":"5",
    "tuning_max_iterations":"20",
    #"hierarchical":"--hierarchical --glue-grammar",
    "phrase":"-reordering",
    #end moses
    #bnplm
    "bnplm_target_context":"5",
    "bnplm_source_context":"4",
    "bnplm_epochs":"40",
    "bnplm_ngram":"14",
    "bnplm_learning_rate":"0.7",
    "bnplm_input_embedding":"150",
    "bnplm_output_embedding":"150",
    "bnplm_hidden":"512",
    #end bnplm
    #nmt
    "nmt_loopiters":"100000",
    "nmt_hidden_dim":"1000",
    #emd nmt
}
