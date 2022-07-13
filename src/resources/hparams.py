batch_size=32
nb_labels=12
xlmr_embedding_dim = 768
weight_decay = 1e-5
lr = 2e-5
max_epoch = 20
max_sent_length = 64
dropout = 0.1
label2index =  {"<pad>": 1, "<s>": 0, "</s>": 2, "O": 3, "B-LOC": 4, "B-ORG": 5, "I-LOC": 6, "B-PER": 7, "I-PER":8,"I-ORG": 9, "B-MISC": 10, "I-MISC": 11}
index2label =  {"1":"<pad>","0":"<s>", "2":"</s>", "3":"O" , "4":"B-LOC", "5":"B-ORG", "6":"I-LOC", "7":"B-PER", "8":"I-PER","9":"I-ORG", "10":"B-MISC", "11":"I-MISC"}

pretrained_envibert= 'src/pretrained/envibert'
checkpoint_path = 'checkpoint/checkpoint_%EPOCH%.pt'
res_path = "results/acc_%EPOCH%.txt"

warm_up = "checkpoint/checkpoint_7.pt"

train_path = "src/resources/data/ner_train_phonlp.txt"
test_path = "src/resources/data/ner_test_phonlp.txt"
val_path = "src/resources/data/ner_valid_phonlp.txt"

PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2

PAD_TAG, PAD_TAG_ID = "<pad>", 1
BOS_TAG, BOS_TAG_ID = "<s>", 0
EOS_TAG, EOS_TAG_ID = "</s>", 2