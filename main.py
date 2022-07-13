from transformers import RobertaTokenizer, RobertaModel
from importlib.machinery import SourceFileLoader
from torch.utils.data import DataLoader
from src.dataset.nerdata import NerDataset
from src.utils import load_data
from src.resources import hparams
from src.model.vndgNER import vndgNER
from src.trainer import trainer
from torch import optim
import json
import torch
import os

cuda = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'
print(cuda)
pretrained_path = hparams.pretrained_envibert

tokenizer = SourceFileLoader("envibert.tokenizer", os.path.join(pretrained_path,'envibert_tokenizer.py')).load_module().RobertaTokenizer(pretrained_path)
roberta = RobertaModel.from_pretrained('nguyenvulebinh/envibert',cache_dir=pretrained_path)

train_path = hparams.train_path
test_path = hparams.test_path
val_path = hparams.val_path

print("-------- Loading training data --------")
train_sents, train_labels, max_sent_length = load_data(train_path)
print("number of sample: ", len(train_sents))
print("number of label: ", len(train_labels))
print(train_sents[0])
print("sent len : ",len(train_sents[0].split()))
print(train_labels[0])

print("-------- Loading val data --------")
val_sents, val_labels, max_sent_length = load_data(val_path)
print("number of sample: ", len(val_sents))
print("number of label: ", len(val_labels))
print("------- Load successful -------")
print(val_sents[0]) 
print(val_labels[0])

train_ner = NerDataset(sents = train_sents, labels=train_labels, tokenizer=tokenizer, max_sent_lenth=hparams.max_sent_length)
train_dl = DataLoader(train_ner, batch_size=hparams.batch_size)
val_ner = NerDataset(sents = val_sents, labels=val_labels, tokenizer=tokenizer, max_sent_lenth=hparams.max_sent_length)
val_dl = DataLoader(val_ner, batch_size=hparams.batch_size)


vndgNER = vndgNER(cuda=cuda,nb_label=hparams.nb_labels, roberta=roberta).to(cuda)  
optimizer = optim.Adam(vndgNER.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

trainer = trainer(model=vndgNER, train_dl=train_dl, val_dl=val_dl, max_epoch=hparams.max_epoch, optimizer=optimizer, cuda=cuda, warm_up=hparams.warm_up)
trainer.train()
