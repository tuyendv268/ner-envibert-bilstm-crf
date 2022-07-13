import json
from tqdm import tqdm
from src.resources import hparams
import os

def join(sp_token, sp_tag):
        join_token, join_tag = [], []
        tmp_token, tmp_tag = [], []
        for token, tag in zip(sp_token, sp_tag):
            token = str(token)
            if token.startswith('▁') or token == '<s>' or token == '</s>':
                if len(tmp_token) > 0:
                    join_token.append(''.join(tmp_token).replace('▁', ''))
                    join_tag.append(tmp_tag[0])
                tmp_token = [token]
                tmp_tag = [tag]
            else:
                tmp_token.append(token)
                tmp_tag.append(tag)

        if len(tmp_token) > 0:
            join_token.append(''.join(tmp_token).replace('▁', ''))
            join_tag.append(tmp_tag[0])

        return join_token[1:-1], join_tag[1:-1]
def joins(sp_tokens, sp_tags):
        join_tokens, join_tags = [], []
        for sp_token, sp_tag in zip(sp_tokens, sp_tags):
            join_token, join_tag = join(sp_token, sp_tag)
            join_tokens.append(join_token)
            join_tags.append(join_tag)

        return join_tokens, join_tags


def extend_label(tokens, labels):
  idx = 0
  # print(len(tokens))
  # print(len(labels))
  new_labels, label_mask = [], []
  for i in range(len(tokens)):
    if tokens[i].startswith("▁") or tokens[i] == "<s>" or tokens[i] == "</s>":
      new_labels.append(labels[idx])
      prev = labels[idx]
      idx += 1
      label_mask.append(1)
    else:
      if prev.startswith("B"):
        new_labels.append(prev.replace("B-","I-"))
        prev = prev.replace("B-","I-")
      else:
        new_labels.append(prev)
      label_mask.append(1)
  return new_labels, label_mask

def cvt_label2ids(label):
  label2index = hparams.label2index
  labels = [label2index[ele] for ele in label]
  return labels
def convert_ids2label(ids):
  ids2label = hparams.index2label
  labels = [ids2label[str(ele)] for ele in ids]
  return labels

def get_input_for_ner(tokenizer, sent):
  tokens = tokenizer.tokenize(sent)

  tokens = ["<s>"] + tokens + ["</s>"]
  input_masks = [1]*len(tokens)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  return input_ids, input_masks

def get_data_for_ner(tokenizer, sent, label, max_sent_length):
  tokens = tokenizer.tokenize(sent)

  labels, label_masks = extend_label(tokens, label)

  if len(labels) > max_sent_length:
    labels = labels[0: max_sent_length]
    tokens = tokens[0: max_sent_length]
    label_masks = label_masks[0:max_sent_length]
  # print(labels)
  max_sent_length += 2
  labels = ["<s>"] + labels + ["</s>"]
  tokens = ["<s>"] + tokens + ["</s>"]
  label_masks = [1] + label_masks + [1]
  # print(labels)

  input_masks = [1]*len(tokens) + [0]*(max_sent_length-len(tokens))
  tokens = tokens + ["<pad>"]*(max_sent_length-len(tokens))
  # print(tokens)
  labels = labels + ["<pad>"]*(max_sent_length-len(labels))
  label_masks = label_masks + [0]*(max_sent_length-len(label_masks))
  # print("input_mask : ", len(input_masks))
  # print("labels : ", len(labels))
  # print("tokens : ", len(tokens))

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  labels = cvt_label2ids(labels)

  return input_ids, input_masks, labels, label_masks

def expand_label_when_loaddata(tokens, labels):
  new_labels = []
  # print(tokens)
  for i, token in enumerate(tokens):
    if (i == 0):
      new_labels.append(labels)
    else:
      new_labels.append(labels.replace("B-","I-"))
  return new_labels

def load_data(path):
  print("loading data: ", path)
  datas,labels = [], []
  for file in tqdm(os.listdir(path)):
    current_path = os.path.join(path, file)
    f = open(current_path, "r",encoding="utf-8")
    lines = f.readlines()
    count = 0
    datas_temp, labels_temp = "", []
    for line in lines:
      temp = line.replace("\n","").strip().split("\t")
      if line.split("\t")[0] in '''! " # $ % & \ ' ( ) * + , - . / : ; < = > ? @ [ \\ ] ^ _ ` { | } ~ '''.split():
          continue
      if(line == "\n"):
          continue
      token_tmp = temp[0].replace("_"," ").lower()
      if(count + len(token_tmp.split()) >= hparams.max_sent_length):
          datas.append(datas_temp)
          labels.append(labels_temp)
          datas_temp, labels_temp = "", []
          count = 0
      token_tmp = temp[0].replace("_"," ").lower()
      datas_temp += token_tmp + " "

      if(len(token_tmp.split()) > 1):
        new_labels = expand_label_when_loaddata(token_tmp.split(), temp[1])
        labels_temp += new_labels
        count += len(token_tmp.split())
      else:
        labels_temp.append(temp[1])
        count += 1
    datas.append(datas_temp)
    labels.append(labels_temp)
    datas_temp, labels_temp = "", []
     # print("Load successful...............")
  return datas, labels, hparams.max_sent_length

def load_sent_data(path):
  datas,labels = [], []
  max_sent_length = -1
  for file in os.listdir(path):
    current_path = os.path.join(path, file)
    # print("Loading data..................")
    f = open(current_path, "r",encoding="utf-8")
    lines = f.readlines()
    
    datas_temp, labels_temp = "", []
    for line in lines:
      temp = line.replace("\n","").strip().split("\t")
      if line.split("\t")[0] in '''! " # $ % & \ ' ( ) * + , - . / : ; < = > ? @ [ \\ ] ^ _ ` { | } ~ '''.split():
          continue
      if(line == "\n"):
          datas.append(datas_temp)
          labels.append(labels_temp)
          max_sent_length = max(len(labels_temp), max_sent_length)
          datas_temp, labels_temp = "", []
      else:
          #print(temp)
          token_tmp = temp[0].replace("_"," ").lower()
          datas_temp += token_tmp + " "
          # print(token_tmp.split())
          if(len(token_tmp.split()) > 1):
            new_labels = expand_label_when_loaddata(token_tmp.split(), temp[1])
            labels_temp += new_labels
          else:
            labels_temp.append(temp[1])
    # print("Load successful...............")
  return datas, labels, max_sent_length
