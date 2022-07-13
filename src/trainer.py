import os
import torch
from torch import optim
from tqdm import tqdm
from src.resources import hparams
from src.utils import get_data_for_ner
from seqeval.metrics import classification_report
import json
import numpy as np

class trainer():
    def __init__(self, model, train_dl, val_dl, max_epoch, optimizer, cuda, warm_up=None):
        self.model = model
        self.train_dl = train_dl
        self.optimizer = optimizer
        self.label2index = hparams.label2index
        self.index2label = hparams.index2label
        self.val_dl = val_dl
        self.max_epoch = max_epoch
        self.warm_up = warm_up
        self.cuda = cuda
        
        if warm_up != None:
            if os.path.exists(warm_up):
                print("Warm up from: ", warm_up)
                model.load_state_dict(torch.load(warm_up))
    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path))
        
    
    def train(self):
        print("---------------start training---------------")
        traning_loss=[]
        vali_loss=[]
        for epoch in range(self.max_epoch):
            loss_list=[]
            train_tqdm = tqdm(self.train_dl)
            for input_data in train_tqdm:
                self.model.zero_grad()
                input_ids = input_data["input_ids"].to(self.cuda)
                input_masks = input_data["input_masks"].to(self.cuda)
                label = input_data["label"].to(self.cuda)
                label_masks = input_data["label_masks"].to(self.cuda)

                loss = self.model.loss(input_ids, input_masks, label, label_masks)
                loss_list.append(loss.item())

                train_tqdm.set_postfix({"epoch":epoch, "loss":torch.tensor(loss_list).to(self.cuda).mean()})
                loss.backward()
                self.optimizer.step()
            if ((epoch) % 1 == 0):
                #PATH = f'./checkpoint/checkpoint_{epoch}.pt'
                PATH = hparams.checkpoint_path.replace("%EPOCH%", str(epoch))
                torch.save(self.model.state_dict(), PATH)
                print("Saved checkpoint: ", PATH)
                print("--------validate--------")
                results = self.val()
                path = hparams.res_path.replace("%EPOCH%", str(epoch))
                with open(path, "w") as tmp:
                    tmp.write(str(results))
                    print("Saved: ", path)
                print("----------done----------")
        print("-------------------done------------------")
    
    def infer(self, text):
        """Các CLB châu Âu hiếm khi để ý tới những cầu thủ châu Á", Castets nói với VnExpress."""
        text = text.lower()
        input_ids, input_masks = get_data_for_ner(sent=text, tokenizer=self.tokenizer)

        _, pred = self.model(input_ids, input_masks)
        print(pred)

    def f1_sc(self, actual, predicted, label):
      TP = torch.sum((actual==label)&(predicted==label))
      FP = torch.sum((actual!=label)&(predicted==label))
      FN = torch.sum((predicted!=label)&(actual==label))

      precision = TP/(TP+FP)
      recall = TP/(TP+FN)
      F1 = 2*(precision*recall)/(precision+recall)
      return F1

    def f1_scores(self, actual, predicted, ignore_label):
        f1 = torch.tensor([], device=self.cuda)
        # print(actual[actual>ignore_label])
        for label in torch.unique(actual[actual>ignore_label]):
            tmp_score = self.f1_sc(actual, predicted, label)
            f1 = torch.cat((f1, tmp_score.view(1)),dim=0)
        f1_macro = torch.mean(f1)
        f1 = torch.cat((torch.tensor([-1]*(ignore_label+1),device=self.cuda), f1),dim=0)
        return f1_macro, f1

    def is_equal(predict, label):
        for pred, lbl in zip(predict, label):
          if len(pred) != len(lbl):
            return False
        return True
    def remove_padding(self, labels, masks):
        label = [sent[0:mask.sum(dim=0)] for sent, mask in zip(labels, masks)]
        return label
    def convert_id2label(self, input_ids):
        output = [[self.index2label[str(token)] for token in inp] for inp in input_ids]
        return output
    def ignore_label(self, predicts, labels, ignore_label):
        predicts = [[pred for pred, lbl in zip(sent_pred, sent_label) if lbl not in ignore_label] for sent_pred, sent_label in zip(predicts, labels)]
        labels = [[token for token in sent_label if token not in ignore_label] for sent_label in labels]

        return predicts, labels

    def val(self):
        predicts = []
        labels = []
        val_tqdm = tqdm(self.val_dl)
        for idx, input_data in enumerate(val_tqdm):
            input_ids = input_data["input_ids"].to(self.cuda)
            input_masks = input_data["input_masks"].to(self.cuda)
            label = input_data["label"].to(self.cuda)
            label_masks = input_data["label_masks"].to(self.cuda)

            predict = self.model(input_ids, input_masks, label_masks)

            label = self.remove_padding(label.tolist(), label_masks)
            predict, label = self.ignore_label(predict, label, [0, 1, 2])

            predicts += predict
            labels += label
        labels = self.convert_id2label(labels)
        predicts = self.convert_id2label(predicts)

        results = classification_report(labels,predicts)
        print(results)
        return results
    
    def test(self, test_dl):
        predicts = []
        labels = []
        test_tqdm = tqdm(test_dl)
        for idx, input_data in enumerate(test_tqdm):
            input_ids = input_data["input_ids"].to(self.cuda)
            input_masks = input_data["input_masks"].to(self.cuda)
            label = input_data["label"].to(self.cuda)
            label_masks = input_data["label_masks"].to(self.cuda)

            _, pred = self.model(input_ids, input_masks, label_masks)
            
            predict = []
            for pred_lb in pred:
                predict += pred_lb
            predict = torch.tensor(predict).to(self.cuda)

            # print(label)
            label = [sent[0:mask] for sent,mask in zip(label, label_masks.sum(dim=1))]

            list_label = []
            for lb in label:
                list_label += lb
            list_label = torch.tensor(list_label).to(self.cuda)
            # print("tmp-log: ",label)
            if idx == 0:
                predicts = predict
                labels = list_label.view(-1)
            else:
                predicts = torch.concat((predicts, predict), dim=0).to(self.cuda)
                labels = torch.concat((labels, list_label.view(-1)), dim=0).to(self.cuda)
        accuracy = self.accuracy(predicts=predicts, labels=labels)
        print("Accuracy : ", accuracy)
        return accuracy