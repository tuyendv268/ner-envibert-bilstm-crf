from torch.utils.data import Dataset
from src.utils import get_data_for_ner
import torch

class NerDataset(Dataset):
  def __init__(self, sents, labels, tokenizer, max_sent_lenth):
    self.sents = sents
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_sent_lenth = max_sent_lenth

  def __len__(self):
    return len(self.sents)
  
  def __getitem__(self, index):
    sentence=self.sents[index]
    label=self.labels[index]
    max_sent_length = self.max_sent_lenth

    input_ids, input_masks, label, label_masks = get_data_for_ner(
        sent = sentence,
        tokenizer=self.tokenizer,
        label = label, 
        max_sent_length = max_sent_length
    )
    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    label = torch.tensor(label)
    label_masks = torch.tensor(label_masks)

    return {
        "input_ids": input_ids,
        "input_masks": input_masks, 
        "label": label, 
        "label_masks": label_masks
    }
