from src.model import bilstm
from torchcrf import CRF
from src.resources import hparams
from torch import nn

class vndgNER(nn.Module):
  def __init__(self, nb_label, cuda, roberta):
    super().__init__()
    self.model = roberta
    self.bilstm = bilstm.BiLSTM(cuda, emb_dim=768, hidden_dim=256)
    self.linear = nn.Linear(256, nb_label).to(cuda)
    self.dropout = nn.Dropout(hparams.dropout)
    self.crf = CRF(num_tags=nb_label, batch_first=True)

  def forward(self, input_ids, input_masks, label_masks):
    output = self.model(input_ids=input_ids, attention_mask = input_masks)
    sequence_output, pooled_output = output[0], output[1]
    sequence_output = self.bilstm(sequence_output)
    emissions = self.linear(sequence_output)

    path = self.crf.decode(emissions, mask=label_masks)
    return path
  def loss(self, input_ids, input_masks, labels, label_masks):
    output = self.model(input_ids=input_ids, attention_mask = input_masks)
    sequence_output, pooled_output = output[0], output[1]
    sequence_output = self.dropout(sequence_output)
    sequence_output = self.bilstm(sequence_output)
    emissions = self.linear(sequence_output)

    nll = -self.crf(emissions, labels, mask=label_masks)
    return nll