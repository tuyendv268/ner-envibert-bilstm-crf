from src.utils import get_input_for_ner
from transformers import RobertaTokenizer, RobertaModel
from importlib.machinery import SourceFileLoader
from src import utils
from src.resources import hparams
import torch
from src.model.vndgNER import vndgNER
import os

cuda = 'cpu'
pretrained_path = hparams.pretrained_envibert
tokenizer = SourceFileLoader("envibert.tokenizer", os.path.join(pretrained_path,'envibert_tokenizer.py')).load_module().RobertaTokenizer(pretrained_path)
roberta = RobertaModel.from_pretrained('nguyenvulebinh/envibert',cache_dir=pretrained_path)
model = vndgNER(cuda=cuda,nb_label=hparams.nb_labels, roberta=roberta).to(cuda) 
model.load_state_dict(torch.load(hparams.warm_up, map_location=torch.device('cpu')))


def infer_entity(text):

    input_ids, input_masks = get_input_for_ner(sent=text, tokenizer=tokenizer)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    input_ids = torch.tensor([input_ids])
    input_masks = torch.tensor([input_masks])

    pred = model(input_ids, input_masks, None)

    join_tokens, join_tags = utils.join(tokens, pred[0])
    join_tags = utils.convert_ids2label(join_tags)
    
    output = [(token, tag) for token, tag in zip(join_tokens, join_tags)]
    return output

