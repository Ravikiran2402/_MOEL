import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import time
from utils.data_loader import prepare_data_seq
from utils import config
from model.transformer import Transformer
from model.transformer_mulexpert import Transformer_experts
from model.common_layer import evaluate, count_parameters, make_infinite , print_custum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from copy import deepcopy
from tqdm import tqdm
import os
import time
import numpy as np
import math
from tensorboardX import SummaryWriter
from utils.data_loader import collate_fn
import pickle
from utils.data_loader import *
from utils.beam_omt_experts import Translator
import copy


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
#% matplotlib inline

f=open("bert_embeddings_last_layer.txt",'w')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

_,_,_,vocab = load_dataset()

model = BertModel.from_pretrained('bert-base-uncased')
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
         
for text,v in tqdm(vocab.word2index.items()):
    #text = "embeddings"
    marked_text="[CLS] " + text + " [SEP]"

    tokenized_text = tokenizer.tokenize(marked_text)

    # Print out the tokens.
    #print (tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    #print (segments_ids)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    #model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    #model.eval()


    with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

    #print(encoded_layers[0][0][0][:5])

    token_embeddings = torch.stack(encoded_layers, dim=0)

    #print(token_embeddings.size(   ))

    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    token_embeddings = token_embeddings.permute(1,0,2)
    #token_embeddings = torch.squeeze(token_embeddings, dim=1)

    #print(token_embeddings.size())

    #token_sum = torch.sum(token_embeddings[-4:],dim = 0)


    #print(token_sum[0])
    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-1:],dim = 0)
        token_vecs_sum.append(sum_vec)

    #print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

    token_vecs = torch.stack(token_vecs_sum,dim = 0)

    word_embedding = torch.mean(token_vecs, dim = 0)
    #print(word_embedding[:5].data)
    #print(word_embedding.size()[0])
    assert(word_embedding.size()[0]==768)
    #print(word_embedding.data)
    #f.write()
    word_embedding_list = word_embedding.tolist()
    f.write(text+" ")
    for item in word_embedding_list:
        f.write("%s " % item)
    f.write("\n")
    #time.sleep(1)
    #break;

f.close()

