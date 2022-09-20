import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self):
        super(BertClassifier).__init__()

