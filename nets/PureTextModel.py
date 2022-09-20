import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class PureTextModel(torch.nn.Module):

    def __init__(self, num_classes, text_dim, claim_dim, fusion_output_size):
        super(PureTextModel, self).__init__()
        # self.fusion = nn.Linear((text_dim + vision_dim), fusion_output_size)
        self.claim_dim = claim_dim
        self.text_dim = text_dim

        self.text_model = nn.LSTM(input_size=2,
                                  hidden_size=text_dim + claim_dim,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)
        # self.claim_model = nn.LSTM(input_size=1,
        #                            hidden_size=claim_dim,
        #                            num_layers=1,
        #                            batch_first=True,
        #                            bidirectional=False)

        # self.resnet = models.resnet50()
        # self.resnet.fc = nn.Linear(claim_dim + text_dim, num_classes)

        self.fusion = torch.nn.Linear(in_features=(claim_dim + text_dim),
                                      out_features=fusion_output_size)

        self.fc_1 = nn.Linear(in_features=claim_dim + text_dim,
                              out_features=num_classes)
        self.fc_2 = nn.Linear(self.fc_1.out_features, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, claim, text):
        # self.claim_model.flatten_parameters()
        self.text_model.flatten_parameters()
        combined = torch.cat((text, claim), dim=2)
        out_1, (h, _) = self.text_model(combined)
        # out_2, (text_h, _) = self.text_model(text)
        # combined_features = torch.cat((text_h, claim_h), dim=2)
        # print(combined_features.shape)
        # fused = self.fusion(combined_features)
        # fused = self.relu(fused)
        # print(h.shape)
        logits = self.fc_1(h)
        # logits = self.relu(logits)
        # logits = self.fc_2(logits)
        # logits = self.relu(logits)

        # logits = self.sigmoid(logits)

        return logits

