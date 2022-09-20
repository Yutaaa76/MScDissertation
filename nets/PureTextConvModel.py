import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class PureTextConvModel(torch.nn.Module):

    def __init__(self, num_classes, text_dim, claim_dim, lang_dim, fc_num, conv_num):
        super(PureTextConvModel, self).__init__()
        # self.fusion = nn.Linear((text_dim + vision_dim), fusion_output_size)
        self.claim_dim = claim_dim
        self.text_dim = text_dim

        self.conv_text = nn.Sequential(
            nn.Conv1d(in_channels=text_dim + 2, out_channels=conv_num, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=conv_num, out_channels=conv_num, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=conv_num, out_channels=conv_num, stride=3, kernel_size=1),
            nn.ReLU()
        )
        self.conv_claim = nn.Sequential(
            nn.Conv1d(in_channels=claim_dim, out_channels=conv_num, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=conv_num, out_channels=conv_num, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=conv_num, out_channels=conv_num, stride=3, kernel_size=1),
            nn.ReLU()
        )
        self.conv_lang = nn.Sequential(
            nn.Conv1d(in_channels=lang_dim, out_channels=256, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=256, out_channels=256, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=256, out_channels=256, stride=3, kernel_size=1),
            nn.ReLU()
        )

        self.fc_1 = nn.Linear(in_features=conv_num * 2 + 256,
                              out_features=fc_num)
        self.fc_2 = nn.Linear(self.fc_1.out_features, fc_num)
        self.fc_3 = nn.Linear(self.fc_2.out_features, num_classes)

        # self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, claim, lang, num_rep, num_ret):
        # num_ret = num_ret[:, 0]
        # num_rep = num_rep[:, 0]
        # lang = lang.squeeze()
        # print(text.shape, num_ret.shape, lang.shape)
        # self.claim_model.flatten_parameters()
        # self.text_model.flatten_parameters()

        text = torch.cat((text, num_ret, num_rep), dim=1)
        # text = torch.cat((text, lang, num_rep, num_ret), dim=1)
        text = self.conv_text(text).squeeze()
        # print(text.shape)
        # print(num_rep.shape)
        # text = torch.cat((text, num_ret, num_rep), dim=1)
        claim = self.conv_claim(claim).squeeze()

        # print(text.shape, lang.shape, num_rep.shape)
        lang = self.conv_lang(lang).squeeze()
        # combined = torch.cat((text, claim), dim=1)
        combined = torch.cat((text, claim, lang), dim=1)

        logits = self.fc_1(combined)
        logits = self.relu(logits)
        logits = self.fc_2(logits)
        logits = self.relu(logits)
        logits = self.fc_3(logits)
        # logits = self.sigmoid(logits)

        return logits

