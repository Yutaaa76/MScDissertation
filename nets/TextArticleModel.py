import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class TextArticleModel(torch.nn.Module):

    def __init__(self, num_classes, text_dim, claim_dim, conv_num, fc_num, title_conv_num, content_conv_num):
        super(TextArticleModel, self).__init__()
        # self.fusion = nn.Linear((text_dim + vision_dim), fusion_output_size)
        self.text_dim = text_dim
        self.claim_dim = claim_dim

        self.conv_total = nn.Sequential(
            nn.Conv1d(in_channels=768 * 4, out_channels=conv_num, stride=3, kernel_size=1),
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

        self.conv_text = nn.Sequential(
            nn.Conv1d(in_channels=text_dim, out_channels=conv_num, stride=3, kernel_size=1),
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

        self.conv_title = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=title_conv_num, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=title_conv_num, out_channels=title_conv_num, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=title_conv_num, out_channels=title_conv_num, stride=3, kernel_size=1),
            nn.ReLU()
        )

        self.conv_content = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=content_conv_num, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=content_conv_num, out_channels=content_conv_num, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=content_conv_num, out_channels=content_conv_num, stride=3, kernel_size=1),
            nn.ReLU()
        )
        self.conv_lang = nn.Sequential(
            nn.Conv1d(in_channels=46, out_channels=256, stride=3, kernel_size=1),
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

        self.fc_1 = nn.Linear(in_features=conv_num * 2 + title_conv_num + content_conv_num,
                              out_features=fc_num)
        self.fc_2 = nn.Linear(in_features=self.fc_1.out_features,
                              out_features=fc_num)
        self.fc_3 = nn.Linear(in_features=self.fc_2.out_features,
                              out_features=num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # def forward(self, text, img, label):
    def forward(self, text, claim, title, content, lang):
        # text = text.squeeze()
        # claim = claim.squeeze()
        # title = title.squeeze()
        # content = content.squeeze()
        # fused = torch.cat((text, claim, title, content), dim=1)
        # fused = self.conv_total(fused).squeeze()
        text = self.conv_text(text).squeeze()
        claim = self.conv_claim(claim).squeeze()
        title = self.conv_title(title).squeeze()
        # lang = self.conv_lang(lang).squeeze()
        # print(text.shape, lang.shape)
        content = self.conv_content(content).squeeze()
        # print(img_features.shape, text.shape, claim.shape)

        fused = torch.cat((text, claim, title, content), dim=1)

        logits = self.fc_1(fused)
        logits = self.relu(logits)
        logits = self.fc_2(logits)
        logits = self.relu(logits)
        logits = self.fc_3(logits)
        # pred = F.softmax(logits)

        return logits

