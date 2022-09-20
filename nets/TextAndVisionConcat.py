import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class TextAndVisionConcat(torch.nn.Module):

    def __init__(self, num_classes, text_dim, claim_dim, vision_dim, fusion_output_size):
        super(TextAndVisionConcat, self).__init__()
        # self.fusion = nn.Linear((text_dim + vision_dim), fusion_output_size)
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.claim_dim = claim_dim

        # get vision feature vector
        self.vis_model = models.resnet34(pretrained=True)
        self.vis_model = torch.nn.Sequential(*(list(self.vis_model.children())[:-1]))
        # self.vis_fc_2 = nn.Linear(256, 128)

        self.conv_text = nn.Sequential(
            nn.Conv1d(in_channels=text_dim, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU()
        )

        self.conv_claim = nn.Sequential(
            nn.Conv1d(in_channels=claim_dim, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU()
        )

        self.conv_title = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU()
        )

        self.conv_content = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(stride=3, kernel_size=1),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=1024, stride=3, kernel_size=1),
            nn.ReLU()
        )

        self.fc_1 = nn.Linear(in_features=1024 * 2 + 512,
                              out_features=1024)
        self.fc_2 = nn.Linear(in_features=1024,
                              out_features=num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # def forward(self, text, img, label):
    def forward(self, img, text, claim, title, content):
        img = self.vis_model(img).squeeze()
        text = self.conv_text(text).squeeze()
        claim = self.conv_claim(claim).squeeze()
        # title = self.conv_title(title).squeeze()
        # content = self.conv_content(content).squeeze()
        # print(img_features.shape, text.shape, claim.shape)
        fused = torch.cat((img, text, claim), dim=1)

        logits = self.fc_1(fused)
        logits = self.relu(logits)
        logits = self.fc_2(logits)
        # pred = F.softmax(logits)

        return logits

