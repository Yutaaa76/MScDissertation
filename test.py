from torchvision import models
import torch.nn as nn

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


res50_model = models.resnet50(pretrained=True)
res50_conv2 = ResNet50Bottom(res50_model)

outputs = res50_conv2(inputs)
outputs.data.shape  # => torch.Size([4, 2048, 7, 7])