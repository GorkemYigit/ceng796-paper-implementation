# (Hopefully) an accurate code approach to the
# generative model proposed in the paper.

# TODO: Add support for the different modes, and train for once.

import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=512):
        super(UNet, self).__init__()
        self.down1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        self.resnet_block = ResNetBlock(512, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)

        self.text_embedding = nn.Linear(embedding_dim, 512)
        self.style_embedding = nn.Linear(embedding_dim, 512)
        self.condition_proj = nn.Linear(512, 512)

    def forward(self, x, text_condition, style_condition, image_condition):
        text_embed = self.text_embedding(text_condition)
        style_embed = self.style_embedding(style_condition)
        combined_embed = text_embed + style_embed
        combined_embed = self.condition_proj(combined_embed).unsqueeze(2).unsqueeze(3)

        image_condition = image_condition.mean(dim=1).unsqueeze(2).unsqueeze(3)

        x1 = F.relu(self.down1(x))
        x2 = F.relu(self.down2(x1))
        x3 = F.relu(self.down3(x2))
        x4 = F.relu(self.down4(x3))

        x4 = x4 + combined_embed + image_condition  # Integrate conditions

        x5 = self.resnet_block(x4)

        x6 = F.relu(self.up1(x5))
        x7 = F.relu(self.up2(x6))
        x8 = F.relu(self.up3(x7))
        x9 = self.up4(x8)

        return x9
