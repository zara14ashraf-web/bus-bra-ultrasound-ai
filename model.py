import torch
import torch.nn as nn
import timm


class SharedDualEfficientNetB3(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.full_branch = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        self.crop_branch = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        feature_dim = 1536

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),

            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        full_image,
        crop_image
    ):

        full_features = self.full_branch(
            full_image
        )

        crop_features = self.crop_branch(
            crop_image
        )

        combined = torch.cat(
            [full_features, crop_features],
            dim=1
        )

        return self.classifier(
            combined
        )
