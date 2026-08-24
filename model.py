```python
import torch
import torch.nn as nn
import timm


class SharedDualEfficientNetB3(nn.Module):
    """
    BUS-BRA Dual-View EfficientNet-B3

    Branch 1:
        Full ultrasound image

    Branch 2:
        Lesion-focused crop

    Feature fusion:
        1536 + 1536 = 3072

    Classifier:
        3072 -> 512 -> 128 -> 2
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # Full ultrasound branch
        self.full_branch = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        # Lesion-focused branch
        self.crop_branch = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        # EfficientNet-B3 feature dimension
        feature_dim = 1536

        # Classification head
        self.classifier = nn.Sequential(

            nn.Linear(
                feature_dim * 2,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(0.30),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                num_classes
            )
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
            [
                full_features,
                crop_features
            ],
            dim=1
        )

        output = self.classifier(
            combined
        )

        return output
```
