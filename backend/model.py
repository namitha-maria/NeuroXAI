import torch
import torch.nn as nn
import timm


class DenseNet_ViT(nn.Module):
    def __init__(self, num_classes=3, embed_dim=256, max_slices=64):
        super().__init__()

        # ----------------------------
        # CNN BACKBONE (DenseNet)
        # ----------------------------
        self.cnn = timm.create_model(
            "densenet121",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        # 🔒 Freeze most of DenseNet
        for name, param in self.cnn.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        cnn_dim = self.cnn.num_features  # 1024

        # ----------------------------
        # PROJECTION
        # ----------------------------
        self.proj = nn.Linear(cnn_dim, embed_dim)

        # ----------------------------
        # POSITIONAL EMBEDDING
        # ----------------------------
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_slices, embed_dim)
        )

        # ----------------------------
        # TRANSFORMER ENCODER
        # ----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        # 🔒 Freeze transformer initially
        for p in self.transformer.parameters():
            p.requires_grad = False

        # ----------------------------
        # CLASSIFIER
        # ----------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        x: (B, S, 3, 224, 224)
        """
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feats = self.cnn(x)              # (B*S, 1024)
        feats = feats.view(B, S, -1)     # (B, S, 1024)

        tokens = self.proj(feats)        # (B, S, 256)
        tokens = tokens + self.pos_embed[:, :S, :]

        encoded = self.transformer(tokens)

        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
