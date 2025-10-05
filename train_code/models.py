import torch
import torch.nn as nn
import timm

class TimmModel(nn.Module):
    def __init__(self, num_labels, model_name):
        super().__init__()
        # Load pretrained model
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.num_features = self.backbone.feature_info[-1]['num_chs']
        self.backbone.reset_classifier(0)
        self.head = nn.Linear(self.num_features, num_labels, bias=True)

    def forward(self, x, return_hidden=False):
        feats = self.backbone(x)
        return feats if return_hidden else self.head(feats)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def classifier_parameters(self):
        return self.head.parameters()

    def set_backbone(self, requires_grad):
        for param in self.backbone_parameters():
            param.requires_grad = requires_grad

class TemporalTransformer(nn.Module):
    def __init__(self, model, hidden_dim, num_labels, num_layers=2, num_heads=4):
        super().__init__()
        self.static_model = model
        self.projection = nn.Linear(model.num_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):  # x: [B, seq_len, num_labels, img_size, img_size]
        x_reshaped = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.static_model(x_reshaped, return_hidden=True).view(x.size(0), x.size(1), -1) # [B, seq_len, hidden_dim]
        x = self.projection(x)
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, hidden_dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to [B, seq_len, hidden_dim]
        out = self.classifier(x)  # [B, seq_len, num_labels]
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.downsample = nn.Identity()  # Keep for structure
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):  # x: [B, C, T]
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual  # Residual connection
        return x

class TemporalTCN(nn.Module):
    def __init__(self, model, hidden_dim, num_labels, num_blocks):
        super().__init__()
        self.static_model = model
        self.input_proj = nn.Conv1d(model.num_features, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dilation=2**i) for i in range(num_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, num_labels, kernel_size=1)

    def forward(self, x):  # x: [B, seq_len, num_labels, img_size, img_size]
        x_reshaped = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.static_model(x_reshaped, return_hidden=True).view(x.size(0), x.size(1), -1)
        x = x.transpose(1, 2)         # [B, 1536, seq_len] → [B, C, T]
        x = self.input_proj(x)        # [B, hidden_dim, seq_len]
        x = self.blocks(x)            # temporal modeling
        x = self.output_proj(x)      # [B, num_labels, seq_len]
        x = x.transpose(1, 2)         # [B, seq_len, 3]
        return x

class TemporalLSTM(nn.Module):
    def __init__(self, model, hidden_dim, num_labels, num_layers, bidirectional=False):
        super().__init__()
        self.static_model = model
        self.lstm = nn.LSTM(input_size=model.num_features,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_labels)

    def forward(self, x):  # x: [B, seq_len, num_labels, img_size, img_size]
        x_reshaped = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.static_model(x_reshaped, return_hidden=True).view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)  # output: [B, seq_len, hidden_dim*2]
        out = self.classifier(x)  # [B, seq_len, num_labels]
        return out
