"""PyTorch CNN + LSTM architecture for T06 MNIST Summation."""

from torch import nn


def build_cnn_feature_extractor(in_channels=1, channels=(64, 128, 256), dropout=0.2):
    """Build the 3-block CNN backbone shared with the MNIST CNN example."""
    layers = []
    cur_channels = in_channels
    for out_channels in channels:
        layers.extend([
            nn.Conv2d(cur_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        ])
        cur_channels = out_channels
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


def _init_keras_style_weights(model):
    """Match Keras default initializers (Glorot uniform, orthogonal LSTM, unit forget bias)."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    if "bias_ih" in name:
                        hidden = module.hidden_size
                        param.data[hidden : 2 * hidden].fill_(1.0)


class MnistCnnLstm(nn.Module):
    """PyTorch conversion of Keras `MNIST_CNN_LSTM` for digit sequence summation.

    Expects input tensor of shape `[B, T, 1, 28, 28]` and outputs `[B, 1]`
    predicted sum logits (linear regression head for `nn.L1Loss`).
    """

    def __init__(
        self,
        in_channels=1,
        cnn_channels=(64, 128, 256),
        cnn_dropout=0.2,
        lstm_hidden=128,
        head_dropout=0.22,
        keras_init=True,
    ):
        super().__init__()
        self.cnn = build_cnn_feature_extractor(
            in_channels=in_channels,
            channels=cnn_channels,
            dropout=cnn_dropout,
        )
        # 28x28 after 3x MaxPool2d(2) -> 3x3 spatial map
        cnn_feature_dim = cnn_channels[-1] * 3 * 3
        self.lstm1 = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(lstm_hidden, 1)

        if keras_init:
            _init_keras_style_weights(self)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D sequence tensor [B, T, C, H, W], got {tuple(x.shape)}")
        batch_size, seq_len, channels, height, width = x.shape
        # PyTorch equivalent of Keras TimeDistributed(CNN + Flatten)
        flat_frames = x.reshape(batch_size * seq_len, channels, height, width)
        frame_features = self.cnn(flat_frames).reshape(batch_size, seq_len, -1)

        seq_out, _ = self.lstm1(frame_features)
        seq_out, _ = self.lstm2(seq_out)
        last_hidden = seq_out[:, -1, :]
        return self.head(self.lstm_dropout(last_hidden))


def count_parameters(model):
    """Return total PyTorch parameter count and Keras-equivalent count (single bias per LSTM)."""
    pytorch_params = sum(p.numel() for p in model.parameters())
    extra_lstm_bias = sum(
        4 * m.hidden_size for m in model.modules() if isinstance(m, nn.LSTM) and m.bias
    )
    return {
        "pytorch_params": pytorch_params,
        "keras_equivalent_params": pytorch_params - extra_lstm_bias,
        "extra_lstm_bias_params": extra_lstm_bias,
    }
