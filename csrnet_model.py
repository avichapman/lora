import collections
import torch
import torch.nn as nn
from typing import Tuple


class CSRNet(nn.Module):
    r"""CSRNet Model. Originally from https://github.com/leeyeehoo/CSRNet-pytorch.

    Arguments:
        weights_path: str. The path to the pretrained weights for the model.
    """

    def __init__(self, weights_path: str):
        super().__init__()
        self.seen = 0
        self.entry_feat = [64, 64]
        self.frontend_1_feat = ['M', 128, 128]
        self.frontend_2_feat = ['M', 256, 256, 256]
        self.frontend_3_feat = ['M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.entry = self._make_layers(self.entry_feat)
        self.frontend_1 = self._make_layers(self.frontend_1_feat, in_channels=64)
        self.frontend_2 = self._make_layers(self.frontend_2_feat, in_channels=128)
        self.frontend_3 = self._make_layers(self.frontend_3_feat, in_channels=256)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        checkpoint = torch.load(weights_path)
        if 'version' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
        else:
            self.load_old_state_dict(checkpoint['state_dict'])

    def load_old_state_dict(self, state_dict: dict, strict=True):
        r"""Loads the weights from a pre-trained CSRNet, making adjustments for the slightly different structure of this
        CSRNet.

        Args:
            state_dict: The data to load.
            strict: bool. If true, enforce all entries matching.
        """
        converted_state = collections.OrderedDict()
        for key, input_param in state_dict.items():
            if key.find("frontend.0") == 0:
                converted_state["entry.0." + key[len("frontend.0."):]] = input_param
            elif key.find("frontend.21") == 0:
                converted_state["frontend_3.5." + key[len("frontend.21."):]] = input_param
            elif key.find("frontend.2") == 0:
                converted_state["entry.2." + key[len("frontend.2."):]] = input_param
            elif key.find("frontend.5") == 0:
                converted_state["frontend_1.1." + key[len("frontend.5."):]] = input_param
            elif key.find("frontend.7") == 0:
                converted_state["frontend_1.3." + key[len("frontend.7."):]] = input_param
            elif key.find("frontend.10") == 0:
                converted_state["frontend_2.1." + key[len("frontend.10."):]] = input_param
            elif key.find("frontend.12") == 0:
                converted_state["frontend_2.3." + key[len("frontend.12."):]] = input_param
            elif key.find("frontend.14") == 0:
                converted_state["frontend_2.5." + key[len("frontend.14."):]] = input_param
            elif key.find("frontend.17") == 0:
                converted_state["frontend_3.1." + key[len("frontend.17."):]] = input_param
            elif key.find("frontend.19") == 0:
                converted_state["frontend_3.3." + key[len("frontend.19."):]] = input_param
            elif key.find("backend.") == 0 or key.find("output_layer.") == 0:
                converted_state[key] = input_param
            else:
                raise IndexError("Unexpected key: " + key)

        super().load_state_dict(converted_state, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Runs data through the network.

        Args:
            x: torch.Tensor. 4D Tensor. Shape (Batch Size, Channel Count = 3, Image Height, Image Width)
        Returns
        -------
        torch.Tensor. Density map
        """
        x = self.entry(x)
        x = self.frontend_1(x)
        x = self.frontend_2(x)
        x = self.frontend_3(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def forward_parts(self, x: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Runs data through the network. Returns the final result as well as the early data before downsizing.
        The image data is put through an initial round of convolutions. This early data is returned in the first output
        from this method. The data is also run through the rest of the network and the final density map is returned
        in the second tensor.

        Args:
            x: torch.Tensor. 4D Tensor. Shape (Batch Size, Channel Count = 3, Image Height, Image Width)
        Returns
        -------
        torch.Tensor. Massaged entry data. Shape (Batch Size, Channel Count = 64, Image Height, Image Width)
        torch.Tensor. Output from first layer of front end.
                      Shape (Batch Size, Channel Count = 128, Image Height / 2, Image Width / 2)
        torch.Tensor. Output from second layer of front end.
                      Shape (Batch Size, Channel Count = 256, Image Height / 4, Image Width / 4)
        torch.Tensor. Backend Output. Shape (Batch Size, Channel Count = 64, Image Height / 8, Image Width / 8)
        torch.Tensor. Density map. Shape (Batch Size, Channel Count = 1, Image Height / 8, Image Width / 8)
        """
        entry_data = self.entry(x)
        frontend_1_data = self.frontend_1(entry_data)
        frontend_2_data = self.frontend_2(frontend_1_data)
        frontend_3_data = self.frontend_3(frontend_2_data)
        backend_data = self.backend(frontend_3_data)
        x = self.output_layer(backend_data)
        return entry_data, frontend_1_data, frontend_2_data, backend_data, x

    @staticmethod
    def _make_layers(cfg: list, in_channels: int = 3, dilation: bool = False) -> nn.Sequential:
        r"""Constructs the layers of the model, based on the configuration values in 'cfg'. Each entry in the 'cfg',
        corresponds to a layer. The entry contains the number of features to output from that layer. If the entry is
        'M', instead of a number, a Max Pooling layer is used.

        Args:
            cfg: list. A list of feature counts, or 'M' for a max pooling layer.
            in_channels: int. The number of input channels to the first layer.
            dilation: bool. If true, apply a 2x dilation to the convolutions.
        Returns
        -------
        nn.Sequential. The layers of the model that were constructed.
        """
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
