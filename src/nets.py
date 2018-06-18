import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_nn.utils import init_weights


def map_activation(name):
    return {
        'relu': F.relu,
        'elu': F.elu,
        'tanh': F.tanh,
        'leaky_relu': F.leaky_relu,
    }[name]

# DenseNet with BatchNorm, 50 layers: 1.00000e-02 * 6.3227
# DenseNet with NoneNorm 1.11770272
# DenseNet with LayerNorm 0.3794
# DenseNet with BatchNorm, 100 layers:1.00000e-02 * 3.1621

class DenseLayer(nn.Module):
    # k0 - the number of features in the input layer
    # k  - the growth rate
    # l  - the current layer
    def __init__(self, input_features, k, l, config):
        super(DenseLayer, self).__init__()
        self.activation = map_activation(config['activation'])
        self.bn1 = nn.BatchNorm1d(input_features + k * (l - 1))
        self.conv1 = nn.Conv1d(input_features + k * (l - 1), 4 * k, 1)
        self.bn2 = nn.BatchNorm1d(4 * k)
        self.conv2 = nn.Conv1d(4 * k, k, 3, padding=1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

class DenseBlock(nn.Module):

    def __init__(self, input_features, config):
        super(DenseBlock, self).__init__()
        self.number_of_layers = config['num_layers']
        self.k = config['k']
        l = 0
        for i in range(self.number_of_layers):
            l += 1
            dense_layer = DenseLayer(input_features, self.k, l, config)
            self.add_module('DenseLayer' + str(i), dense_layer)
        self.output_features = input_features + self.k * self.number_of_layers


    def forward(self, x):
        for i in range(self.number_of_layers):
            x0 = self._modules['DenseLayer' + str(i)].forward(x)
            x = torch.cat((x0, x), dim=1)
        return x

class TransitionLayer(nn.Module):

    def __init__(self, input_features):
        super(TransitionLayer, self).__init__()
        self.output_features = int(input_features * 0.5)
        self.bn1 = nn.BatchNorm1d(input_features)
        self.conv1 = nn.Conv1d(input_features, self.output_features, kernel_size=1)
        self.pool1 = nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.pool1(x)
        return x

class EntranceLayer(nn.Module):

    def __init__(self, k0):
        super(EntranceLayer, self).__init__()
        self.conv1 = nn.Conv1d(40, k0, kernel_size=7, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, config, k0=40):
        super(DenseNet, self).__init__()
        self.config = config

        self.entrance = EntranceLayer(k0)
        self.num_layers = config["num_layers"]
        self.num_blocks = config["num_blocks"]
        features = k0

        for i in range(0, self.num_blocks):
            dense_block = DenseBlock(features, config)
            self.add_module('dense_block' + str(i), dense_block)
            if i != self.num_blocks - 1:
                trans_layer = TransitionLayer(dense_block.output_features)
                self.add_module('transition_layer' + str(i), trans_layer)
                features = trans_layer.output_features
            else:
                features = dense_block.output_features

        self.beta = nn.Conv1d(features, 10, 1)
        # Initialize weights
        self.apply(init_weights(config))

    def forward(self, x):
        x = self.entrance.forward(x)
        for i in range(0, self.num_blocks):
            x = self._modules['dense_block' + str(i)].forward(x)
            # Ends by a dense block
            if i != self.num_blocks - 1:
                x = self._modules['transition_layer' + str(i)].forward(x)
        beta = self.beta(x)
        beta = torch.mean(beta, 2)
        beta = torch.squeeze(beta)
        return  beta




