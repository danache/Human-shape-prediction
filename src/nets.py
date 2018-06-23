import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.bn1 = nn.BatchNorm1d(input_features)
        self.fc1 = nn.Linear(input_features, 1)
        self.output_features = input_features + 1

    def forward(self, x):
        x = self.bn1(x)
        x = self.activation(x)
        x = self.fc1(x)
        return x


class DenseNet(nn.Module):

    def __init__(self,  config):
        super(DenseNet, self).__init__()
        input_features = 59
        self.number_of_layers = config['num_layers']
        self.k = config['k']
        l = 0

        for i in range(self.number_of_layers):
            l += 1
            dense_layer = DenseLayer(input_features, self.k, l, config)
            self.add_module('DenseLayer' + str(i), dense_layer)
            input_features = dense_layer.output_features

        self.out = nn.Linear(input_features, 10)


    def forward(self, x):
        for i in range(self.number_of_layers):
            x0 = self._modules['DenseLayer' + str(i)].forward(x)
            x = torch.cat((x0, x), dim=1)
        return self.out(x)

