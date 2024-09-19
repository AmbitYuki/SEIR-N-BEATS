
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_size, output_size_list):
        '''
        Parameters:
        -input_size (int): number of dimensions for each point in each sequence.
        -output_size_list (list of ints): number of output dimensions for each layer.
        '''
        super().__init__()
        self.input_size = input_size  #
        self.output_size_list = output_size_list  #
        network_size_list = [input_size] + self.output_size_list  #
        network_list = []

        # 建立神经网络.
        for i in range(1, len(network_size_list) - 1):
            network_list.append(nn.Linear(network_size_list[i-1], network_size_list[i], bias=True))
            network_list.append(nn.ReLU())

        # Add final layer, create sequential container.其实就是建立一个三层的神经网络
        network_list.append(nn.Linear(network_size_list[-2], network_size_list[-1]))
        self.mlp = nn.Sequential(*network_list)

    def forward(self, x):
        # encoder_input (b, seq_len, input_dim=input_x_dim + input_y_dim)
        # = (b, seq_len, input_dim)

        # Pass final axis through MLP
        # hidden_r_i = self.deter_encoder_mlp(encoder_input)
        return self.mlp(x)
