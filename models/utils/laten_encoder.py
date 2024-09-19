'''
Implementation of latent encoder of neural process.

This is one of the modules of attentive neural process (ANP),
The others are: Deterministic Encoder, Decoder.

Structure:
context_x(batch, seq_len, input_d), context_y(batch, seq_len, output_d)
	V concatenation
 variable(batch, seq_len, input_d + output_d)
 	V pass final axis through MLP
 variable(batch, seq_len, latent_dim)
 	V optional self-attention phase
 variable(batch, seq_len, latent_dim)
 	V find mean of latent representations
 variable(batch, 1, latent_dim)
 	V reparameterization trick
'''

import torch
from torch import nn

from .mlp import MLP



class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_x_dim,
        input_y_dim,
        hidden_dim_list,    # the dims of hidden starts of mlps
        latent_dim=32,      # the dim of last axis of sc and z..
        # self_attention_type="dot",
        # use_self_attn=True,
        # attention_layers=2,
        use_lstm=False
    ):
        super().__init__()
        self.input_dim = input_x_dim + input_y_dim
        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.latent_dim = latent_dim

        if latent_dim != hidden_dim_list[-1]:
            print('Warning, Check the dim of latent z and the dim of mlp last layer!')

        # NOTICE: On my paper, we seems to substitute the mlp with LSTM
        #  but we actually add a LSTM before the mlp
        if use_lstm:
            # self._encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout,
            #                           num_layers=n_encoder_layers)
            pass
        else:
            self.latent_encoder_mlp = MLP(input_size=self.input_dim, output_size_list=hidden_dim_list)
            # Output should be (b, seq_len, hidden_dim_list[-1])

        # if use_self_attn:
        #     self._self_attention = Attention(
        #         self.latent_dim,
        #         self_attention_type,
        #         attention_layers,
        #         rep="identity",
        #     )
            pass
        self.penultimate_hidden_num = int(0.5*(self.hidden_dim + self.latent_dim))
        # print('0.5*(self.hidden_dim + self.latent_dim) =', 0.5*(self.hidden_dim + self.latent_dim))
        self.penultimate_layer = nn.Linear(self.hidden_dim, self.penultimate_hidden_num)

        self.mean_layer = nn.Linear(self.penultimate_hidden_num, self.latent_dim)
        self.std_layer = nn.Linear(self.penultimate_hidden_num, self.latent_dim)
        self._use_lstm = use_lstm
        # self._use_self_attn = use_self_attn

    def forward(self, x, y):
        # print('x.size() =', x.size())
        # print('y.size() =', y.size())
        encoder_input = torch.cat([x, y], dim=-1)
        # encoder_input (b, seq_len, input_dim=input_x_dim + input_y_dim)
        # = (b, seq_len, input_dim)

        # NOTICE:Pass final axis through MLP
        # print('encoder_input.size() =', encoder_input.size())
        hidden = self.latent_encoder_mlp(encoder_input)
        # hidden (b, seq_len, hidden_dim)