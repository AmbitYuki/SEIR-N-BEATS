import torch.nn as nn


class featuresEmbedding(nn.Embedding):
    def __init__(self, features_size, embed_size, padding_idx):
        super().__init__(features_size, embed_size, padding_idx=padding_idx)# padding_idx not 0 加入的