import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None, k = 16 ):

        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))

        scores = torch.exp(torch.matmul(query, key.transpose(-2, -1))) \
                 / math.sqrt(query.size(-1))# matmul是tensor的乘法-1为列的个数-2为行数转置为了使k和v能够相乘 最后变成一个方阵

        if mask is not None:
            if k > scores.size()[3]:
                k = scores.size()[3]
            
            if k:

                v, _ = torch.topk(scores,k) #在scores中选择前k个最大值
                # print('打印出v')
                # print(v)
                # print(value)
                # vk = v[:, :, :, -1:]
                # print(vk)
                vk = v[:, :, :, -1:].expand_as(scores) #选择k个数最后一个数，也就是前k个最小的数 然后把v的维度扩展成和scores一样
                mask_k = torch.lt(scores, vk) #把scores中小于第k个数的值变为true
                scores = scores.masked_fill(mask_k, -1e9)#变为稀疏得分
                # print(scores)
                # scores = scores.masked_fill(mask == 0, -1e9)`
                scores = scores.masked_fill(mask == 0, -1e9)#最后把scores变为只有下三角是稀疏的
                # print(scores)
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn