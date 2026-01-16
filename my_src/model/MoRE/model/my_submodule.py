import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentivePooling(nn.Module):                                  #自学习池化层，相对比简单的加和，引入权重，对于重要的部分权重大，加和比重大
    def __init__(self, input_dim):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = attn_weights.squeeze(-1)  # (batch_size, seq_len)

        attn_weights = F.softmax(attn_weights, dim=1)  # (batch_size, seq_len)

        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, input_dim)

        return pooled

class LearnablePositionalEncoding(nn.Module):                       #一个词的信息浓缩在一行，然后在低维的列，两个词越接近，位置信息数值越相似，越疏远，位置信息数值越不同；在高维，保持同样的规律，但是差异不会像低维那么明显
    def __init__(self, d_model, max_len=16):
        super(LearnablePositionalEncoding, self).__init__()

        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.d_model = d_model

        self.init_weights()

    def init_weights(self):
        position = torch.arange(0, self.positional_encoding.size(0)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.positional_encoding.size(1), 2) *     #让振荡频率随维度上升而下降
                             -(math.log(10000.0) / self.positional_encoding.size(1)))

        pe = torch.zeros_like(self.positional_encoding)         #创建临时positional_encoding pe矩阵

        pe[:, 0::2] = torch.sin(position * div_term)            #给pe矩阵赋值
        pe[:, 1::2] = torch.cos(position * div_term)

        self.positional_encoding.data.copy_(pe)                 #将临时矩阵pe的值拷贝给self_p_e矩阵

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 4:
            x = x.view(-1, original_shape[2], self.d_model)

        x = x + self.positional_encoding[:x.size(1), :]         #位置矩阵为了能够最大程度适配任何长度的序列，它会被初始化成最大的样子，但是实际句子长度不一定都满足最大值，
                                                                #所以需要裁剪（切割）位置矩阵，适配序列。x.size[1]就是序列长度，位置矩阵需要适配这个长度
        if len(original_shape) == 4:
                x = x.view(original_shape)

        return x

def check_shape(x, retreival):                                      #x：词嵌入；retrieval：是否为检索。对于MoRE而言，retrieval=False
    if not retreival:
        # B, L, D
        if len(x.shape) == 3:
            x = x
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        else:
            raise ValueError(f'Invalid shape: {x.shape}')
    else:
        # B, N, L, D
        if len(x.shape) == 4:
            x = x
        elif len(x.shape) == 3:
            x = x.unsqueeze(2)
        else:
            raise ValueError(f'Invalid shape: {x.shape}')
    return x