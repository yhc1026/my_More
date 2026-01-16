import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from einops import rearrange

from .my_submodule import LearnablePositionalEncoding, check_shape, AttentivePooling


class ModalityFFN(nn.Module):                               # 通过FFN，使得不同模态的特征统一到统一格式
    def __init__(self, hid_dim):
        super(ModalityFFN, self).__init__()
        self.modailty_ffn = nn.Sequential(nn.LazyLinear(hid_dim), nn.ReLU(), nn.LazyLinear(hid_dim))

    def forward(self, x, pos_x, neg_x):
        x = self.modailty_ffn(x)
        pos_x = self.modailty_ffn(pos_x)
        neg_x = self.modailty_ffn(neg_x)

        return x, pos_x, neg_x


class ModalityExpert(nn.Module):                    # 专家类，后文会用这个专家类创造出不同模态的专家
    def __init__(self, hid_dim, dropout, num_head, alpha, ablation):
        super(ModalityExpert, self).__init__()
        self.ffn = nn.Sequential(nn.LazyLinear(hid_dim), nn.ReLU(), nn.LazyLinear(hid_dim))
        self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)     # 创建正样本注意力层
        self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)     # 创建负样本注意力层
        self.alpha = alpha
        self.ablation = ablation

    def forward(self, query, pos, neg):
        # 首先处理 query 的维度
        original_query_shape = query.shape

        # 如果 query 是4维（视觉特征），需要先处理
        if len(query.shape) == 4:
            # [b, 1, seq_len, d] -> [b, seq_len, d]
            query = query.squeeze(1)  # 或者 query = query.view(query.shape[0], -1, query.shape[-1])

        query = self.ffn(query)
        pos = self.ffn(pos)
        neg = self.ffn(neg)

        # 处理 query 的维度：确保是2D [b, d] 或 3D [b, seq_len, d]
        if len(query.shape) == 3:
            # 如果是3D，取平均值或使用第一个时间步
            query = query.mean(dim=1)  # [b, seq_len, d] -> [b, d]
            # 或者 query = query[:, 0, :]  # 取第一个时间步

        # 修复：添加序列长度维度
        if len(pos.shape) == 3:
            pos = pos.unsqueeze(2)  # [b, n, 1, d]
        if len(neg.shape) == 3:
            neg = neg.unsqueeze(2)  # [b, n, 1, d]

        # 现在形状是 [b, n, 1, d]，可以正常 rearrange
        pos = rearrange(pos, 'b n l d -> b (n l) d')  # [b, n, d]
        neg = rearrange(neg, 'b n l d -> b (n l) d')  # [b, n, d]

        # 修复注意力机制维度问题
        batch_size = query.shape[0]
        num_pos = pos.shape[1]

        # 将 query 扩展到与 key/value 相同的序列长度
        query_expanded = query.unsqueeze(1).expand(-1, num_pos, -1)  # [b, n, d]

        # 使用扩展的query进行注意力计算
        pos_attn, _ = self.pos_attn(query_expanded, pos, pos)  # [b, n, d]
        neg_attn, _ = self.neg_attn(query_expanded, neg, neg)  # [b, n, d]

        # 对注意力输出进行池化，得到单个向量
        pos_attn = pos_attn.mean(dim=1)  # [b, d]
        neg_attn = neg_attn.mean(dim=1)  # [b, d]

        ret = self.alpha * pos_attn + (1 - self.alpha) * neg_attn + query

        return ret


class MoRE(nn.Module):
    def __init__(self, text_encoder, fea_dim=768, dropout=0.2, num_head=8, alpha=0.5, delta=0.25, num_epoch=20,
                 ablation='No', loss='No', **kargs):
        super(MoRE, self).__init__()

        local_bert_path = r"D:\models\bert\bert-base-uncased"
        self.bert = AutoModel.from_pretrained(local_bert_path).requires_grad_(False)

        self.text_ffn = nn.LazyLinear(fea_dim)
        self.vision_ffn = nn.LazyLinear(fea_dim)
        self.audio_ffn = nn.LazyLinear(fea_dim)

        self.text_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)
        self.vision_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)
        self.audio_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)

        self.positional_encoding = LearnablePositionalEncoding(768, 16)

        self.text_pre_router = nn.LazyLinear(fea_dim)
        self.vision_pre_router = nn.LazyLinear(fea_dim)
        self.audio_pre_router = nn.LazyLinear(fea_dim)

        self.router = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(3), nn.Softmax(dim=-1))            #路由器：一个小型网络，专门用于优化生成权重分配的参数

        self.classifier = nn.Sequential(nn.LazyLinear(200), nn.ReLU(), nn.Dropout(dropout), nn.Linear(200, 2))          #三模态融合特征预测

        self.text_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))                         #三模态分别预测
        self.vision_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))                       #三模态分别预测
        self.audio_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))                        #三模态分别预测

        self.text_pooler = AttentivePooling(fea_dim)
        self.vision_pooler = AttentivePooling(fea_dim)

        self.delta = delta                                  #训练早期，重视三模态融合特征训练；到后期，重视三模态分别特征训练
        self.total_epoch = num_epoch
        self.ablation = ablation
        self.loss = loss

    def forward(self, **inputs):
        text_fea = inputs['text_fea']
        audio_fea = inputs['audio_fea']
        vision_fea = inputs['vision_fea']
        text_sim_pos_fea = inputs['text_sim_pos_fea']
        text_sim_neg_fea = inputs['text_sim_neg_fea']
        frame_sim_pos_fea = inputs['vision_sim_pos_fea']
        frame_sim_neg_fea = inputs['vision_sim_neg_fea']
        mfcc_sim_pos_fea = inputs['audio_sim_pos_fea']
        mfcc_sim_neg_fea = inputs['audio_sim_neg_fea']

        vision_fea = self.positional_encoding(vision_fea)
        frame_sim_pos_fea = self.positional_encoding(frame_sim_pos_fea)
        frame_sim_neg_fea = self.positional_encoding(frame_sim_neg_fea)

        text_fea_aug = self.text_expert(text_fea, text_sim_pos_fea, text_sim_neg_fea)
        vision_fea_aug = self.vision_expert(vision_fea, frame_sim_pos_fea, frame_sim_neg_fea)
        audio_fea_aug = self.audio_expert(audio_fea, mfcc_sim_pos_fea, mfcc_sim_neg_fea)

        # 修复：在concat之前统一所有特征的维度为2D
        # 处理原始特征用于router
        text_fea_router = text_fea.mean(dim=1) if len(text_fea.shape) > 2 else text_fea
        vision_fea_router = vision_fea.mean(dim=1) if len(vision_fea.shape) > 2 else vision_fea
        audio_fea_router = audio_fea.mean(dim=1) if len(audio_fea.shape) > 2 else audio_fea

        # 确保所有router特征都是2D [batch_size, feature_dim]
        router_fea = torch.cat([text_fea_router, vision_fea_router, audio_fea_router], dim=-1)
        weight = self.router(router_fea).squeeze(1)

        # 修复：对增强后的特征进行池化（统一为2D）
        text_fea_aug = text_fea_aug.mean(dim=1) if len(text_fea_aug.shape) > 2 else text_fea_aug
        vision_fea_aug = vision_fea_aug.mean(dim=1) if len(vision_fea_aug.shape) > 2 else vision_fea_aug
        audio_fea_aug = audio_fea_aug.mean(dim=1) if len(audio_fea_aug.shape) > 2 else audio_fea_aug

        text_pred = self.text_preditor(text_fea_aug)
        vision_pred = self.vision_preditor(vision_fea_aug)
        audio_pred = self.audio_preditor(audio_fea_aug)

        if self.ablation == 'w/o-router':
            fea = (text_fea_aug + vision_fea_aug + audio_fea_aug) / 3
        else:
            fea = (text_fea_aug * weight[:, 0].unsqueeze(1) +
                   vision_fea_aug * weight[:, 1].unsqueeze(1) +
                   audio_fea_aug * weight[:, 2].unsqueeze(1))

        output = self.classifier(fea)

        return {
            'pred': output,
            'text_pred': text_pred,
            'vision_pred': vision_pred,
            'audio_pred': audio_pred,
            'weight': weight,
        }

    def calculate_loss(self, **inputs):
        delta = self.delta
        total_epoch = self.total_epoch

        pred = inputs['pred']
        label = inputs['label']
        text_pred = inputs['text_pred']
        vision_pred = inputs['vision_pred']
        audio_pred = inputs['audio_pred']
        cur_epoch = inputs['epoch']

        f_epo = (float(cur_epoch) / float(total_epoch)) ** 2

        l_mix = F.cross_entropy(pred, label)
        if text_pred is not None:
            text_loss = F.cross_entropy(text_pred, label)
        else:
            text_loss = 0.0
        if vision_pred is not None:
            vision_loss = F.cross_entropy(vision_pred, label)
        else:
            vision_loss = 0.0
        if audio_pred is not None:
            audio_loss = F.cross_entropy(audio_pred, label)
        else:
            audio_loss = 0.0

        l_exp = (text_loss + vision_loss + audio_loss) / 3

        l_join = min(1 - f_epo, delta) * l_exp + max(f_epo, 1 - delta) * l_mix

        return l_join, l_mix