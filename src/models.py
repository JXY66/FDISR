import math
import os
import pickle
from tqdm import tqdm
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
import faiss
import time
from modules import Encoder, LayerNorm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np





# 定义 1D UNet 模型
class UNet1D(nn.Module):
    def __init__(self, emb_dim, seq_len, num_steps=100, beta_start=1e-4, beta_end=0.02):
        super(UNet1D, self).__init__()

        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.num_steps = num_steps
        
        # 定义正向扩散过程的 beta 序列
        self.beta = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, emb_dim, kernel_size=3, padding=1)
        )
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )
        
    def forward_diffusion1(self, x_0, t):
        """
        正向扩散过程：向输入数据中添加噪声
        x_0: 原始输入 [batchsize, seq_len, emb_dim]
        t: 时间步 [batchsize]，每个样本的时间步
        """
        batch_size = x_0.shape[0]
        noise = torch.randn_like(x_0).to(x_0.device)  # 随机噪声
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1).to(x_0.device)  # shape [batchsize, 1, 1]
        
        # 添加噪声
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise


    def forward_diffusion(self, x_0, t):
        """
        正向扩散过程：向输入数据中添加噪声
        x_0: 原始输入 [batchsize, seq_len, emb_dim]
        t: 时间步 [batchsize]，每个样本的时间步
        """
        batch_size = x_0.shape[0]
        noise = torch.randn_like(x_0).to(x_0.device)  # 随机噪声
        
        # 确保 t 和 self.alpha_bar 在同一设备上
        t = t.to(self.alpha_bar.device)
        
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1).to(x_0.device)  # shape [batchsize, 1, 1]
        
        # 计算扩散后的结果
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t, noise




    def reverse_denoise(self, x_t, t):
        """
        反向去噪过程：通过 UNet 模型去噪
        x_t: 带噪声的输入 [batchsize, seq_len, emb_dim]
        t: 时间步 [batchsize]，每个样本的时间步
        """
        # 将时间步嵌入到序列中
        time_embedding = self.time_mlp(self.time_embedding(t, self.emb_dim)).unsqueeze(1)  # [batchsize, 1, emb_dim]
        x_t_with_time = x_t + time_embedding  # 将时间步信息加入到序列中
        
        # 编码器：将输入转换为特征表示
        x_t_with_time = x_t_with_time.permute(0, 2, 1)  # 转换为 [batchsize, emb_dim, seq_len]
        encoded = self.encoder(x_t_with_time)
        
        # 解码器：还原序列
        decoded = self.decoder(encoded)
        decoded = decoded.permute(0, 2, 1)  # 转换回 [batchsize, seq_len, emb_dim]
        
        return decoded
    
    def time_embedding(self, t, emb_dim):
        """
        时间步嵌入：将时间步映射到高维空间
        t: 时间步 [batchsize]
        emb_dim: 嵌入维度
        """
        half_dim = emb_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, x_0, t):
        """
        完整的扩散模型过程：正向扩散 + 反向去噪
        x_0: 原始输入 [batchsize, seq_len, emb_dim]
        t: 时间步 [batchsize]
        """
        # 正向扩散过程
        t = t.to(x_0.device)
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # 反向去噪过程
        x_denoised = self.reverse_denoise(x_t, t)
        
        return x_denoised, noise







class EnhancedTemporalContrastiveLearning(nn.Module):
    def __init__(self, emb_dim, seq_len, temperature=0.07, hard_negative_weight=0.1):
        super(EnhancedTemporalContrastiveLearning, self).__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.hard_negative_weight = hard_negative_weight
        
        self.temporal_projection = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, emb_dim))
        
    def forward(self, x1, x2):

        # 基于序列的 对比学习，所以设置了一个可学习的参数
        batch_size = x1.size(0)
        x1 = x1 + self.positional_encoding  # 加上位置编码
        x2 = x2 + self.positional_encoding
        
        z1 = self.temporal_projection(x1)
        z2 = self.temporal_projection(x2)
        
        z1 = F.normalize(z1.view(batch_size, -1), dim=-1)
        z2 = F.normalize(z2.view(batch_size, -1), dim=-1)
        
        sim_matrix = torch.matmul(z1, z2.t())
        sim_matrix /= self.temperature  # 温度系数
        
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # Hard negative
        hard_negative_mask = self.get_hard_negative_mask(sim_matrix, labels)
        
        # 普通对比损失 + hard-sample 对比损失
        standard_loss = F.cross_entropy(sim_matrix, labels)
        hard_negative_loss = F.cross_entropy(sim_matrix + hard_negative_mask, labels)
        
        total_loss = (1 - self.hard_negative_weight) * standard_loss + self.hard_negative_weight * hard_negative_loss
        
        return total_loss
    
    def get_hard_negative_mask(self, sim_matrix, labels):
        with torch.no_grad():
            negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
            sim_matrix = sim_matrix.masked_fill(~negative_mask, float('-inf'))
            hard_negatives = sim_matrix.max(dim=1)[1]
            hard_negative_mask = torch.zeros_like(sim_matrix)
            hard_negative_mask.scatter_(1, hard_negatives.unsqueeze(1), float('-inf'))
        return hard_negative_mask








class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]





class MultiHeadIntentExtractor(nn.Module):
    def __init__(self, dim, intent_dim, num_intents, num_heads=4):
        super(MultiHeadIntentExtractor, self).__init__()
        self.num_intents = num_intents
        self.intent_dim = intent_dim

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.intent_linear = nn.Linear(dim, intent_dim)
        self.intent_queries = nn.Parameter(torch.randn(num_intents, dim))

    def forward(self, x):
        """
        参数 session embs [batch_size, seq_len, dim]
        返回值 intent [batch_size, num_intents, intent_dim]
        """
        batch_size = x.size(0)

        intent_queries = self.intent_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # multihead to extract intent
        intent_context, _ = self.multihead_attn(intent_queries, x, x)
        intent_embeddings = self.intent_linear(intent_context)  # [batch_size, num_intents, intent_dim]

        return intent_embeddings



class DynamicIntentAttentionAggregator(nn.Module):
    def __init__(self, intent_dim, intent_num):
        super().__init__()
        
        # 多尺度权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Linear(intent_dim, intent_dim * 2),
            nn.GELU(),
            nn.LayerNorm(intent_dim * 2),
            nn.Linear(intent_dim * 2, intent_num),
            nn.Softmax(dim=-1)
        )
        
        # 注意力增强机制
        self.attention_enhancer = nn.MultiheadAttention(
            embed_dim=intent_dim, 
            num_heads=4, 
            batch_first=True
        )
    
    def forward(self, intent_vectors):
        # intent_vectors: [B, intent_num, intent_dim]
        
        # 自适应权重生成
        adaptive_weights = self.weight_generator(
            torch.mean(intent_vectors, dim=1)
        )  # [B, intent_num]
        
        # 注意力增强
        enhanced_vectors, _ = self.attention_enhancer(
            intent_vectors, 
            intent_vectors, 
            intent_vectors
        )  # [B, intent_num, intent_dim]
        
        # 加权聚合
        aggregated_repr = torch.einsum(
            'bni,bn->bi', 
            enhanced_vectors, 
            adaptive_weights
        )  # [B, intent_dim]
        
        return aggregated_repr




class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()

        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        
        # 多专家：创建多个 Encoder 实例
        self.num_experts = args.num_experts
        self.experts = nn.ModuleList([Encoder(args) for _ in range(self.num_experts)])
        
        # 路由器：一个简单的全连接层，用于生成专家的权重
        self.router = nn.Linear(args.hidden_size, self.num_experts)

        # 扩散模型
        self.diff_model = UNet1D(emb_dim=args.hidden_size, seq_len=args.max_seq_length, num_steps=500)

        # 对比学习
        self.contrast = EnhancedTemporalContrastiveLearning(emb_dim=args.hidden_size, seq_len=args.max_seq_length)

        # 意图建模
        self.intent_num = 4
        self.user_intent = MultiHeadIntentExtractor(dim=self.args.hidden_size, intent_dim=self.args.hidden_size, num_intents=self.intent_num, num_heads=4)

        # 意图的聚合
        self.intent_agg = DynamicIntentAttentionAggregator(intent_dim=self.args.hidden_size, intent_num=self.intent_num)
        
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb




    def intent_contrastive_loss(self, vectors, temperature=0.5):
        """
            contrastive_loss

            vectors: [batchsize, K, dim] 
            temperature: 
   
        """
        batchsize, K, dim = vectors.shape
        
        # cos 相似度
        vectors_normalized = F.normalize(vectors, p=2, dim=2)
        similarity_matrix = torch.matmul(vectors_normalized, vectors_normalized.transpose(1, 2))
        
        # temperature
        similarity_matrix /= temperature
        
        # 对角线
        labels = torch.eye(K).unsqueeze(0).repeat(batchsize, 1, 1).to(vectors.device)
        
        # ce loss
        loss = F.cross_entropy(similarity_matrix.view(-1, K), labels.view(-1, K))
        
        return loss


    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 获取序列嵌入
        sequence_emb = self.add_position_embedding(input_ids)

        # 扩散模型
        batch_size = input_ids.shape[0]
        timesteps = torch.randint(0, 500, (batch_size,), device=sequence_emb.device)
        sequence_emb_denoised, noise = self.diff_model(sequence_emb, timesteps)


        # 扩散模型的对比学习
        diff_cl_loss = self.contrast(sequence_emb, sequence_emb_denoised)


        # 意图建模
        user_intent_embeddings = self.user_intent(sequence_emb_denoised) # [batchsize, intent_num, intent_dim]


        # 意图解纠缠
        intent_disentangle_loss = self.intent_contrastive_loss(user_intent_embeddings)


        # 意图的自适应加权聚合
        aggregated_intent = self.intent_agg(user_intent_embeddings)
        aggregated_intent = aggregated_intent.unsqueeze(1)
        sequence_emb = sequence_emb + aggregated_intent  # 将意图加到序列中



        # 路由器：根据输入序列生成专家的权重
        router_logits = self.router(sequence_emb)  # (batch_size, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (batch_size, seq_len, num_experts)

        # 多专家：每个专家处理输入序列
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
            expert_outputs.append(expert_output[-1])  # 只取最后一层的输出

        # 将多个专家的输出按权重加权
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (batch_size, seq_len, hidden_size, num_experts)
        sequence_output = torch.einsum('bshn,bse->bsh', expert_outputs, router_probs)  # 加权求和

        return sequence_output, diff_cl_loss + intent_disentangle_loss




    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

