import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn


kg=np.load('./traffic_data/bs_beijing.npz')['bs_kge']
adj_lists_train = np.load('citydata/dis_nor_beijing_train_960.npz')['res']
poi_list =  np.load('citydata/poi_nor_beijing_train_960.npz')['res']

src_full, dst_full = np.where(adj_lists_train > 0)
src_full_p, dst_full_p = np.where(poi_list > 0)
g_full_dis = dgl.graph((src_full, dst_full))
g_full_poi = dgl.graph((src_full_p, dst_full_p))


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def extract_frequency(x_ft,m=3):
    B, C, K, E = x_ft.shape
    q = 4
    temp_ft = x_ft.clone()
    sub_tensors = []
    b_idx = torch.arange(B)[:, None, None, None]
    c_idx = torch.arange(C)[None, :, None, None]
    k_idx = torch.arange(K)[None, None, :, None]
    for i in range(m):
        _, top_indices = torch.topk(abs(temp_ft), q, dim=3)
        result = torch.zeros_like(temp_ft)
        selected_values = torch.gather(temp_ft, 3, top_indices)
        result[b_idx, c_idx, k_idx, top_indices] = selected_values
        sub_tensors.append(result)
        temp_ft = temp_ft - result
    return sub_tensors,temp_ft

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe

class Frequency_position_Embedding(nn.Module):
    def __init__(self, d_model, max_len=120//2+1):
        super(Frequency_position_Embedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe
class STKDiff(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection_1 = nn.Conv1d(1, self.channels, 1)
        self.output_projection1 = nn.Linear(self.channels, self.channels//2, 1)
        self.output_projection2 = nn.Linear(self.channels//2, 1, 1)
        self.position_code = PositionalEmbedding(self.channels,max_len=168)


        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    config,
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    freq_select=config["freq_select"],
                )
                for _ in range(config["layers"])
            ]
        )



    def forward(self, x, diffusion_step, idx):
        B, K, L = x.shape
        x = x.unsqueeze(1).reshape(B,1,K*L)
        x = F.relu(self.input_projection_1(x))
        x =x.permute(0,2,1).reshape(B,K,L,-1)

        x_pe = self.position_code(x)
        x_pe = x_pe.unsqueeze(0).expand(B,-1,-1,-1)
        x = x.permute(0,3,1,2)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb,idx, x_pe)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.permute(0,2,3,1)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.tanh(x)
        x = self.output_projection2(x)  # b,k,l,1
        x = x.squeeze(-1)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, config, channels, diffusion_embedding_dim, nheads,freq_select):
        super().__init__()

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.first_res_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.first_res_projection2 = Conv1d_with_init(channels+ channels, channels, 1)
        self.double_res_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.channel= config['channels']
        self.pe_proj = Conv1d_with_init(channels, channels, 1)
        self.kg_proj = Conv1d_with_init(config['kg_emb'], channels, 1)
        self.bs_emb_projection = get_torch_trans(heads=nheads, layers=1, channels=self.channel).cuda()
        self.freq_cross = FourierCrossAttention(self.channel,self.channel)
        self.fre_select = freq_select
        self.freq_weight_project1 =nn.Sequential(
                nn.Linear(self.channel, self.channel),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.channel, self.channel)
        )
        self.freq_weight_project2 =nn.Sequential(
                nn.Linear(self.channel, self.channel),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.channel, self.channel)
        )

        self.layer1 = nn.Linear(self.channel, self.channel).cuda()
        self.layer2 = nn.Linear(self.channel, self.channel).cuda()
        self.layer3 = nn.Linear(self.channel, self.channel).cuda()
        self.layer4 = nn.Linear(self.channel, self.channel).cuda()
        self.layer5 = nn.Linear(2* self.channel, self.channel).cuda()

    def grapg_compute(self, g, feature):

        # num_nodes = g.num_nodes()
        g.ndata['h'] = feature

        message_func = fn.copy_u(u='h', out='m')

        reduce_func = fn.sum(msg='m', out='h_neigh')

        g.update_all(message_func, reduce_func)
        h = g.ndata['h_neigh']
        return h



    def forward(self, x, diffusion_emb, idx, x_pe):

        B, C, K, L = x.shape
        base_shape = x.shape
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)  # (B,channel,1,1)
        x_d = x+diffusion_emb#b,c,k,l

        bs_emb = kg[idx.cpu()]
        bs_emb_torch = torch.from_numpy(bs_emb).unsqueeze(-1).expand(B, -1 , K * L).to('cuda') #b,c_bs
        bs_emb_torch = self.kg_proj(bs_emb_torch).float()
        x_pe = self.pe_proj(x_pe.reshape(B,K*L,-1).permute(0,2,1))


        bs_embed_c = self.bs_emb_projection(bs_emb_torch.permute(2,0,1)).permute(1,2,0).reshape(B,-1,K,L)
        bs_emb_fre = torch.fft.rfft(bs_embed_c)#b,c/2,k,e
        x_ft_fre = torch.fft.rfft(x_d)#(B, C, K, E)
        _,_,_,E = x_ft_fre.shape

        x_fre, resid_freq = extract_frequency(x_ft_fre,self.fre_select)

        cross_att=[]
        for i in range(self.fre_select):
            cross_scores = self.freq_cross(x_fre[i], x_ft_fre, x_ft_fre, bs_emb_fre)
            cross_att.append(cross_scores)#b,k,e,c

        xf_abs = torch.abs(x_ft_fre)  # b.c.k.e
        x_weight = F.softmax(self.freq_weight_project1(xf_abs.permute(0,2,3,1)).permute(0,3,1,2),dim=-1)
        cross_sum = torch.sum(torch.stack(cross_att), dim=0).permute(0,3,1,2)
        x_per = torch.fft.irfft(cross_sum*x_weight)
        resid_time = torch.fft.irfft(resid_freq)#b,c,k,l

        yall = x_per+resid_time
        y_res = torch.cat((yall.reshape(B,C,-1),bs_embed_c.reshape(B,-1,K*L)), dim=1)
        y_res = self.first_res_projection2(y_res)

        sub = g_full_dis.subgraph(idx.to(torch.int64).cpu()).to('cuda')
        sub_poi = g_full_poi.subgraph(idx.to(torch.int64).cpu()).to('cuda')
        y_dis = self.grapg_compute(sub,y_res.permute(0,2,1)) + y_res.permute(0,2,1)
        yd = self.layer2(F.relu(self.layer1(y_dis)))
        y_poi = self.grapg_compute(sub_poi,y_res.permute(0,2,1))
        yp = self.layer4(F.relu(self.layer3(y_poi))) + y_res.permute(0,2,1)
        y_g_all = self.layer5(torch.cat((yd, yp), dim=-1)).permute(0,2,1) #(B,C,K*L)

        y_res = y_g_all + x_pe
        y = self.first_res_projection(y_res)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,2*channel,K*L)
        y = self.double_res_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation='softmax'):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.W_Q = nn.Linear(in_channels + in_channels , in_channels, dtype=torch.cfloat).cuda()
        self.W_K = nn.Linear(in_channels + in_channels , in_channels, dtype=torch.cfloat).cuda()
        self.W_V = nn.Linear(in_channels + in_channels , in_channels, dtype=torch.cfloat).cuda()
        # nn.Conv1d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)

    def forward(self, q, k, v, bs_emb):
        B, C, K, E = q.shape

        xq = self.W_Q(torch.cat((q, bs_emb), dim=1).reshape(B, -1, K * E).permute(0, 2, 1)).permute(0, 2, 1)
        xk = self.W_K(torch.cat((k, bs_emb), dim=1).reshape(B, -1, K * E).permute(0, 2, 1)).permute(0, 2, 1)
        xv = self.W_V(torch.cat((k, bs_emb), dim=1).reshape(B, -1, K * E).permute(0, 2, 1)).permute(0, 2, 1)

        xq_ft_ = xq.reshape(B,-1,K,E).permute(0, 2, 3, 1)

        xk_ft_ = xk.reshape(B,-1,K,E).permute(0, 2, 3, 1)
        xv_ft_ = xv.reshape(B,-1,K,E).permute(0, 2, 3, 1)

        xqk_ft = (torch.einsum("bkec,bkfc->bkef", xq_ft_, xk_ft_))/(E**0.5)

        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft_ = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft_ = torch.complex(xqk_ft_, torch.zeros_like(xqk_ft_))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bkxy,bkyz->bkxz", xqk_ft_, xv_ft_)#b,k,e,c

        return xqkv_ft

