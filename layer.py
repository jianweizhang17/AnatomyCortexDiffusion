import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange
from einops import repeat

import math

from aux_data.load_ico_order import *

################################################################################
#region Layers
################################################################################

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class ico_conv_layer(nn.Module):
    """
    A memory-efficient convolutional layer on an icosahedron-discretized sphere
    using a mini-batching approach to reduce memory consumption.

    Parameters:
        in_ch (int): num of input channels
        out_ch (int): num of output channels
        neigh_orders (np.ndarray): (N*7) numpy array of neighbor indices
        bias (bool): whether to use a bias term
    
    Input:
        B x in_feats x N tensor
    Return:
        B x out_feats x N tensor
    """
    def __init__(self, in_ch: int, out_ch: int,level_idx, bias: bool = True):
        super(ico_conv_layer, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # Convert the flattened numpy array to a PyTorch tensor once,
        # which is the most efficient method for GPU operations.
        self.neigh_orders = torch.from_numpy(level_data_dict[level_idx]['ico_order']).long()
        self.num_neighbors = self.neigh_orders.shape[1]
        
        self.weight = nn.Linear(self.num_neighbors * self.in_ch, self.out_ch)
        self.bias = bias
        if self.bias:
            self.bias_tensor = nn.Parameter(torch.zeros(1, out_ch, 1), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,2,1)
        b,n,_ = x.shape
        mat = x[:,self.neigh_orders,:]
        mat = mat.view(b,n, 7*self.in_ch)
        y = self.weight(mat).permute(0,2,1)
        return y
    
    
class ico_pool_layer(nn.Module):
    """
    The pooling layer on icosahedron discretized sphere using 1-ring filter
    
    Input: 
        B x D x N tensor
    Return:
        B x D x ((N+6)/4) tensor
    
    """  

    def __init__(self,level_idx,pooling_type='mean'):
        super(ico_pool_layer, self).__init__()
        self.neigh_orders = level_data_dict[level_idx]['ico_order']
        self.pooling_type = pooling_type
 
    def forward(self, x:torch.Tensor)->torch.Tensor:
        
        # _,_,n = x.shape
        # pooled_n = int((n+6)/4)
               
        # return x[:,:,:pooled_n]
        
        batch_num, feat_num, num_nodes = x.shape
        num_nodes = int((x.size()[2]+6)/4)
        feat_num = x.size()[1]
        x = x[:, :, self.neigh_orders[:num_nodes,:]].view(batch_num, feat_num, num_nodes, 7)
        if self.pooling_type == "mean":
            x = torch.mean(x, 3)
        if self.pooling_type == "max":
            x = torch.max(x, 3)
            assert(x[0].size() == torch.Size([batch_num, feat_num, num_nodes]))
            x = x[0]
        
        assert(x.size() == torch.Size([batch_num, feat_num, num_nodes]))
        return x
    
class ico_upconv_layer(nn.Module):
    """
    The transposed convolution layer on icosahedron discretized sphere using 1-ring filter
    
    Input: 
        B x in_feats x N, tensor
    Return:
        B x out_feats x ((Nx4)-6), tensor
    
    """  

    def __init__(self):
        super(ico_upconv_layer, self).__init__()
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        b,c,n = x.shape
        pad_n = int(n*4 - 6) - n
        return torch.cat((x,torch.zeros((b,c,pad_n),device=x.device)),dim=2)
    

class Attention(nn.Module):
    def __init__(self, dim,conv_layer,*conv_param,heads = 4, dim_head = 16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = conv_layer(dim, hidden_dim * 3,*conv_param)
        self.to_out = conv_layer(hidden_dim,dim,*conv_param)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)
        return out

#endregion

#region Blocks

class Block(nn.Module): 
    def __init__(self,in_ch,out_ch,conv_layer,*conv_params,norm_type='group',group_num=32):
        super().__init__()
        self.conv = conv_layer(in_ch,out_ch,*conv_params)
        if norm_type == 'group':
            self.norm = nn.GroupNorm(group_num,out_ch)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False)
        else:
            raise ValueError(f'Unknown norm type {norm_type}')
        self.act = nn.ReLU()

    def forward(self,x,scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale,shift = scale_shift
            x = x * (scale+1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,in_ch,out_ch,time_dim,conv_layer,*conv_params,norm_type='group',group_num=32):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(),nn.Linear(time_dim,out_ch*2))
        self.block1 = Block(in_ch,out_ch,conv_layer,*conv_params,norm_type=norm_type,group_num=group_num)
        self.block2 = Block(out_ch,out_ch,conv_layer,*conv_params,norm_type=norm_type,group_num=group_num)
        self.res_conv = conv_layer(in_ch,out_ch,*conv_params)

    def forward(self,x,time_emb=None):
        if time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)
        else:
            scale_shift = None
        h = self.block1(x,scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionEmbedding(nn.Module):
    def __init__(self,hidden_ch,in_type='num',cat_num=None):
        """Module for processing input conditions,which can be numberical or categorical data

        Args:
            hidden_ch (int): hidden size for mlp
            in_type (str, optional): num or cat. Defaults to 'num'.
            cat_num (_type_, optional): Total number of categories. Defaults to None.
        """
        super().__init__()
        
        # input can be either numeric or categorical labels
        assert in_type in ['num','cat'], f'Unknown input type {in_type}' 
        
        if in_type == 'num':
            self.embed = SinusoidalPosEmb(hidden_ch)
        elif in_type == 'cat':
            assert cat_num is not None, f'cat num not provided!'
            self.embed = nn.Embedding(cat_num, hidden_ch)
        
        self.mlp = nn.Sequential(
                        nn.Linear(hidden_ch,hidden_ch),
                        nn.GELU(),
                        nn.Linear(hidden_ch,hidden_ch))
    def forward(self,x):
        x_emb = self.embed(x)
        y = self.mlp(x_emb)
        return y
    
class EmbeddingApplication(nn.Module):
    def __init__(self,emb_dim,out_ch):
        super().__init__()
        self.emb_mlp = nn.Sequential(nn.SiLU(),nn.Linear(emb_dim,out_ch*2))
    def forward(self,x,emb):
        time_emb = self.emb_mlp(emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1')
        scale_shift = time_emb.chunk(2, dim = 1)
        scale,shift = scale_shift
        x = x * (scale+1) + shift
        return x
    
class EmbeddingApplicationSiT(nn.Module):
    def __init__(self,emb_dim,out_ch):
        super().__init__()
        self.emb_mlp = nn.Sequential(nn.SiLU(),nn.Linear(emb_dim,out_ch*2))
    def forward(self,x,emb):
        # x size: b n out_ch
        time_emb = self.emb_mlp(emb)
        scale_shift = time_emb.chunk(2, dim = 2)
        scale,shift = scale_shift
        # print ('x shape:',x.shape)
        # print ('scale shape:',scale.shape)
        # print ('shift shape:',shift.shape)
        x = x * (scale+1) + shift
        return x
#enderegion