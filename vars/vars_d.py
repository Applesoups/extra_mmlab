#code for vars_d

import torch
from torch import nn
import torch.nn.functional as F
from mmcls.models.builder import BACKBONES
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks import DropPath, Dropout
from functools import partial
from einops import rearrange

def fft_conv_grouped(x, weight, image_size):
    # weight should be complex!
    # weight: group * int_channel_per_group * in_channel_per_group * img_size * img_size//2+1
    group, int_c, in_c, _, __ = weight.shape
    B, C, N = x.shape
    x = x.view(B, group, 1, in_c, image_size, image_size)
    weight = weight.view(1, group, int_c, in_c, _, __)

    x = x.to(torch.float32)

    x = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
    x = (x * weight).sum(dim=3)
    x = torch.fft.irfftn(x, s=(image_size, image_size), dim=(-2, -1), norm='ortho')
    x = x.view(B, group*int_c, N)

    return x

def to_random_feature(x, m, proj):
    x = torch.exp(x @ proj - x.norm(dim=-1, keepdim=True)**2 / 2) / m**0.5

    return x

class Mlp(BaseModule):
    def __init__(self, 
                 in_features, 
                 hidden_features=None,
                 output_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        out_features = output_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.drop = Dropout(drop)
    def forward(self, x):

        B,N,C = x.shape
        x = x.reshape(B, int(N**0.5), int(N**0.5), C).permute(0,3,1,2).contiguous()
        x = self.bn1(self.fc1(x))
        x = self.act(x)
        x = self.drop(x)
        x = self.act(self.bn2(self.dwconv(x)))
        x = self.bn3(self.fc2(x))
        x = self.drop(x)
        x = x.permute(0,2,3,1).contiguous().reshape(B, -1, C)
        return x

class VARS_D(BaseModule):
    def __init__(self,
                 dim,
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 rand_feat_dim_ratio=2, 
                 lam=0.3,
                 num_step=5,
                 use_mask=False,
                 init_cfg=None):
        super(VARS_D, self).__init__(init_cfg)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.rand_feat_dim = head_dim * rand_feat_dim_ratio
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        #print('dim:',dim, dim*2)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)

        self.use_mask = use_mask
        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))

        self.rand_matrix = nn.Parameter(torch.randn((num_heads, head_dim, self.rand_feat_dim)), requires_grad=False)

        self.relu = nn.ReLU(inplace=False)
        self.L = torch.nn.parameter.Parameter(torch.ones(self.num_heads) * 25, requires_grad=False)
        self.if_update_L = 0
        self.lam = lam
        self.num_step = num_step

    def att_func(self, x, input, kk):
        lam = self.lam

        x = torch.sign(input) * self.relu(input.abs() - lam)
        L = kk.norm(p=1, dim=-1).max(dim=-1)[0].detach()[..., None, None] + 1

        for k in range(self.num_step):
            x = x - (kk @ x - input) / L
            x = torch.sign(x) * self.relu(x.abs() - lam / L)

        return x, lam

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        #print("qkv shape:",self.qkv(x).shape)
        q, _, v = qkv[0], qkv[0], qkv[1]   # B * num_heads * N * dim

        q = to_random_feature(q * self.scale ** 0.5, self.rand_feat_dim, self.rand_matrix)  # unnormalized!!
        q = F.normalize(q, p=2, dim=-2)
        k = q.clone()

        kk = q.transpose(-2, -1).contiguous() @ k

        x = v
        x = k.transpose(-2, -1).contiguous() @ x  # B * num_heads * m * dim

        input = x
        sparse_recon, lam = self.att_func(x, input, kk)

        x = q @ sparse_recon

        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def update_L(self, kk, lam=0.1):
        return

class VARS_D_Block(BaseModule):

    def __init__(self, 
                 dim, 
                 image_size, 
                 num_heads, 
                 drop_path,
                 use_mask=False,
                 act_layer='Gelu',
                 norm_layer='Partial',    
                 proj_drop=.0,          
                 mlp_ratio=4.,   
                 attn_cfgs=dict(),
                 init_cfg=None):
        super(VARS_D_Block, self).__init__(init_cfg)

        #only one choice now
        if act_layer == 'Gelu':
            self.act_layer = nn.GELU
        else:
            self.act_layer = nn.GELU
        if norm_layer == 'Partial':
            self.norm_layer=partial(nn.LayerNorm, eps=1e-6)
        else:
            self.norm_layer=partial(nn.LayerNorm, eps=1e-6)

        self.norm1 = self.norm_layer(dim)

        vars_cfgs={
            'dim':dim,
            'num_heads':num_heads,
            'use_mask': use_mask,
            'proj_drop': proj_drop,
            **attn_cfgs
        }
        self.attn = VARS_D(**vars_cfgs)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = self.norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def update_L(self, lam=0.1):
        self.attn.update_L(lam)


class VarsTransformer(BaseModule):
    def __init__(self,
                 base_dim, 
                 depth, 
                 heads, 
                 attention_type='vars_d',
                 drop_rate=.0, 
                 drop_path_prob=None, 
                 rand_feat_dim_ratio=2, 
                 use_mask=False, 
                 masked_block=None,
                 feature_map_size=0,
                 proj_drop=.0,
                 mlp_ratio=4.,
                 attn_cfgs=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(VarsTransformer, self).__init__(init_cfg)
        self.layers = ModuleList()
        self.depth = depth
        embed_dim = base_dim * heads

        attentions={
            # 'self-att': Attention,
            # 'perf': Performer_Block,
            # 'vars_s': VARS_S_Block,
            'vars_d': VARS_D_Block,
            # 'vars_sd': VARS_SD_Block
        }

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            if_use_mask = True if (use_mask and masked_block != None and i < masked_block) else False
            trans_cfgs = {
                'dim': embed_dim,
                'num_heads': heads,
                'drop_path': drop_path_prob[i],
                'use_mask': if_use_mask,
                'image_size': feature_map_size,
                'attn_cfgs': attn_cfgs,
                'proj_drop': proj_drop,
                'mlp_ratio': mlp_ratio,
                **layer_cfgs,
            }
            self.blocks.append(attentions[attention_type](**trans_cfgs))

    def forward(self, x):
        B ,C ,H ,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        for i in range(self.depth):
            x = self.blocks[i](x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        #print(x)
        return x


class conv_embedding(BaseModule):
    # 函数中参数除in_channels 和 out_channels 被全部写死
    def __init__(self,
                 in_channels,
                 out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=(4, 4), stride=(4, 4))
        )
    
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        return x

class conv_head_pooling(BaseModule):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):

        x = self.conv(x)

        return x

@BACKBONES.register_module()
class PoolingTransformer(BaseModule):
    """Robust vision transformer with vars

    Args:
        image_size (int): The expected input image shape. 
        patch_size (int): The patch size in patch embedding.
        base_dims (int): Numbers of input channels for transformer blocks
        heads (int): Number of attention heads.
        in_chans (int): Number of input channels. Default to 3.
        attention_type (str): Select the attention type used in RVT. 
            Choose from 'self_att', 'perf', 'vars_s', 'vars_d', 'vars_sd'.
        drop_rate (float): Probability of an element to be zeroed. 
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection in attention. Defaults to 0.
        mlp_ratio (int): The ratio for hidden layer in mlp after attention block.
            Defaults to 4.
        attn_cfgs (dict): The config for args used in attention block.
            Including: 
            attn_drop (float): Dropout rate of the attention block.
                Defaults to 0.
            rand_feat_dim_ratio (int): The dimension of random feature in the approximation to attention blocks.
                Defaults to 2.
            lam (float): Strength of sparse regularization. Defaults to 0.3.
            num_step (int): How many steps taken to optimize sparse reconstruction.
                Defaults to 5.
            qkv_bias (bool): Enable bias for qkv if True. Defaults to True.
            qk_scale (float): Scale factor for qk. Default to head_dim ** -0.5.
        layer_cfgs (dict): The config for setting norm_layers and act_layers in attention blocks.
            including: 
            act_layer (str): Choose Active layer type. Choosing from 'Gelu' only (temporally).
                Defaults to 'Gelu'.
            norm_layer (str): Choose Normalized layer type. Choosinf from 'Partial' only (temporally).
                Defaults to 'Partial'.
        masked_cfgs (dict): The configs for setting args related to mask operation.
            including: 
            use_mask (bool): Whether to use mask in attention blocks.
            mask_blocks (int): Num of attention blocks using mask.


        
    """

    def __init__(self, 
                 image_size, 
                 patch_size, 
                 base_dims, 
                 depth, 
                 heads,
                 in_chans=3,
                 attention_type='vars_d',
                 drop_rate=.0,  
                 drop_path_rate=.0,
                 proj_drop=.0,
                 mlp_ratio=4,
                 attn_cfgs=dict(attn_drop=.0, 
                            rand_feat_dim_ratio=2, 
                            lam=0.3,
                            num_step=5,
                            qkv_bias=True,
                            qk_scale=None,
                            ),
                 layer_cfgs=dict(
                            act_layer='Gelu',
                            norm_layer='Partial',                     
                 ),
                 masked_cfgs=dict(use_mask=False, 
                                masked_block=None),
                init_cfg=None
                 ):
        super(PoolingTransformer, self).__init__(init_cfg)  

        total_block = sum(depth)
        # set padding as 0
        padding = 0
        block_idx = 0

        self.base_dims = base_dims
        self.heads = heads

        self.patch_size = patch_size
        embed_cfgs = {
            'in_channels': in_chans,
            'out_channels': base_dims[0] * heads[0]
        }
        self.patch_embed = conv_embedding(**embed_cfgs)

        self.pos_drop = Dropout(drop_rate)
        _masked_cfgs=[{'use_mask': False} for _ in range(len(depth))]
        _masked_cfgs[0]= masked_cfgs

        self.pos_drop = Dropout(drop_rate)

        self.transformers = ModuleList()
        self.pools = ModuleList()


        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            trans_cfgs={
                'base_dim': base_dims[stage],
                'depth': depth[stage],
                'heads': heads[stage],
                'drop_rate': drop_rate,
                'drop_path_prob': drop_path_prob,
                'feature_map_size': image_size // (patch_size * 2** stage),
                'attention_type': attention_type,
                'proj_drop': proj_drop,
                'mlp_ratio': mlp_ratio,
                'attn_cfgs':attn_cfgs,
                'layer_cfgs':layer_cfgs,
                **_masked_cfgs[stage]
            }

            self.transformers.append(
                VarsTransformer(**trans_cfgs)
            )

            if stage < len(heads) -1:
                convhead_cfgs={
                    'in_feature': base_dims[stage] * heads[stage], 
                    'out_feature': base_dims[stage + 1] * heads[stage + 1],
                    'stride': 2
                }
                self.pools.append(
                    conv_head_pooling(**convhead_cfgs)
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x):
        x=x.float()
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)
        cls_features = self.norm(self.gap(x).squeeze())
        return cls_features

