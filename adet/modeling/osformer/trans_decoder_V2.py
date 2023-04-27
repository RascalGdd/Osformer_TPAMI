import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from adet.modeling.ops.modules.ms_deform_attn import MSDeformAttn
from .trans_utils import _get_clones, get_reference_points, with_pos_embed
from .feed_forward import get_ffn
import torch.nn.functional as F
import math

class CISTransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 ffn_type="default", num_feature_levels=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.ref_point_head = MLP(4 // 2 * d_model, d_model, d_model, 2)

        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward,
                                                dropout, ffn_type,
                                                num_feature_levels, nhead, enc_n_points)
        self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 4)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, tgts, pos_embeds, valid_masks, memorys=None, pos_memorys=None, reference_points=None):

        # prepare input for decoder
        tgt_flatten = []
        memory_flatten = []
        lvl_pos_embed_flatten = []
        lvl_pos_memory_flatten = []
        spatial_shapes = []
        # spatial_shape_grids = []
        # reference_points = []
        point_flatten = reference_points
        reference_points_sigmoid = reference_points.sigmoid().to("cuda")

        for lvl, (memory, pos_memory) in enumerate(zip(memorys, pos_memorys)):
            # B, N, C = tgt.shape
            # spatial_shape_tgt = N
            # spatial_shape_grids.append(spatial_shape_tgt)
            bs, c, h, w = memory.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # tgt = tgt.flatten(2).transpose(1, 2)
            memory = memory.flatten(2).transpose(1, 2)
            # pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_memory = pos_memory.flatten(2).transpose(1, 2)
            # lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # lvl_pos_embed_flatten.append(lvl_pos_embed)
            lvl_pos_memory = pos_memory + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_memory_flatten.append(lvl_pos_memory)
            # tgt_flatten.append(tgt)
            memory_flatten.append(memory)
            # print("shape of pos_embed is", pos_embed.shape)
            # print("shape of tgt is", tgt.shape)
            # reference_point = self.reference_points(pos_embed).sigmoid()
            # print("shape of reference_point is", reference_point.shape)
            # reference_points.append(reference_point)

        query_sine_embed = gen_sineembed_for_position(reference_points_sigmoid.transpose(0, 1))  # nq, bs, 256*2
        raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
        pos_scale = 1
        query_pos = pos_scale * raw_query_pos

        # print("query_sine_embed", query_sine_embed.shape)
        # print("raw_query_pos", raw_query_pos.shape)
        # asd



        # point_flatten = torch.cat(reference_points, 1)
        # tgt_flatten = torch.cat(tgt_flatten, 1)
        memory_flatten = torch.cat(memory_flatten, 1)
        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        lvl_pos_memory_flatten = torch.cat(lvl_pos_memory_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device="cuda")
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 维度为N*C时去掉prod操作
        # spatial_shape_grids = torch.as_tensor(spatial_shape_grids, dtype=torch.long, device=tgt_flatten.device)
        # level_start_index_grid = torch.cat((spatial_shape_grids.new_zeros((1, )), spatial_shape_grids.cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in valid_masks], 1)
        # we don't use these two grids
        spatial_shape_grids = 0
        level_start_index_grid = 0

        # decoder
        memory = self.decoder(tgts, memory_flatten, spatial_shapes, spatial_shape_grids, level_start_index_grid,
                              level_start_index, query_pos, lvl_pos_memory_flatten, point_flatten, valid_ratios)

        return memory


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, ffn_type="default",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = get_ffn(d_model, ffn_type)

    def forward(self, src, pos_embed, memorys, pos_memory, reference_points, spatial_shapes,
                level_start_index, spatial_shape_grids, level_start_index_grid):
        # self attention
        src2 = self.self_attn(with_pos_embed(src, pos_embed), reference_points,
                              with_pos_embed(memorys, pos_memory), spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.ffn(src, spatial_shape_grids, level_start_index_grid)

        return src


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, memorys, spatial_shapes, spatial_shape_grids,
                level_start_index_grid, level_start_index, pos_embed, pos_memory, reference_points, valid_ratios):
        output = src
        # batch_size = src.shape[0]
        # reference_points = get_reference_points(spatial_shape_grids, batch_size, device=src.device)
        for _, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(output, pos_embed, memorys, pos_memory, reference_points_input, spatial_shapes,
                           level_start_index, spatial_shape_grids, level_start_index_grid)

        return output


def build_transformer_decoder_v2(cfg):
    return CISTransformerDecoder(
        d_model=cfg.MODEL.OSFormer.HIDDEN_DIM,
        nhead=cfg.MODEL.OSFormer.NHEAD,
        num_encoder_layers=cfg.MODEL.OSFormer.DEC_LAYERS,
        dim_feedforward=cfg.MODEL.OSFormer.DIM_FEEDFORWARD,
        dropout=0.1,
        ffn_type=cfg.MODEL.OSFormer.FFN,
        num_feature_levels=len(cfg.MODEL.OSFormer.FEAT_INSTANCE_STRIDES),
        enc_n_points=cfg.MODEL.OSFormer.ENC_POINTS)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos