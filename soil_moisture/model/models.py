from typing import List, Tuple

import torch
import torch.nn as nn
from terratorch.models.pixel_wise_model import freeze_module
from lightning import LightningModule
from terratorch.models.model import ModelOutput
from terratorch.models.backbones.prithvi_mae import PatchEmbed, TemporalEncoder, LocationEncoder, _init_weights, get_3d_sincos_pos_embed
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block


class PatchEmbed(nn.Module):
    """3D version of timm.models.vision_transformer.PatchEmbed"""
    def __init__(
            self,
            input_size: Tuple[int, int, int] = (1, 224, 224),
            patch_size: Tuple[int, int, int] = (1, 16, 16),
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: nn.Module | None = None,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        assert self.grid_size >= [1,1,1], "Patch size is bigger than input size."
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.log_warning = True

    def forward(self, x):
        B, C, T, H, W = x.shape

        if (self.log_warning and
                (T / self.patch_size[0] % 1 or H / self.patch_size[1] % 1 or W / self.patch_size[2] % 1)):
            print(f"Input {x.shape[-3:]} is not divisible by patch size {self.patch_size}."
                           f"The border will be ignored, add backbone_padding for pixel-wise tasks.")
            self.log_warning = False

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x
        
class PrithviViT(nn.Module):
    """ Prithvi ViT Encoder"""
    def __init__(self,
                 img_size: int | Tuple[int, int] = 224,
                 patch_size: int | Tuple[int, int, int] = (1, 16, 16),
                 num_frames: int = 1,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 coords_encoding: List[str] | None = None,
                 coords_scale_learn: bool = False,
                 encoder_only: bool = True,  # needed for timm
                 ** kwargs,
                ):
        super().__init__()

        self.feature_info = []
        self.encoder_only = encoder_only
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)

        # 3D patch embedding
        self.patch_embed = PatchEmbed(
            input_size=(num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            assert patch_size[0] == 1, f"With temporal encoding, patch_size[0] must be 1, received {patch_size[0]}"
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Q. So this guy only get's used if the pos_embedding matches. Otherwise you use the 
        self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        # Transformer layers
        self.blocks = []
        for i in range(depth):
            self.blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
            self.feature_info.append(
                {"num_chs": embed_dim * self.patch_embed.grid_size[0], "reduction": 1, "module": f"blocks.{i}"}
            )
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, add_cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def random_masking(self, sequence, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.FloatTensor` of shape `(batch_size, sequence_length, dim)`)
            mask_ratio (float): mask ratio to use.
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def _get_pos_embed(self, x):
        t, h, w = x.shape[-3:]

        pos_embed = torch.from_numpy(get_3d_sincos_pos_embed(
            self.embed_dim,
            (
                t // self.patch_embed.patch_size[0],
                h // self.patch_embed.patch_size[1],
                w // self.patch_embed.patch_size[2],
            ),
            add_cls_token=True,
        )).float().unsqueeze(0).to(x)

        return pos_embed


    def forward(
        self, x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        mask_ratio=0.75
    ):
        if x.shape[-3:] != self.patch_embed.input_size:
            # changed input size
            # print('Input size of pos embedding mismatches. Generating a new one on-the-fly.')
            pos_embed = self._get_pos_embed(x)
        else:
            pos_embed = self.pos_embed

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding and location_coords is not None:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    # def forward_features(
    #     self,
    #     x: torch.Tensor,
    #     temporal_coords: None | torch.Tensor = None,
    #     location_coords: None | torch.Tensor = None,
    # ) -> list[torch.Tensor]:
    #     if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
    #         # add time dim
    #         x = x.unsqueeze(2)

    #     if x.shape[-3:] != self.patch_embed.input_size:
    #         pos_embed = self._get_pos_embed(x)
    #     else:
    #         pos_embed = self.pos_embed

    #     # embed patches
    #     x = self.patch_embed(x)

    #     # add pos embed w/o cls token
    #     x = x + pos_embed[:, 1:, :]

    #     if self.temporal_encoding and temporal_coords is not None:
    #         num_tokens_per_frame = x.shape[1] // self.num_frames
    #         temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
    #         x = x + temporal_encoding
    #     if self.location_encoding and location_coords is not None:
    #         location_encoding = self.location_embed_enc(location_coords)
    #         x = x + location_encoding

    #     # append cls token
    #     cls_token = self.cls_token + pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)

    #     # apply Transformer blocks
    #     out = []
    #     for block in self.blocks:
    #         x = block(x)
    #         out.append(x.clone())

    #     x = self.norm(x)
    #     out[-1] = x
    #     return out

    # def prepare_features_for_image_model(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
    #     out = []
    #     effective_time_dim = self.patch_embed.input_size[0] // self.patch_embed.patch_size[0]
    #     for x in features:
    #         x_no_token = x[:, 1:, :]
    #         number_of_tokens = x_no_token.shape[1]
    #         tokens_per_timestep = number_of_tokens // effective_time_dim
    #         h = int(np.sqrt(tokens_per_timestep))
    #         encoded = rearrange(
    #             x_no_token,
    #             "batch (t h w) e -> batch (t e) h w",
    #             e=self.embed_dim,
    #             t=effective_time_dim,
    #             h=h,
    #         )
    #         out.append(encoded)
    #     return out

class prithvi_terratorch(nn.Module):
    """
    Wrapper model around prithvi encoder.
    """

    def __init__(
        self,
        prithvi_weight,
        model_instance,
        use_TL_encoding=False,
    ):
        super(prithvi_terratorch, self).__init__()
        self.weights_path = prithvi_weight
        self.use_TL_encoding = use_TL_encoding
        self.prithvi_model = model_instance

        if prithvi_weight is not None:
            checkpoint = torch.load(self.weights_path)
            parsed_checkpoint = self.parse_weight(checkpoint)
            # NOTE: Need strict = False because pos_embed depends on data input size. 
            self.prithvi_model.load_state_dict(parsed_checkpoint, strict=False)

    def parse_weight(self, checkpoint: str):
        parsed_weight = {}
        for k, v in checkpoint.items():
            if (not self.use_TL_encoding) and ("embed_enc.scale" in k):
                continue

            # Only encoder weights are used.
            if "decoder" in k:
                continue

            # pos_embed is a fixed state that can be constructed on the fly and dependent on data input size.
            if "pos_embed" in k:
                continue

            parsed_k = k.removeprefix("encoder.")  # no op if encoder not present
            parsed_weight[parsed_k] = v

        return parsed_weight

    def freeze_encoder(self):

        freeze_module(self.prithvi_model)

    def forward(self, x, temp, loc, mask):

        latent, _, ids_restore = self.prithvi_model.forward(x, temp, loc, mask)

        return latent

class SimpleDecoder_comb_v2(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=64, n_tokens=10):
        super(SimpleDecoder_comb_v2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)# 1024 to 256; shape Nx1024 to Nx256
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.drp = nn.Dropout(p=drp_rate)
        # n_tokens = patches_per_frame * n_frame + 1 (cls token); e.g. 9*1+1=10, 9*4+1=37.
        self.hidden_dim_flattened=n_tokens*hidden_dim
        self.fc2=nn.Linear(self.hidden_dim_flattened, output_dim)
        #self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        #self.gelu = nn.GELU()

    def forward(self, x):
        x = self.relu(self.fc1(x))#shape 10x1024 to 10x256 ORG
        x = torch.reshape(x,(x.shape[0], x.shape[1]*x.shape[2]))#10x256 to 2560 
        x = self.fc2(x)  # 2560 to 64 Output shape 
        return x

# Define the 1D convolutional layers for the MERRA input, which is a temporal
# sequence of point values: [batch_size, n_vars, T_MERRA].
class Pt1dConvBranch(nn.Module):
    def __init__(self, n_vars=10, T_MERRA=1, kernel_size=3):
        super(Pt1dConvBranch, self).__init__()
        # kernel_size>1 lets the conv model temporal structure. padding='same' keeps the
        # time axis length fixed for any T_MERRA, including T_MERRA < kernel_size (the
        # kernel sees zero-padding at the edges), so layer shapes never depend on T_MERRA.
        # kernel_size = 1 # DEBUG

        self.T_MERRA = T_MERRA
        if False:
            seq_len = T_MERRA
            # self.conv1 = nn.Conv1d(n_vars, 32, kernel_size=)
            self.conv2 = nn.Conv1d(32, 16, kernel_size=kernel_size)
            self.conv3 = nn.Conv1d(16, 8, kernel_size=kernel_size)
        else:
            self.conv1 = nn.Conv1d(n_vars, 32, kernel_size=kernel_size, padding="same")
            self.conv2 = nn.Conv1d(32, 16, kernel_size=kernel_size, padding="same")
            self.conv3 = nn.Conv1d(16, 8, kernel_size=kernel_size, padding="same")
            self.pool = nn.AdaptiveAvgPool1d(1)
            
        # Collapse the (variable-length) temporal axis to a fixed-size summary so the
        # head input size is independent of T_MERRA. AdaptiveAvgPool1d(1) == mean over time.
        self.fc = nn.Linear(8, 64)  # Final output matches decoder output

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # [B, n_vars, T] -> [B, 32, T]
        x = torch.relu(self.conv2(x))  # [B, 32, T]    -> [B, 16, T]
        x = torch.relu(self.conv3(x))  # [B, 16, T]    -> [B, 8,  T]
        x = self.pool(x).squeeze(-1)   # [B, 8, T]      -> [B, 8]
        x = self.fc(x)                 # [B, 8]         -> [B, 64]
        return x

# Simple regression head: concatenate the active branch(es) and regress to a
# single scalar -- the soil-moisture (SM) target. `modality` selects which
# branches are built (hls | merra | both); each branch emits a 64-dim vector.
class RegressionModelSM(LightningModule):
    def __init__(self, prithvi_model, n_tokens=10, T_MERRA=1, modality="both"):
        super(RegressionModelSM, self).__init__()
        assert modality in ("hls", "merra", "both"), f"bad modality {modality}"
        self.modality = modality
        self.use_hls = modality in ("hls", "both")
        self.use_merra = modality in ("merra", "both")

        self.prithvi_model = None
        self.decoder = None
        self.pt1d_conv_branch = None

        feat_dim = 0
        if self.use_hls:
            self.prithvi_model = prithvi_model
            prithvi_emb_dim = self.prithvi_model.prithvi_model.embed_dim
            self.decoder = SimpleDecoder_comb_v2(
                prithvi_emb_dim, hidden_dim=256, output_dim=64, n_tokens=n_tokens
            )
            feat_dim += 64
        if self.use_merra:
            self.pt1d_conv_branch = Pt1dConvBranch(T_MERRA=T_MERRA)
            feat_dim += 64

        self.fc_final = nn.Linear(feat_dim, 1)  # Regression output

    def forward(self, im2d, pt1d=None, temporal_coords=None, location_coords=None, **kwargs):
        feats = []
        if self.use_hls:
            # The single-frame SM loader emits im2d as [B, C, H, W]; add a
            # singleton T axis for the 3D patch embed. temporal/location coords
            # are no-ops unless the backbone was built with coords_encoding.
            if im2d.dim() == 4:
                im2d = im2d.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
            pri_enc = self.prithvi_model(im2d, temporal_coords, location_coords, 0)
            feats.append(self.decoder(pri_enc))        # [B, 64]
        if self.use_merra:
            # The MERRA conv branch expects [B, n_vars, T]; the loader emits
            # pt1d as [B, n_vars], so add a singleton T=1 axis.
            if pt1d.dim() == 2:
                pt1d = pt1d.unsqueeze(-1)              # [B, n_vars] -> [B, n_vars, 1]
            feats.append(self.pt1d_conv_branch(pt1d))  # [B, 64]

        combined = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        output1 = self.fc_final(combined)              # [B, 1]
        return ModelOutput(output=output1)