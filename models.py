import torch
import torch.nn as nn
from terratorch.models.pixel_wise_model import freeze_module
from lightning import LightningModule
from terratorch.models.model import ModelOutput


class prithvi_terratorch(nn.Module):
    """
    Wrapper model around prithvi encoder.
    """

    def __init__(
        self,
        prithvi_weight,
        model_instance,
        manually_parse_weights=True,
        use_TL_encoding=False,
    ):

        super(prithvi_terratorch, self).__init__()

        # load checkpoint for Prithvi_global

        self.weights_path = prithvi_weight
        self.checkpoint = torch.load(self.weights_path)
        self.use_TL_encoding = use_TL_encoding

        self.prithvi_model = model_instance

        if manually_parse_weights:
            parsed_checkpoint = self.parse_weight(self.checkpoint)
            # NOTE: strict = False because pos_embed should be constructed on the fly and not throw an error when loading it.
            self.prithvi_model.load_state_dict(parsed_checkpoint, strict=False)
        else:
            self.prithvi_model.load_state_dict(self.checkpoint, strict=False)

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

# Define the regression model --simple regression to concatenate prithvi merra and regress to gpp lfux
class RegressionModel_flux(LightningModule):
    def __init__(self, prithvi_model, n_tokens=10, T_MERRA=1):
        super(RegressionModel_flux, self).__init__()
        self.prithvi_model = prithvi_model
        prithvi_emb_dim = self.prithvi_model.prithvi_model.embed_dim
        self.decoder = SimpleDecoder_comb_v2(
            prithvi_emb_dim, hidden_dim=256, output_dim=64, n_tokens=n_tokens
        )
        self.pt1d_conv_branch = Pt1dConvBranch(T_MERRA=T_MERRA)
        self.fc_final = nn.Linear(128, 1)  # Regression output

    def forward(self, im2d, pt1d, temporal_coords=None, location_coords=None, **kwargs):
        # Pass HLS im2d through the pretrained prithvi MAE encoder (with frozen weights).
        # temporal_coords / location_coords are no-ops unless the backbone was built with coords_encoding.
        pri_enc = self.prithvi_model(im2d, temporal_coords, location_coords, 0)

        # Pass pri_enc through the simple decoder
        dec_out = self.decoder(pri_enc)  # op Shape [batch_size, 64]
        # Pass MERRA pt1d through the convolutional layers
        pt1d_out = self.pt1d_conv_branch(pt1d)  # Shape [batch_size, 64]
        # Concatenate decoder output and pt1d output
        combined = torch.cat((dec_out[:, :], pt1d_out), dim=1) # op: [batch x 128]
        # Final regression output
        output1 = self.fc_final(combined)  # Shape [batch_size, 1]
        #output2 = self.fc_final2(output1)  # Shape [batch_size, 1]
        output = ModelOutput(output=output1)
        
        return output