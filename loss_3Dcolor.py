import torch
import torch.nn.functional as F
from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
from einops import einsum, rearrange

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from .vggt.vggt.models.vggt import VGGT
from .vggt.vggt.utils.load_fn import load_and_preprocess_images



@dataclass
class Loss3DcolorCfg:
    weight: float


@dataclass
class Loss3DcolorCfgWrapper:
    mse: Loss3DcolorCfg


class Loss3Dcolor(Loss[Loss3DcolorCfg, Loss3DcolorCfgWrapper]):

    def get_pointcloud(
        self,
        images: Tensor,
        devices: list[int],
    ):
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(devices)
        images = load_and_preprocess_images()
        with torch.no_grad():
             with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
                predictions = model(images)

        return predictions
    

    def forward(
        self,
        scales:Tensor,
        rotations: Tensor,
        input_images: Tensor | None,
        means: Tensor,
        prediction: DecoderOutput,
        batch: BatchedExample,
        raw_gaussians: Gaussians,
        global_step: int,
        l1_loss: bool,
        sh: Tensor,

        clamp_large_error: float,
        valid_depth_mask: Tensor | None,
        opacities: Float[Tensor, "*#batch"],
        eps: float = 1e-8,
        position_loss_weight: float = 1.0,    # 位置损失权重
        color_loss_weight: float = 1.0,       # 颜色损失权重

    ) -> Float[Tensor, ""]:
        
        prediction=self.get_pointcloud(input_images)
        

        
        
        
       
           

        return 


def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
    