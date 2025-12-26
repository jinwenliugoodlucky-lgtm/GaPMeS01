from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_gapmes_cfg import EncoderVisualizerGapmesCfg

import torchvision.transforms as T

from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.dpt_head import DPTHead


@dataclass
class EncoderGapmesCfg:             
    name: Literal["gapmes"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerGapmesCfg
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    supervise_intermediate_depth: bool
    return_depth: bool

    train_depth_only: bool

    monodepth_vit_type: str

    local_mv_match: int
    
    num_moe_experts: int = 8
    num_moe_tasks: int = 4
    moe_top_k: int = 2
    moe_hidden_dim: int = 256
    patch_size: int = 16
    unified_feature_dim: int = 128


class MoEGate(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
    
    def forward(self, x):
        logits = self.net(x)
        return logits


class TopKGating(nn.Module):
    def __init__(self, k: int = 2, use_gumbel: bool = True):
        super().__init__()
        self.k = k
        self.use_gumbel = use_gumbel
    
    def forward(self, logits, temperature: float = 1.0, training: bool = True):
        if self.use_gumbel and training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = logits + gumbel_noise * temperature
        
        soft_weights = F.softmax(logits, dim=-1)
        
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        hard_weights = F.softmax(top_k_logits, dim=-1)
        
        return top_k_indices, soft_weights, hard_weights


class FeatureAligner(nn.Module):
    def __init__(self, depth_dim: int, cnn_dim: int, dino_dim: int, unified_dim: int):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_dim, unified_dim, kernel_size=1)
        self.cnn_proj = nn.Conv2d(cnn_dim, unified_dim, kernel_size=1)
        self.dino_proj = nn.Conv2d(dino_dim, unified_dim, kernel_size=1)
    
    def forward(self, f_depth, f_cnn, f_dino, target_h, target_w):
        x_depth = self.depth_proj(f_depth)
        x_cnn = self.cnn_proj(f_cnn)
        x_dino = self.dino_proj(f_dino)
        
        x_depth = F.interpolate(x_depth, size=(target_h, target_w), mode='bilinear', align_corners=True)
        x_cnn = F.interpolate(x_cnn, size=(target_h, target_w), mode='bilinear', align_corners=True)
        x_dino = F.interpolate(x_dino, size=(target_h, target_w), mode='bilinear', align_corners=True)
        
        Q = torch.cat([x_depth, x_cnn, x_dino], dim=1)
        
        return Q, (x_depth, x_cnn, x_dino)


class PatchTokenizer(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', 
                      ph=self.patch_size, pw=self.patch_size)
        
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        
        return x, h_p, w_p


class PatchAggregator(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, patches, h_p, w_p, patch_dim):
        b = patches.shape[0]
        feature_dim = patch_dim // (self.patch_size * self.patch_size)
        
        patches = rearrange(patches, 'b (h w) (ph pw c) -> b c (h ph) (w pw)',
                           h=h_p, w=w_p, ph=self.patch_size, pw=self.patch_size, c=feature_dim)
        
        return patches


class MoEExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class MultimodalMoE(nn.Module):
    def __init__(self, unified_dim: int, num_experts: int, num_tasks: int, 
                 top_k: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.unified_dim = unified_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.top_k = top_k
        
        query_dim = 3 * unified_dim
        self.expert_gate = MoEGate(query_dim, num_experts, hidden_dim)
        self.task_gate = MoEGate(query_dim, num_tasks, hidden_dim)
        
        self.expert_gating = TopKGating(k=top_k)
        
        self.experts = nn.ModuleList([
            MoEExpert(unified_dim, hidden_dim, unified_dim)
            for _ in range(num_experts)
        ])
    
    def forward(self, q, training: bool = True):
        expert_logits = self.expert_gate(q)
        expert_indices, soft_weights, hard_weights = self.expert_gating(
            expert_logits, training=training
        )
        
        task_logits = self.task_gate(q)
        task_assignment = F.softmax(task_logits, dim=-1)
        
        return expert_indices, soft_weights, hard_weights, task_assignment


class SparseExpertApplication(nn.Module):
    def __init__(self, unified_dim: int, num_experts: int, top_k: int, hidden_dim: int = 256):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([
            MoEExpert(unified_dim, hidden_dim, unified_dim)
            for _ in range(num_experts)
        ])
    
    def forward(self, x_modality, expert_indices, hard_weights):
        b = x_modality.shape[0]
        output = torch.zeros_like(x_modality)
        
        for i in range(self.top_k):
            expert_idx = expert_indices[:, i]
            weight = hard_weights[:, i]
            
            expert_outputs = []
            for j in range(b):
                exp_id = expert_idx[j].item()
                expert_out = self.experts[exp_id](x_modality[j:j+1])
                expert_outputs.append(expert_out)
            
            expert_output = torch.cat(expert_outputs, dim=0)
            output = output + weight.unsqueeze(1) * expert_output
        
        return output


class TaskAwareHeads(nn.Module):
    def __init__(self, unified_dim: int, num_tasks: int, output_dims: dict):
        super().__init__()
        self.num_tasks = num_tasks
        
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Conv2d(unified_dim, 64, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(64, output_dims.get(task, 1), 3, 1, 1),
            )
            for task in ['mu', 'alpha', 'eps', 'color']
        })
    
    def forward(self, feature_maps):
        outputs = {}
        
        for task in ['mu', 'alpha', 'eps', 'color']:
            if task in feature_maps:
                feat = feature_maps[task]
                
                if task == 'mu':
                    outputs[task] = torch.tanh(self.task_heads[task](feat))
                elif task == 'alpha':
                    outputs[task] = torch.sigmoid(self.task_heads[task](feat))
                elif task == 'eps':
                    outputs[task] = torch.relu(self.task_heads[task](feat))
                else:
                    outputs[task] = torch.tanh(self.task_heads[task](feat))
        
        return outputs


class EncoderGapmes(Encoder[EncoderGapmesCfg]):
    def __init__(self, cfg: EncoderGapmesCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }

        self.feature_upsampler = DPTHead(**model_configs[cfg.monodepth_vit_type],
                                        downsample_factor=cfg.upsample_factor,
                                        return_feature=True,
                                        num_scales=cfg.num_scales,
                                        )
        feature_upsampler_channels = model_configs[cfg.monodepth_vit_type]["features"]
        
        self.feature_aligner = FeatureAligner(
            depth_dim=1,
            cnn_dim=feature_upsampler_channels,
            dino_dim=model_configs[cfg.monodepth_vit_type]['in_channels'],
            unified_dim=cfg.unified_feature_dim,
        )
        
        self.patch_tokenizer = PatchTokenizer(cfg.patch_size)
        self.patch_aggregator = PatchAggregator(cfg.patch_size)
        
        self.multimodal_moe = MultimodalMoE(
            unified_dim=cfg.unified_feature_dim,
            num_experts=cfg.num_moe_experts,
            num_tasks=cfg.num_moe_tasks,
            top_k=cfg.moe_top_k,
            hidden_dim=cfg.moe_hidden_dim,
        )
        
        self.sparse_expert_app = SparseExpertApplication(
            unified_dim=cfg.unified_feature_dim,
            num_experts=cfg.num_moe_experts,
            top_k=cfg.moe_top_k,
            hidden_dim=cfg.moe_hidden_dim,
        )
        
        self.task_heads = TaskAwareHeads(
            unified_dim=cfg.unified_feature_dim,
            num_tasks=cfg.num_moe_tasks,
            output_dims={'mu': 2, 'alpha': 1, 'eps': 3, 'color': 3},
        )
        
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)
                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        depth_preds = results_dict['depth_preds']
        depth = depth_preds[-1]

        if self.cfg.train_depth_only:
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")
            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                num_depths = len(depth_preds)
                intermediate_depths = torch.cat(depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")
                depths = torch.cat((intermediate_depths, depths), dim=0)
                b *= num_depths

            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)

            return {
                "gaussians": None,
                "depths": depths
            }

        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )

        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[0]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')

        Q, (x_depth, x_cnn, x_dino) = self.feature_aligner(
            match_prob,
            features,
            results_dict["features_mono_intermediate"][-1],
            h, w
        )
        
        patches_q, h_p, w_p = self.patch_tokenizer(Q)
        patches_depth, _, _ = self.patch_tokenizer(x_depth)
        patches_cnn, _, _ = self.patch_tokenizer(x_cnn)
        patches_dino, _, _ = self.patch_tokenizer(x_dino)
        
        bv, num_patches, patch_dim = patches_q.shape
        query_dim = 3 * self.cfg.unified_feature_dim
        
        task_outputs = {task: [] for task in ['mu', 'alpha', 'eps', 'color']}
        
        for patch_idx in range(num_patches):
            q_i = patches_q[:, patch_idx, :]
            
            expert_indices, soft_weights, hard_weights, task_assignment = self.multimodal_moe(
                q_i, training=self.training
            )
            
            x_depth_i = patches_depth[:, patch_idx, :self.cfg.unified_feature_dim]
            x_cnn_i = patches_cnn[:, patch_idx, :self.cfg.unified_feature_dim]
            x_dino_i = patches_dino[:, patch_idx, :self.cfg.unified_feature_dim]
            
            phi_depth = self.sparse_expert_app(x_depth_i, expert_indices, hard_weights)
            phi_cnn = self.sparse_expert_app(x_cnn_i, expert_indices, hard_weights)
            phi_dino = self.sparse_expert_app(x_dino_i, expert_indices, hard_weights)
            
            phi_fused = phi_depth + phi_cnn + phi_dino
            
            for task_idx, task_name in enumerate(['mu', 'alpha', 'eps', 'color']):
                task_weight = task_assignment[:, task_idx]
                task_output = phi_fused * task_weight.unsqueeze(1)
                task_outputs[task_name].append(task_output)
        
        feature_maps = {}
        for task_name in ['mu', 'alpha', 'eps', 'color']:
            task_patches = torch.stack(task_outputs[task_name], dim=1)
            task_map = self.patch_aggregator(task_patches, h_p, w_p, self.cfg.unified_feature_dim)
            feature_maps[task_name] = task_map
        
        gaussian_params = self.task_heads(feature_maps)
        
        depth_expanded = rearrange(depth, "b v h w -> b v (h w) () ()")
        
        mu = rearrange(gaussian_params['mu'], "(b v) c h w -> b v (h w) c", b=b, v=v)
        alpha = rearrange(gaussian_params['alpha'], "(b v) c h w -> b v (h w) c", b=b, v=v)
        eps = rearrange(gaussian_params['eps'], "(b v) c h w -> b v (h w) c", b=b, v=v)
        
        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        
        opacities = alpha.sigmoid().unsqueeze(-1)
        
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        
        offset_xy = mu[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        
        sh_input_images = context["image"]

        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depth_expanded,
            opacities,
            rearrange(eps, "b v r srf c -> b v r srf () c"),
            (h, w),
            input_images=sh_input_images if self.cfg.init_sh_input_img else None,
        )

        gaussians = Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
        )

        if self.cfg.return_depth:
            depths = rearrange(
                depth_expanded, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)

            return {
                "gaussians": gaussians,
                "depths": depths
            }

        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor,
            )
            return batch
        return data_shim

    @property
    def sampler(self):
        return None