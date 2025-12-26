from dataclasses import dataclass


@dataclass
class EncoderVisualizerGapmesCfg:
    num_samples: int
    min_resolution: int
    export_ply: bool