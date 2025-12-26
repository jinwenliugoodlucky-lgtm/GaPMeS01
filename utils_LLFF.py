<<<<<<< HEAD
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LLFF format from images using COLMAP SfM reconstruction.
Output: poses_bounds.npy (N, 17) for NeRF/Mip-NeRF training.
"""
>>>>>>> e22e98c79a4894d55847fc6b779eac7b101e22f6

import os
import sys
import cv2
import shutil
import argparse
import numpy as np
import pycolmap
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess


class LLFFGenerator:
    """LLFF format generator from images"""
    
    def __init__(self, image_dir: str, output_dir: str):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.images_dir = self.output_dir / "images"
        self.sparse_dir = self.output_dir / "sparse" / "0"
        self.database_path = self.output_dir / "database.db"
        
        self.images_dir.mkdir(exist_ok=True, parents=True)
        self.sparse_dir.mkdir(exist_ok=True, parents=True)
        
    def prepare_images(self, image_list: Optional[List[str]] = None, 
                       max_size: Optional[int] = None):
        """Copy and optionally resize images to output directory"""
        if image_list is None:
            exts = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']
            image_files = []
            for ext in exts:
                image_files.extend(sorted(self.image_dir.glob(ext)))
        else:
            image_files = [self.image_dir / img for img in image_list]
        
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Failed to read {img_path}")
                continue
            
            if max_size is not None:
                h, w = img.shape[:2]
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    print(f"Resize {img_path.name}: ({w}, {h}) -> ({new_w}, {new_h})")
            
            output_path = self.images_dir / img_path.name
            cv2.imwrite(str(output_path), img)
        
        print(f"Images prepared at {self.images_dir}")
    
    def run_colmap_sfm(self, camera_model: str = "SIMPLE_RADIAL", 
                       matcher: str = "exhaustive"):
        """Run COLMAP SfM reconstruction"""
        print("Starting COLMAP SfM reconstruction...")
        
        print("Step 1/4: Feature extraction...")
        pycolmap.extract_features(
            database_path=str(self.database_path),
            image_path=str(self.images_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model=camera_model,
        )
        
        print("Step 2/4: Feature matching...")
        if matcher == "exhaustive":
            pycolmap.match_exhaustive(database_path=str(self.database_path))
        elif matcher == "sequential":
            pycolmap.match_sequential(database_path=str(self.database_path))
        else:
            raise ValueError(f"Unsupported matcher: {matcher}")
        
        print("Step 3/4: Incremental mapping...")
        maps = pycolmap.incremental_mapping(
            database_path=str(self.database_path),
            image_path=str(self.images_dir),
            output_path=str(self.sparse_dir.parent),
        )
        
        if len(maps) > 0:
            largest_idx = max(range(len(maps)), key=lambda i: len(maps[i].points3D))
            maps[largest_idx].write(str(self.sparse_dir))
            print(f"Reconstruction: {len(maps[largest_idx].images)} images, "
                  f"{len(maps[largest_idx].points3D)} 3D points")
        else:
            raise RuntimeError("COLMAP reconstruction failed")
        
        print("Step 4/4: Bundle adjustment...")
        reconstruction = pycolmap.Reconstruction(str(self.sparse_dir))
        pycolmap.bundle_adjustment(reconstruction, pycolmap.BundleAdjustmentOptions())
        reconstruction.write(str(self.sparse_dir))
        
        print("COLMAP reconstruction completed!")
    
    def colmap_to_llff(self, bd_factor: float = 0.75) -> np.ndarray:
        """Convert COLMAP reconstruction to LLFF poses_bounds.npy format"""
        print("Converting COLMAP to LLFF format...")
        
        reconstruction = pycolmap.Reconstruction(str(self.sparse_dir))
        
        image_names = []
        for img_id in reconstruction.images:
            image_names.append(reconstruction.images[img_id].name)
        image_names = sorted(image_names)
        
        poses = []
        bounds = []
        
        for img_name in image_names:
            img_id = None
            for id in reconstruction.images:
                if reconstruction.images[id].name == img_name:
                    img_id = id
                    break
            
            if img_id is None:
                continue
            
            image = reconstruction.images[img_id]
            camera = reconstruction.cameras[image.camera_id]
            
            # World to camera transform
            R = image.cam_from_world.rotation.matrix()
            t = image.cam_from_world.translation
            
            # Camera to world transform
            R_inv = R.T
            t_inv = -R_inv @ t
            
            # Convert COLMAP [right, down, forward] to LLFF [down, right, backwards]
            flip_mat = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            R_inv = flip_mat @ R_inv
            t_inv = flip_mat @ t_inv
            
            pose = np.concatenate([R_inv, t_inv[:, None]], axis=1)
            
            h = camera.height
            w = camera.width
            
            if camera.model_name == "SIMPLE_PINHOLE":
                f = camera.params[0]
            elif camera.model_name == "PINHOLE":
                f = (camera.params[0] + camera.params[1]) / 2.0
            elif camera.model_name in ["SIMPLE_RADIAL", "RADIAL"]:
                f = camera.params[0]
            elif camera.model_name == "OPENCV":
                f = (camera.params[0] + camera.params[1]) / 2.0
            else:
                print(f"Warning: Unknown camera model {camera.model_name}")
                f = (h + w) / 2.0
            
            hwf = np.array([h, w, f])[:, None]
            pose_hwf = np.concatenate([pose, hwf], axis=1)
            poses.append(pose_hwf)
            
            # Compute near/far bounds from 3D point depths
            depths = []
            for point2D in image.points2D:
                if point2D.point3D_id != -1:
                    point3D = reconstruction.points3D[point2D.point3D_id]
                    point_cam = R @ point3D.xyz + t
                    depth = point_cam[2]
                    if depth > 0:
                        depths.append(depth)
            
            if len(depths) > 0:
                depths = np.array(depths)
                near = np.percentile(depths, 5) * bd_factor
                far = np.percentile(depths, 95) / bd_factor
            else:
                near = 0.1
                far = 100.0
            
            bounds.append([near, far])
        
        poses = np.stack(poses, axis=0)
        bounds = np.array(bounds)
        
        poses = self._recenter_poses(poses)
        
        # poses_bounds: (N, 17) = (N, 15) + (N, 2)
        poses_bounds = np.concatenate([poses.reshape(-1, 15), bounds], axis=1)
        
        output_path = self.output_dir / "poses_bounds.npy"
        np.save(output_path, poses_bounds)
        print(f"poses_bounds.npy saved to {output_path}")
        print(f"Shape: {poses_bounds.shape}")
        
        return poses_bounds
    
    def _recenter_poses(self, poses: np.ndarray) -> np.ndarray:
        """Recenter and normalize camera poses"""
        translations = poses[:, :, 3]
        center = translations.mean(axis=0)
        poses[:, :, 3] -= center
        
        radius = np.linalg.norm(poses[:, :, 3], axis=1).mean()
        if radius > 0:
            poses[:, :, 3] /= radius
        
        return poses
    
    def generate_llff_format(self, camera_model: str = "SIMPLE_RADIAL",
                            matcher: str = "exhaustive",
                            max_image_size: Optional[int] = 1024,
                            bd_factor: float = 0.75):
        """Complete pipeline: images -> LLFF format"""
        print("=" * 60)
        print("Step 1: Prepare images")
        print("=" * 60)
        self.prepare_images(max_size=max_image_size)
        
        print("\n" + "=" * 60)
        print("Step 2: COLMAP SfM reconstruction")
        print("=" * 60)
        self.run_colmap_sfm(camera_model=camera_model, matcher=matcher)
        
        print("\n" + "=" * 60)
        print("Step 3: Convert to LLFF format")
        print("=" * 60)
        poses_bounds = self.colmap_to_llff(bd_factor=bd_factor)
        
        print("\n" + "=" * 60)
        print("Completed! LLFF format files generated")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"- images/           : {len(list(self.images_dir.glob('*')))} images")
        print(f"- sparse/0/         : COLMAP reconstruction")
        print(f"- poses_bounds.npy  : {poses_bounds.shape} poses and bounds")
        
        return poses_bounds


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLFF format from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils_colmap.py --input_dir ./my_images --output_dir ./llff_output
  python utils_colmap.py --input_dir ./images --camera_model PINHOLE --max_size 1024
        """
    )
    
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--camera_model", type=str, default="SIMPLE_RADIAL",
                        choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"],
                        help="COLMAP camera model (default: SIMPLE_RADIAL)")
    parser.add_argument("--matcher", type=str, default="exhaustive",
                        choices=["exhaustive", "sequential"],
                        help="Feature matcher type (default: exhaustive)")
    parser.add_argument("--max_size", type=int, default=1024,
                        help="Max image size, resize if larger (default: 1024, 0 for no resize)")
    parser.add_argument("--bd_factor", type=float, default=0.75,
                        help="Bounds shrink factor (default: 0.75)")
    
    args = parser.parse_args()
    
    max_size = args.max_size if args.max_size > 0 else None
    
    generator = LLFFGenerator(args.input_dir, args.output_dir)
    generator.generate_llff_format(
        camera_model=args.camera_model,
        matcher=args.matcher,
        max_image_size=max_size,
        bd_factor=args.bd_factor
    )


if __name__ == "__main__":
    main()
