import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import trimesh
from torchvision.utils import save_image

from omegaconf import OmegaConf
from pixel3dmm import env_paths
from pixel3dmm.tracking.flame.FLAME import FLAME
from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix, matrix_to_rotation_6d
from pixel3dmm.tracking import nvdiffrast_util, util

# ------ 
from adamuniform import AdamUniform
from renderer import NormalRenderer

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

from abc import ABC, abstractmethod

console = Console()

class TrackerBase(ABC):
    """
    Simplified tracker for post-processing analysis.
    Loads tracker.py outputs and performs vertex-level optimization in canonical space.
    """

    def __init__(self, config, device='cuda:0'):
        self.config = config
        self.device = device
        self.actor_name = self.config.video_name

        # Setup paths
        self.data_folder = f'{env_paths.PREPROCESSED_DATA}/{self.actor_name}'
        frame_dst = "_nV1_noPho_no_jaw_uv2000.0_n1000.0"
        self.output_folder = f'{env_paths.TRACKING_OUTPUT}/{self.actor_name}' + frame_dst
        self.checkpoint_folder = os.path.join(self.output_folder, "checkpoint")
        self.save_folder = f'./outputs/{self.actor_name}'
        self.refined_mesh_folder = os.path.join(self.save_folder, "refined_mesh")
        self.optimization_vis_folder = os.path.join(self.save_folder, "optimization_progress")  
        os.makedirs(self.refined_mesh_folder, exist_ok=True)
        os.makedirs(self.optimization_vis_folder, exist_ok=True)  

        console.print(Panel.fit(
            f"[bold cyan]Initializing TrackerBase[/bold cyan]\n"
            f"Actor: [yellow]{self.actor_name}[/yellow]\n"
            f"Data folder: [green]{self.data_folder}[/green]\n"
            f"Output folder: [green]{self.output_folder}[/green]",
            border_style="cyan"
        ))

        # Initialize FLAME model
        mesh_file = f'{env_paths.head_template_noeye}'
        
        with console.status("[bold yellow]Loading FLAME model...", spinner="dots"):
            self.flame_model = FLAME(self.config).to(self.device)
            self.FLAME_EYE_IDX = 3931
            shapedirs = self.flame_model.shapedirs[:self.FLAME_EYE_IDX]
            print(shapedirs.shape)
            self.shape_basis, self.shape_std = self.decompose_basis(shapedirs[..., :300].reshape(-1, 300))
            self.exp_basis, self.exp_std = self.decompose_basis(shapedirs[..., 300:].reshape(-1, 100))
            
            # Load face mask for regularization
            flame_mesh_mask = np.load(f'{env_paths.FLAME_ASSETS}/FLAME2020/FLAME_masks/FLAME_masks.pkl', 
                                       allow_pickle=True, encoding='latin1')
            self.vertex_face_mask = torch.from_numpy(flame_mesh_mask['face']).cuda().long()

            # Initialize renderer
            self.diff_renderer = NormalRenderer(
                512,
                obj_filename=mesh_file,
                eyehole_path=env_paths.EYEHOLE_MASK,
                mouthhole_path=env_paths.MOUTHHOLE_MASK,
            ).to(self.device)
        
        console.print("[bold green]✓[/bold green] FLAME model and renderer loaded!")

        # Store landmark indices (from tracker.py)
        self.left_iris_flame = [4597, 4542, 4510, 4603, 4570]
        self.right_iris_flame = [4051, 3996, 3964, 3932, 4028]
        
        # WFLW to iBUG68 mapping
        self.WFLW_2_iBUG68 = torch.tensor([
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 
            42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 67, 68, 
            69, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 
            92, 93, 94, 95
        ]).cuda()

        self.weight_data = 1.0
        self.weight_lm = 1.0
        self.weight_reg = 1.0
        self.weight_mask = 50.0

        console.print("[bold green]✓[/bold green] TrackerSimple initialized successfully!")


    def decompose_basis(self, basis):
        eigvals = torch.norm(basis, dim=0)
        eigvecs = basis / eigvals

        return eigvecs, eigvals

    def load_checkpoint(self, frame_id):
        """Load checkpoint from tracker.py"""
        checkpoint_path = f'{self.checkpoint_folder}/{frame_id:05d}.frame'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        frame_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        processed_data = {}

        # Extract FLAME parameters
        if 'flame' in frame_data:
            flame_params = frame_data['flame']
            for key in ['exp', 'shape', 'eyes', 'eyelids', 'jaw', 'neck', 'R', 't', 'R_rotation_matrix']:
                if key in flame_params:
                    processed_data[key] = torch.from_numpy(flame_params[key]).float().to(self.device)

        # Extract camera parameters
        if 'camera' in frame_data:
            cam_params = frame_data['camera']
            for key in cam_params.keys():
                if key.startswith('R_base'):
                    processed_data['R_base'] = torch.from_numpy(cam_params[key]).float().to(self.device)
                elif key.startswith('t_base'):
                    processed_data['t_base'] = torch.from_numpy(cam_params[key]).float().to(self.device)

            if 'fl' in cam_params:
                processed_data['focal_length'] = torch.from_numpy(cam_params['fl']).float().to(self.device)
            if 'pp' in cam_params:
                processed_data['principal_point'] = torch.from_numpy(cam_params['pp']).float().to(self.device)

        if 'frame_id' in frame_data:
            processed_data['frame_id'] = frame_data['frame_id']
        if 'img_size' in frame_data:
            processed_data['img_size'] = frame_data['img_size']

        return processed_data

    def read_ground_truth_data(self, timestep):
        """Read ground truth data for optimization"""
        # Load RGB image
        try:
            rgb_path = f'{self.data_folder}/cropped/{timestep:05d}.jpg'
            rgb = np.array(Image.open(rgb_path).resize((self.config.size, self.config.size))) / 255.0
        except:
            rgb_path = f'{self.data_folder}/cropped/{timestep:05d}.png'
            rgb = np.array(Image.open(rgb_path).resize((self.config.size, self.config.size))) / 255.0

        # Load normal map
        normal_path = f'{self.data_folder}/p3dmm/normals/{timestep:05d}.png'
        normals = ((np.array(Image.open(normal_path).resize((self.config.size, self.config.size))) / 255.0).astype(np.float32) - 0.5) * 2

        # Load segmentation for masks
        seg = np.array(Image.open(f'{self.data_folder}/seg_og/{timestep:05d}.png').resize(
            (self.config.size, self.config.size), Image.NEAREST))
        if len(seg.shape) == 3:
            seg = seg[..., 0]

        # 8, 9: eyes, 11: mouth
        # normal_mask = ((seg == 2) | (seg == 6) | (seg == 7) | (seg == 8) | (seg == 9) |(seg == 10) | (seg == 12) | (seg == 13))
        normal_mask = ((seg == 2) | (seg == 6) | (seg == 7) |(seg == 10) | (seg == 12) | (seg == 13))
        hole_mask = (seg == 8) | (seg == 9) | (seg==11)
        if self.config.big_normal_mask:
            normal_mask = normal_mask | (seg == 1) | (seg == 4) | (seg == 5)

        # Load landmarks
        try:
            lms = np.load(f'{self.data_folder}/PIPnet_landmarks/{timestep:05d}.npy') * self.config.size
            lmk68 = lms[self.WFLW_2_iBUG68.cpu().numpy(), :]
            lmk_mask = ~(lmk68.sum(1, keepdims=True) == 0)
            left_iris = lms[96:97, :]
            right_iris = lms[97:98, :]
            mask_left_iris = ~(lms.sum(1, keepdims=True) == 0)[96:97, :]
            mask_right_iris = ~(lms.sum(1, keepdims=True) == 0)[97:98, :]
        except:
            lmk68 = np.zeros([68, 2])
            lmk_mask = np.zeros([68, 1])
            left_iris = np.zeros([1, 2])
            right_iris = np.zeros([1, 2])
            mask_left_iris = np.zeros([1, 1])
            mask_right_iris = np.zeros([1, 1])

        return {
            'rgb': torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device),
            'normal': torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0).float().to(self.device),
            'normal_mask': torch.from_numpy(normal_mask).unsqueeze(0).unsqueeze(0).float().to(self.device),
            'lmk68': torch.from_numpy(lmk68).unsqueeze(0).float().to(self.device),
            'lmk_mask': torch.from_numpy(lmk_mask).unsqueeze(0).float().to(self.device),
            'left_iris': torch.from_numpy(left_iris).unsqueeze(0).float().to(self.device),
            'right_iris': torch.from_numpy(right_iris).unsqueeze(0).float().to(self.device),
            'mask_left_iris': torch.from_numpy(mask_left_iris).unsqueeze(0).float().to(self.device),
            'mask_right_iris': torch.from_numpy(mask_right_iris).unsqueeze(0).float().to(self.device),
            'hole_mask': torch.from_numpy(hole_mask).unsqueeze(0).unsqueeze(0).float().to(self.device),
        }

    def get_canonical_vertices(self, frame_data):
        """
        Get canonical space vertices from FLAME model.
        Returns vertices in canonical space BEFORE transformation.
        """
        # Extract FLAME parameters
        shape_params = frame_data['shape']
        exp_params = frame_data['exp']
        # shape_parms = torch.zeros_like(shape_params).type_as(shape_params)
        # exp_params = torch.zeros_like(exp_params).type_as(exp_params)
        # exp_params[..., 0] = 0.5
        eyes_params = frame_data.get('eyes')
        jaw_params = frame_data.get('jaw')
        neck_params = frame_data.get('neck')
        eyelids_params = frame_data.get('eyelids')
        R = frame_data['R']
        R_base = frame_data.get('R_base', torch.eye(3).unsqueeze(0).to(self.device))
        
        if R_base.dim() == 2:
            R_base = R_base.unsqueeze(0)

        bs = exp_params.shape[0]
        print(f'bs shape: {bs}')
        
        # Generate canonical vertices from FLAME (before transformation)
        with torch.no_grad():
            vertices_can, lmks_posed, _, vertices_can_wexpr, vertices_noneck_can = self.flame_model(
                cameras=torch.inverse(R_base[:1, ...]).repeat(bs, 1, 1),
                shape_params=shape_params[:1, ...].repeat(bs, 1) if shape_params.dim() > 1 else shape_params.unsqueeze(0).repeat(bs, 1),
                expression_params=exp_params,
                eye_pose_params=eyes_params,
                jaw_pose_params=jaw_params,
                neck_pose_params=neck_params,
                # rot_params_lmk_shift=matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R))),
                rot_params_lmk_shift=R,
                eyelid_params=eyelids_params,
            )

        # Return canonical space vertices (before R and t transformation)
        return vertices_can, vertices_noneck_can, lmks_posed

    def apply_transformation(self, vertices_can, R, t):
        """
        Apply rotation R and translation t to canonical vertices.
        This transforms from canonical space to world space.
        """
        R_matrix = rotation_6d_to_matrix(R)
        vertices_world = torch.einsum('bny,bxy->bnx', vertices_can, R_matrix) + t.unsqueeze(1)
        return vertices_world

    def project_points_screen_space(self, points3d, focal_length, principal_point, R_base, t_base):
        """Project 3D points to screen space (from tracker.py)"""
        size = self.config.size
        
        # Construct intrinsics
        intrinsics = torch.eye(3)[None, ...].float().cuda()
        intrinsics[:, 0, 0] = focal_length.squeeze() * size
        intrinsics[:, 1, 1] = focal_length.squeeze() * size
        intrinsics[:, :2, 2] = size/2 + 0.5 + principal_point * (size/2 + 0.5)

        # Construct extrinsics
        w2c_openGL = torch.eye(4)[None, ...].float().cuda()
        w2c_openGL[:, :3, :3] = R_base[0] if R_base.dim() == 3 else R_base
        w2c_openGL[:, :3, 3] = t_base[0] if t_base.dim() == 2 else t_base

        # Apply transformation
        points_homo = torch.cat([points3d, torch.ones_like(points3d[..., :1])], dim=-1)
        lmk_cam_space = torch.bmm(points_homo, w2c_openGL.permute(0, 2, 1))

        # Project to screen space
        lmk_cam_space_prime = lmk_cam_space[..., :3] / -lmk_cam_space[..., [2]]
        lmk_screen_space = (-1) * torch.bmm(lmk_cam_space_prime, intrinsics.permute(0, 2, 1))[..., :2]
        lmk_screen_space = torch.stack([
            size - 1 - lmk_screen_space[..., 0], 
            lmk_screen_space[..., 1], 
            lmk_cam_space[..., 2]
        ], dim=-1)
        
        return lmk_screen_space

    
    def save_optimization_progress(self, iteration, frame_id, rendered_normals, gt_normal, gt_normal_mask, valid_mask, R, losses, gt_lmk68, pred_lmk68, lmk_mask):
        """
        Save normal comparison during optimization at specific intervals.
        Saves: GT with landmarks, Predicted with landmarks, Masked Predicted, Difference
        """
        R_matrix = rotation_6d_to_matrix(R)
        pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', R_matrix, rendered_normals)
        
        # Visualizations in [0, 1] range
        gt_vis = (gt_normal[0] + 1) / 2
        pred_vis = (pred_normals_flame_space[0] + 1) / 2
        
        # Masked predicted normals (applying both normal mask and eye mask)
        combined_mask = gt_normal_mask[0]
        pred_masked_vis = pred_vis * combined_mask * valid_mask[0]
        
        # Difference map
        diff = (gt_normal[0] - pred_normals_flame_space[0]).abs()
        diff_vis = (diff / 2) * combined_mask
        
        error_mean = (diff * combined_mask).sum().item() / (combined_mask.sum().item() + 1e-8)
        
        gt_vis = gt_vis * combined_mask
        gt_vis_with_lmk = gt_vis.clone()
        if pred_lmk68 is not None and lmk_mask is not None:
            gt_vis_with_lmk = self._draw_landmarks_on_image(gt_vis_with_lmk, pred_lmk68[0], color='red')
        pred_masked_vis_with_lmk = pred_masked_vis.clone()
        if pred_lmk68 is not None and lmk_mask is not None:
            pred_masked_vis_with_lmk = self._draw_landmarks_on_image(pred_masked_vis_with_lmk, pred_lmk68[0], color='red')
            pred_masked_vis_with_lmk = self._draw_landmarks_on_image(pred_masked_vis_with_lmk, gt_lmk68[0], lmk_mask[0], color='green')
        combined_vis = pred_vis * valid_mask[0]
        
        diff_vis_with_text = self._add_text_to_image(diff_vis, f'Err: {error_mean:.4f}')
        
        # Create comparison
        comparison = torch.cat([gt_vis, combined_vis, diff_vis_with_text], dim=2)
        
        # Save comparison image
        progress_path = os.path.join(
            self.optimization_vis_folder, 
            f'frame_{frame_id:05d}_iter_{iteration:04d}.png'
        )
        save_image(comparison, progress_path)


    def _add_text_to_image(self, img_tensor, text, position='bottom_left', font_size=36):
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/opentype/linux-libertine/LinBiolinum_R.otf", font_size)
        except:
            font = ImageFont.load_default()
        
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = 15  
        y = H - text_height - 15  
        
        padding = 2
        draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding], 
                    fill=(0, 0, 0, 180))
        
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor_out = torch.from_numpy(img_np).permute(2, 0, 1).to(img_tensor.device)
        
        return img_tensor_out


    def _draw_landmarks_on_image(self, image_tensor, landmarks, lmk_mask=None, color='red', radius=2):
        """
        Draw landmarks on image tensor.
        
        Args:
            image_tensor: [C, H, W] tensor in [0, 1] range
            landmarks: [N, 2] landmark coordinates (x, y)
            lmk_mask: [N, 1] mask indicating valid landmarks
            color: 'red' or 'green'
            radius: radius of landmark points
        
        Returns:
            image_tensor with landmarks drawn
        """
        # Convert to numpy for drawing
        img_np = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        
        # Set color
        color_map = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
        }
        draw_color = color_map.get(color, (255, 0, 0))
        
        # Draw each landmark
        landmarks_np = landmarks.detach().cpu().numpy()
        if lmk_mask is not None:
            lmk_mask_np = lmk_mask.detach().cpu().numpy()

        for i, (x, y) in enumerate(landmarks_np):
            if lmk_mask is not None:
                if lmk_mask_np[i] > 0:  # Only draw valid landmarks
                    # Draw a small circle
                    draw.ellipse(
                        [(x - radius, y - radius), (x + radius, y + radius)],
                        fill=draw_color,
                        outline=draw_color
                    )
            else:
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    fill=draw_color,
                    outline=draw_color
                )
        
        # Convert back to tensor
        img_with_lmk = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1) / 255.0
        return img_with_lmk.to(image_tensor.device)

    def optimize_vertices(self, frame_data, gt_data, frame_id, num_iterations=500, save_interval=50):
        """
        Optimize vertices in canonical space to minimize residual error.
        Modified to use rich progress display and save intermediate results.
        """
        # Get canonical vertices (before transformation)
        canonical_vertices, canonical_vertices_noneck, flame_lmks = self.get_canonical_vertices(frame_data)
        import potpourri3d as pp3d

        pp3d.write_mesh(canonical_vertices[0].detach().cpu().numpy(), self.diff_renderer.faces[0].detach().cpu().numpy(), os.path.join(self.optimization_vis_folder, 'test.obj'))
        # Create optimizable vertex offsets IN CANONICAL SPACE
        shape_offsets = nn.Parameter(torch.zeros_like(canonical_vertices)[:, :self.FLAME_EYE_IDX])
        exp_offsets = nn.Parameter(torch.zeros_like(canonical_vertices)[:, :self.FLAME_EYE_IDX])
        eyes_offsets = torch.zeros_like(canonical_vertices)[:, self.FLAME_EYE_IDX:]
        
        # Fixed transformation parameters
        R = frame_data['R']
        t = frame_data['t']
        
        # Fixed camera parameters
        focal_length = frame_data.get('focal_length', torch.tensor([[1.0]]).to(self.device))
        principal_point = frame_data.get('principal_point', torch.tensor([[0.0, 0.0]]).to(self.device))
        R_base = frame_data.get('R_base', torch.eye(3).unsqueeze(0).to(self.device))
        t_base = frame_data.get('t_base', torch.zeros(3).unsqueeze(0).to(self.device))
        
        if R_base.dim() == 2:
            R_base = R_base.unsqueeze(0)
        if t_base.dim() == 1:
            t_base = t_base.unsqueeze(0)

        basis = torch.cat([self.shape_basis, self.exp_basis], dim=-1)
        std = torch.cat([self.shape_std, self.exp_std], dim=-1)

        alpha1 = 1 / (self.beta + 1)
        # Setup optimizer
        shape_eigvals = self.shape_std ** 2
        exp_eigvals = self.exp_std ** 2
        shape_eig_max = shape_eigvals.max()
        exp_eig_max = exp_eigvals.max()

        shape_eigvals = (shape_eigvals / shape_eig_max) + self.eps
        exp_eigvals = (exp_eigvals / exp_eig_max) + self.eps

        shape_proj_mat = self.shape_basis @ torch.diag(shape_eigvals) @ self.shape_basis.T
        exp_proj_mat = self.exp_basis @ torch.diag(exp_eigvals) @ self.exp_basis.T

        shape_precond_mat = (1 - alpha1) * shape_proj_mat + alpha1 * torch.eye(shape_proj_mat.shape[0]).type_as(shape_proj_mat)
        exp_precond_mat = (1 - alpha1) * exp_proj_mat + alpha1 * torch.eye(exp_proj_mat.shape[0]).type_as(exp_proj_mat)

        self.optimizer = self.configure_optimizer(shape_offsets, exp_offsets)
        
        # Get MVP matrix
        r_mvps = self._get_mvp_matrix(focal_length, principal_point, R_base, t_base)

        console.print(Panel.fit(
            f"[bold cyan]Starting Vertex Optimization[/bold cyan]\n"
            f"Frame: [yellow]{frame_id}[/yellow]\n"
            f"Iterations: [green]{num_iterations}[/green]\n"
            f"Optimization space: [magenta]Canonical Space[/magenta]\n"
            f"Save interval: [blue]{save_interval}[/blue]",
            border_style="cyan"
        ))
        
        best_loss = float('inf')
        best_offsets = None
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.fields[loss_info]}"),
            TimeElapsedColumn(),
            console=console
        )
        
        with progress:
            task = progress.add_task(
                f"[cyan]Optimizing Frame {frame_id}...", 
                total=num_iterations,
                loss_info=""
            )
            
            for iteration in range(num_iterations):
                self.optimizer.zero_grad()
                # STEP 1: Apply offsets in canonical space
                shape_offsets_eye = torch.cat([shape_offsets, eyes_offsets], dim=1)
                exp_offsets_eye = torch.cat([exp_offsets, eyes_offsets], dim=1)

                proj_offsets = (shape_precond_mat @ (shape_offsets_eye).reshape(-1)).reshape(-1, 3) + (exp_precond_mat @ (exp_offsets_eye).reshape(-1)).reshape(-1, 3)
                optimized_canonical_vertices = canonical_vertices + proj_offsets
                optimized_canonical_vertices_noneck = canonical_vertices_noneck + proj_offsets
                
                # STEP 2: Transform from canonical space to world space
                optimized_world_vertices = self.apply_transformation(optimized_canonical_vertices, R, t)
                optimized_world_vertices_noneck = self.apply_transformation(optimized_canonical_vertices_noneck, R, t)
                
                # STEP 3: Render with world space vertices
                ops = self.diff_renderer(
                    optimized_world_vertices[:, :self.FLAME_EYE_IDX],
                    optimized_world_vertices_noneck[:, :self.FLAME_EYE_IDX],
                    r_mvps
                )

                # STEP 4: Compute losses
                losses = {}
                
                # 1. Normal loss
                rendered_normals = ops['normal_images']
                R_matrix = rotation_6d_to_matrix(R)
                pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', R_matrix, rendered_normals)
                
                '''
                # Apply eye mask (from tracker.py)
                dilated_eye_mask = 1 - (gaussian_blur(
                    ops['mask_images_eyes'], 
                    [self.config.normal_mask_ksize, self.config.normal_mask_ksize],
                    sigma=[self.config.normal_mask_ksize, self.config.normal_mask_ksize]
                ) > 0).float()
                '''

                mask_for_loss = gt_data['normal_mask'] # * ops['mask_images_rendering']
                normal_diff = (gt_data['normal'] - pred_normals_flame_space)
                # valid_normals = ((normal_diff.abs().sum(dim=1) / 3) < self.config.delta_n).unsqueeze(1)
                # normal_loss_map = normal_diff * valid_normals.float() *  mask_for_loss
                normal_loss_map = normal_diff * mask_for_loss
                if self.config.normal_l2:
                    losses['normal'] = self.weight_data * normal_loss_map.square().mean()
                else:
                    losses['normal'] = self.weight_data * normal_loss_map.abs().mean()

                losses['hole'] = self.weight_mask * (ops['hole_mask'] - gt_data['hole_mask']).square().mean()
                losses['skin'] = self.weight_mask * (gt_data['normal_mask'] * (ops['skin_mask'] - gt_data['normal_mask'])).square().mean()

                # 2. Landmark losses
                if gt_data['lmk68'] is not None and not torch.all(gt_data['lmk68'] == 0):
                    eye_lmk_idx = [2422, 2454, 2471, 2508, 2368, 2269, 3833, 1343, 1216, 1154, 814, 827, 2370, 1012]
                    pred_lmks = optimized_world_vertices[:, eye_lmk_idx]
                    proj_eye_lmks = self.project_points_screen_space(
                        pred_lmks, focal_length, principal_point, R_base, t_base
                    )[...,:2]
                    '''
                    pred_lmks = self.flame_model.convert_to_static_landmarks(optimized_world_vertices)
                    proj_eye_lmks = self.project_points_screen_space(
                        pred_lmks, focal_length, principal_point, R_base, t_base
                    )[:,19:31,:2]
                    '''
                    gt_eye_lmks = gt_data['lmk68'][:,36:48,:]
                    gt_eye_lmks_mask = gt_data['lmk_mask'][:, 36:48, :]

                    extend_lmks = torch.tensor([[158, 191], [300, 156]]).type_as(gt_eye_lmks).unsqueeze(0)
                    extend_lmks_mask = torch.tensor([[1], [1]]).type_as(gt_eye_lmks_mask).unsqueeze(0)
                    gt_eye_lmks = torch.cat([gt_eye_lmks, extend_lmks], dim=1)
                    gt_eye_lmks_mask = torch.cat([gt_eye_lmks_mask, extend_lmks_mask], dim=1)
                    
                    # Eye closure loss
                    losses['lmk_eye'] = util.lmk_loss(
                        proj_eye_lmks, gt_eye_lmks, 
                        [self.config.size, self.config.size],
                        gt_eye_lmks_mask,
                    ) * self.weight_lm
                    '''
                    
                    # Iris losses
                    losses['lmk_iris_left'] = util.lmk_loss(
                        proj_vertices[:, self.left_iris_flame[:1], ..., :2],
                        gt_data['left_iris'], [self.config.size, self.config.size],
                        gt_data['mask_left_iris']
                    ) * 0.0 
                    
                    losses['lmk_iris_right'] = util.lmk_loss(
                        proj_vertices[:, self.right_iris_flame[:1], ..., :2],
                        gt_data['right_iris'], [self.config.size, self.config.size],
                        gt_data['mask_right_iris']
                    ) * 0.0 
                    '''

                # Total loss
                total_loss = sum(losses.values())
                
                # Backward and optimize
                total_loss.backward()
                self.optimizer.step()
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_offsets = vertex_offsets_canonical.detach().clone()
                    best_offsets = torch.cat([best_offsets, eyes_offsets], dim=1)
                
                # Save intermediate visualization at specified intervals
                if iteration % save_interval == 0 or iteration == num_iterations - 1:
                    with torch.no_grad():
                        self.save_optimization_progress(
                            iteration, frame_id, rendered_normals, 
                            gt_data['normal'], mask_for_loss, 1 - ops['hole_mask'],
                            R, losses,
                            gt_eye_lmks, proj_eye_lmks, gt_eye_lmks_mask
                        )

                # Update progress bar with loss information
                loss_info = f"total: {total_loss.item():.4f}"
                if 'normal' in losses:
                    loss_info += f" | normal: {losses['normal'].item():.4f}"
                if 'lmk_eye' in losses:
                    loss_info += f" | lmk: {losses['lmk_eye'].item():.4f}"
                if 'hole' in losses:
                    loss_info += f" | hole: {losses['hole'].item():.4f}"
                if 'skin' in losses:
                    loss_info += f" | skin: {losses['skin'].item():.4f}"

                
                progress.update(task, advance=1, loss_info=loss_info)

        console.print(f"[bold green]✓[/bold green] Optimization complete! Best loss: [yellow]{best_loss:.4f}[/yellow]")

        
        # Return optimized vertices in both spaces
        final_canonical_vertices = canonical_vertices + best_offsets
        final_world_vertices = self.apply_transformation(final_canonical_vertices, R, t)
        
        return final_world_vertices, final_canonical_vertices, best_offsets

    def _get_mvp_matrix(self, focal_length, principal_point, R_base, t_base):
        """Create Model-View-Projection matrix"""
        size = self.config.size

        if focal_length.dim() == 0:
            focal_length = focal_length.unsqueeze(0)
        if principal_point.dim() == 1:
            principal_point = principal_point.unsqueeze(0)

        # Create intrinsics matrix
        intrinsics = torch.eye(3)[None, ...].float().cuda().repeat(focal_length.shape[0], 1, 1)
        intrinsics[:, 0, 0] = focal_length.squeeze() * size
        intrinsics[:, 1, 1] = focal_length.squeeze() * size
        intrinsics[:, :2, 2] = size/2 + 0.5 + principal_point * (size/2 + 0.5)

        # Create projection matrix
        proj = nvdiffrast_util.intrinsics2projection(
            intrinsics, znear=0.1, zfar=5, width=size, height=size
        )

        # Create extrinsics matrix
        extr = torch.eye(4).float().cuda().unsqueeze(0)
        extr[:, :3, :3] = R_base[0] if R_base.dim() == 3 else R_base
        extr[:, :3, 3] = t_base[0] if t_base.dim() == 2 else t_base

        # Compute MVP matrix
        r_mvps = torch.matmul(proj, extr)
        return r_mvps

    def save_results(self, frame_id, world_vertices, canonical_vertices, vertex_offsets, frame_data, gt_data):
        """
        Save final results:
        - Refined mesh (world space)
        - Refined canonical mesh (canonical space)
        - Vertex offsets (canonical space)
        - Rendered normal comparison
        """
        faces = self.flame_model.faces.cpu().numpy()
        
        # 1. Save refined mesh in WORLD SPACE
        vertices_world_np = world_vertices[0].detach().cpu().numpy()
        mesh_world_path = os.path.join(self.refined_mesh_folder, f'{frame_id:05d}_refined_world.ply')
        trimesh.Trimesh(vertices=vertices_world_np, faces=faces, process=False).export(mesh_world_path)
        print(f"Saved refined world space mesh to {mesh_world_path}")
        
        # 2. Save refined mesh in CANONICAL SPACE
        vertices_canonical_np = canonical_vertices[0].detach().cpu().numpy()
        mesh_canonical_path = os.path.join(self.refined_mesh_folder, f'{frame_id:05d}_refined_canonical.ply')
        trimesh.Trimesh(vertices=vertices_canonical_np, faces=faces, process=False).export(mesh_canonical_path)
        print(f"Saved refined canonical space mesh to {mesh_canonical_path}")
        
        # 3. Save vertex offsets (in canonical space)
        offset_path = os.path.join(self.refined_mesh_folder, f'{frame_id:05d}_offsets_canonical.npy')
        np.save(offset_path, vertex_offsets.detach().cpu().numpy())
        print(f"Saved canonical space offsets to {offset_path}")
        
        # 4. Render and save normal comparison
        focal_length = frame_data.get('focal_length', torch.tensor([[1.0]]).to(self.device))
        principal_point = frame_data.get('principal_point', torch.tensor([[0.0, 0.0]]).to(self.device))
        R_base = frame_data.get('R_base', torch.eye(3).unsqueeze(0).to(self.device))
        t_base = frame_data.get('t_base', torch.zeros(3).unsqueeze(0).to(self.device))
        R = frame_data['R']
        t = frame_data['t']
        
        if R_base.dim() == 2:
            R_base = R_base.unsqueeze(0)
        if t_base.dim() == 1:
            t_base = t_base.unsqueeze(0)
        
        r_mvps = self._get_mvp_matrix(focal_length, principal_point, R_base, t_base)
        
        # Get vertices_noneck by applying the same offsets
        # First get canonical vertices_noneck from FLAME
        _, canonical_vertices_noneck, _ = self.get_canonical_vertices(frame_data)
        # Apply the same offsets to canonical_vertices_noneck
        optimized_canonical_vertices_noneck = canonical_vertices_noneck + vertex_offsets
        # Transform to world space
        world_vertices_noneck = self.apply_transformation(optimized_canonical_vertices_noneck, R, t)
        
        ops = self.diff_renderer(
            world_vertices[:, :self.FLAME_EYE_IDX],
            world_vertices_noneck[:, :self.FLAME_EYE_IDX],
            r_mvps
        )
        
        rendered_normals = ops['normal_images']
        R_matrix = rotation_6d_to_matrix(R)
        pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', R_matrix, rendered_normals)
        
        # Create comparison: [GT | Refined | Difference]
        gt_vis = (gt_data['normal'][0] + 1) / 2
        pred_vis = (pred_normals_flame_space[0] + 1) / 2
        diff_vis = ((gt_data['normal'][0] - pred_normals_flame_space[0]).abs() / 2)
        
        comparison = torch.cat([gt_vis, pred_vis, diff_vis], dim=2)
        
        comparison_path = os.path.join(self.refined_mesh_folder, f'{frame_id:05d}_normal_comparison.png')
        save_image(comparison, comparison_path)
        print(f"Saved normal comparison to {comparison_path}")

        # 5. Render & save FRONTAL view normal map
        # canonical_vertices are already personalized (zero-shape + optimized offsets).
        # Bypassing R, t keeps the face in canonical (frontal) orientation; the original
        # camera (r_mvps) was set up via look_at(origin), so this yields a frontal view.
        frontal_R_base = torch.eye(3).float().to(self.device).unsqueeze(0)
        frontal_t_base = t_base.clone()
        print(frontal_t_base)
        frontal_t_base[:, 2] = -0.9
        if frontal_t_base.dim() == 1:
            frontal_t_base = frontal_t_base.unsqueeze(0)
        r_mvps_frontal = self._get_mvp_matrix(focal_length, principal_point, frontal_R_base, frontal_t_base)

        ops_frontal = self.diff_renderer(
            canonical_vertices[:, :self.FLAME_EYE_IDX],
            optimized_canonical_vertices_noneck[:, :self.FLAME_EYE_IDX],
            r_mvps_frontal
        )
        frontal_normals = ops_frontal['normal_images']
        frontal_vis = (frontal_normals[0] + 1) / 2
        frontal_path = os.path.join(self.refined_mesh_folder, f'{frame_id:05d}_normal_frontal.png')
        save_image(frontal_vis, frontal_path)
        print(f"Saved frontal normal map to {frontal_path}")

        # 6. Save offset visualization (optional)
        # Create a heatmap showing the magnitude of offsets
        offset_magnitude = torch.norm(vertex_offsets[0], dim=-1).detach().cpu().numpy()
        offset_vis_path = os.path.join(self.refined_mesh_folder, f'{frame_id:05d}_offset_magnitude.npy')
        np.save(offset_vis_path, offset_magnitude)
        print(f"Saved offset magnitude to {offset_vis_path}")

    def run_analysis(self, start_frame=None, end_frame=None, num_iterations=500, save_interval=50):
        """
        Main pipeline: load checkpoints, optimize vertices in canonical space, save results.
        Modified to use rich console output.
        """
        start_frame = start_frame or self.config.start_frame

        # Find available checkpoint files
        with console.status("[bold yellow]Scanning checkpoint files...", spinner="dots"):
            checkpoint_files = [f for f in os.listdir(self.checkpoint_folder) if f.endswith('.frame')]
            available_frames = sorted([int(f.split('.')[0]) for f in checkpoint_files])

        if end_frame is None:
            end_frame = max(available_frames)

        console.print(Panel.fit(
            f"[bold cyan]Starting Vertex Refinement Pipeline[/bold cyan]\n"
            f"Optimization space: [magenta]Canonical Space[/magenta]\n"
            f"Frame range: [yellow]{start_frame}[/yellow] → [yellow]{end_frame}[/yellow]\n"
            f"Available frames: [green]{len(available_frames)}[/green]\n"
            f"Iterations per frame: [blue]{num_iterations}[/blue]\n"
            f"Visualization interval: [blue]{save_interval}[/blue]",
            border_style="cyan"
        ))

        for frame_id in available_frames:
            if frame_id < start_frame or frame_id > end_frame:
                continue

            console.rule(f"[bold cyan]Frame {frame_id}[/bold cyan]")

            # Load checkpoint and ground truth data
            with console.status(f"[bold yellow]Loading frame {frame_id} data...", spinner="dots"):
                frame_data = self.load_checkpoint(frame_id)
                gt_data = self.read_ground_truth_data(frame_id)
            console.print("[bold green]✓[/bold green] Data loaded!")

            # Optimize vertices in canonical space
            world_vertices, canonical_vertices, vertex_offsets = self.optimize_vertices(
                frame_data, gt_data, frame_id, 
                num_iterations=num_iterations,
                save_interval=save_interval
            )

            # Save results
            with console.status(f"[bold yellow]Saving results for frame {frame_id}...", spinner="dots"):
                self.save_results(frame_id, world_vertices, canonical_vertices, vertex_offsets, frame_data, gt_data)
            console.print("[bold green]✓[/bold green] Results saved!")

        console.print(Panel.fit(
            "[bold green]✓ Vertex Refinement Complete![/bold green]\n"
            f"Results saved to:\n"
            f"  • Meshes: [cyan]{self.refined_mesh_folder}[/cyan]\n"
            f"  • Progress: [cyan]{self.optimization_vis_folder}[/cyan]",
            border_style="green"
        ))

    @abstractmethod
    def configure_optimizer(self, opt_vars, precond_mat):
        pass

class TrackerStat(TrackerBase):
    def __init__(self, config):
        super().__init__(config)
        self.beta = 30
        self.eps = 0.3
        self.weight_lm = 0 
        self.weight_reg = 0

        self.lr = 0.01

        print("stat preconditioner")

    def configure_optimizer(self, shape_vars, exp_vars):
        return AdamUniform([shape_vars, exp_vars], lr=self.lr)


def main(cfg):
    """Main function to run vertex refinement in canonical space"""
    tracker = TrackerStat(cfg)
    
    # Check if save_interval is specified in config, otherwise use default
    save_interval = getattr(cfg, 'save_interval', 50)
    
    tracker.run_analysis(
        num_iterations=2000,
        save_interval=save_interval
    )


if __name__ == '__main__':
    # Load configuration
    from omegaconf import OmegaConf
    from pixel3dmm import env_paths
    base_conf = OmegaConf.load(f'{env_paths.CODE_BASE}/configs/tracking.yaml')
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)
    print(cfg)

    main(cfg)
