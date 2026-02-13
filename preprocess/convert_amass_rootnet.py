#!/usr/bin/env python3
"""
Convert AMASS data to HDF5 format for MultiScale RootNet training.

AMASS provides true global orientation and translation from SMPL parameters
(has_orient=True). Outputs H36M 17-joint format with root-relative joints.

Uses model-based contact detection (required) for foot contact probabilities.

Usage:
python convert_amass_rootnet.py \
        --motions_pkl /home/pi/pose/motion_bert/MotionBERT/data/motion3d/MB3D_f243s81/AMASS/all_motions_fps60.pkl \
        --fps_csv /home/pi/pose/motion_bert/MotionBERT/data/motion3d/MB3D_f243s81/AMASS/fps.csv \
        --j_regressor /home/pi/pose/rootnet/J_regressor_h36m_correct.npy \
        --body_models_dir /home/pi/pose/motion_bert/MotionBERT/data/motion3d/MB3D_f243s81/AMASS/body_models \
        --output_dir ./data \
        --contact_checkpoint /home/pi/pose/rootnet/checkpoints/ground_detector/model_best.pth \
        --target_fps 60 \
        --window_size 243 \
        --stride 81
"""

import argparse
import pickle
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import os
from os import path as osp

# SMPL body model
from human_body_prior.body_model.body_model import BodyModel
import gc

# Import model-based contact detector
try:
    from contact_detector import ContactDetector
    CONTACT_DETECTOR_AVAILABLE = True
except ImportError:
    CONTACT_DETECTOR_AVAILABLE = False
    print("Warning: contact_detector module not found, model-based contact detection unavailable")

# Global contact detector instance (lazy loaded)
_contact_detector: Optional['ContactDetector'] = None


def get_contact_detector(checkpoint_path: str) -> 'ContactDetector':
    """Get or create the global contact detector instance."""
    global _contact_detector
    if _contact_detector is None:
        if not CONTACT_DETECTOR_AVAILABLE:
            raise RuntimeError("contact_detector module not available")
        _contact_detector = ContactDetector(checkpoint_path=checkpoint_path)
    return _contact_detector


def replace_non_finite(values: np.ndarray, label: str) -> np.ndarray:
    """Replace NaN/Inf values with zeros."""
    non_finite_mask = ~np.isfinite(values)
    count = int(np.count_nonzero(non_finite_mask))
    if count > 0:
        print(f"Warning: Replacing {count} non-finite values in {label} with 0")
        values = values.copy()
        values[non_finite_mask] = 0
    return values


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle to rotation matrix using Rodrigues formula (numpy version).

    Args:
        axis_angle: (..., 3) axis-angle vectors

    Returns:
        matrix: (..., 3, 3) rotation matrices
    """
    # Add epsilon inside sqrt for numerical stability
    angle = np.sqrt(np.sum(axis_angle ** 2, axis=-1, keepdims=True) + 1e-8)
    axis = axis_angle / angle

    cos_a = np.cos(angle)[..., np.newaxis]  # (..., 1, 1)
    sin_a = np.sin(angle)[..., np.newaxis]  # (..., 1, 1)

    # Reshape for broadcasting
    cos_a = cos_a.reshape(axis_angle.shape[:-1] + (1, 1))
    sin_a = sin_a.reshape(axis_angle.shape[:-1] + (1, 1))

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)

    # Skew-symmetric matrix K
    K = np.stack([
        np.stack([zeros, -z, y], axis=-1),
        np.stack([z, zeros, -x], axis=-1),
        np.stack([-y, x, zeros], axis=-1),
    ], axis=-2)  # (..., 3, 3)

    # Identity matrix
    I = np.eye(3).reshape((1,) * (len(axis_angle.shape) - 1) + (3, 3))
    I = np.broadcast_to(I, K.shape)

    # Rodrigues formula: R = I + sin(θ)K + (1 - cos(θ))K²
    R = I + sin_a * K + (1 - cos_a) * np.matmul(K, K)

    return R


def matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to axis-angle (numpy version).

    Args:
        matrix: (..., 3, 3) rotation matrices

    Returns:
        axis_angle: (..., 3) axis-angle vectors
    """
    # Compute angle from trace
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    # Clamp to avoid numerical issues with acos
    cos_angle = np.clip((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    angle = np.arccos(cos_angle)

    # Extract axis from skew-symmetric part
    axis = np.stack([
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1],
    ], axis=-1)

    # Normalize axis (add epsilon for numerical stability)
    axis_norm = np.sqrt(np.sum(axis ** 2, axis=-1, keepdims=True) + 1e-8)

    # For small angles, axis is undefined - use default [1, 0, 0]
    small_angle_mask = angle[..., np.newaxis] < 1e-6
    default_axis = np.zeros_like(axis)
    default_axis[..., 0] = 1.0

    axis_normalized = np.where(small_angle_mask, default_axis, axis / axis_norm)

    return axis_normalized * angle[..., np.newaxis]


def make_orient_frame0_relative(orient: np.ndarray) -> np.ndarray:
    """
    Make orientations relative to frame 0 using proper rotation composition.

    For rotations: R_relative = R_t @ R_0^(-1)
    NOT: axis_angle_relative = axis_angle_t - axis_angle_0

    Args:
        orient: (T, 3) axis-angle orientations

    Returns:
        orient_rel: (T, 3) frame-0-relative axis-angle orientations
    """
    # Convert to rotation matrices
    R = axis_angle_to_matrix_np(orient)  # (T, 3, 3)

    # Get R_0 inverse (transpose for rotation matrices)
    R_0_inv = R[0:1].transpose(0, 2, 1)  # (1, 3, 3)

    # Compute relative rotations: R_rel = R_t @ R_0^(-1)
    R_rel = np.matmul(R, R_0_inv)  # (T, 3, 3)

    # Convert back to axis-angle
    orient_rel = matrix_to_axis_angle_np(R_rel)  # (T, 3)

    return orient_rel


def resample_sequence(data: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample sequence from source_fps to target_fps using linear interpolation."""
    if abs(source_fps - target_fps) < 0.1:
        return data
    
    T_src = data.shape[0]
    duration = T_src / source_fps
    T_tgt = int(duration * target_fps)
    
    if T_tgt < 2:
        return data[:2]
    
    t_src = np.linspace(0, duration, T_src)
    t_tgt = np.linspace(0, duration, T_tgt)
    
    # Flatten for interpolation, then reshape back
    original_shape = data.shape
    data_flat = data.reshape(T_src, -1)
    
    interp_func = interp1d(t_src, data_flat, axis=0, kind='linear', fill_value='extrapolate')
    resampled = interp_func(t_tgt)
    
    new_shape = (T_tgt,) + original_shape[1:]
    return resampled.reshape(new_shape)


def estimate_contacts_model(
    poses_2d: np.ndarray,
    checkpoint_path: str,
    fps: float = 50.0
) -> np.ndarray:
    """
    Estimate foot contacts using pretrained neural network model.

    Args:
        poses_2d: (T, 17, 2/3) 2D keypoints in H36M format (pixels)
        checkpoint_path: Path to the pretrained model checkpoint
        fps: Frame rate of the sequence (unused, kept for API compatibility)

    Returns:
        contacts: (T, 4) contact probabilities [L_heel, L_toe, R_heel, R_toe]
                  Values are continuous in [0, 1], not binary.
    """
    detector = get_contact_detector(checkpoint_path)

    # Model outputs [L_toe, L_heel, R_toe, R_heel] as probabilities
    # We need [L_heel, L_toe, R_heel, R_toe] to match expected output format
    contacts_model = detector.predict(
        poses_2d[:, :, :2],  # Only use x, y
        return_probabilities=True  # Output probabilities, not binary
    )

    # Reorder: [L_toe, L_heel, R_toe, R_heel] -> [L_heel, L_toe, R_heel, R_toe]
    contacts = contacts_model[:, [1, 0, 3, 2]]

    return contacts


def estimate_contacts(
    poses_2d: np.ndarray,
    checkpoint_path: str,
    fps: float = 50.0,
) -> np.ndarray:
    """
    Estimate foot contacts using model-based approach.

    Args:
        poses_2d: (T, 17, 2/3) 2D keypoints (H36M format)
        checkpoint_path: Path to contact detection model checkpoint (required)
        fps: Frame rate of the sequence

    Returns:
        contacts: (T, 4) contacts [L_heel, L_toe, R_heel, R_toe]
    """
    if not CONTACT_DETECTOR_AVAILABLE:
        raise RuntimeError(
            "contact_detector module not available. "
            "Model-based contact detection is required."
        )

    return estimate_contacts_model(poses_2d, checkpoint_path, fps)


def generate_pseudo_2d_poses(joints_3d: np.ndarray) -> np.ndarray:
    """
    Generate pseudo-2D poses from 3D joints for MotionBERT compatibility.

    Uses orthographic projection (X, Y) normalized to [-1, 1] range.

    Args:
        joints_3d: (T, 17, 3) root-relative joints in meters

    Returns:
        poses_2d: (T, 17, 3) with [x_norm, y_norm, confidence=1.0]
    """
    T, J, _ = joints_3d.shape
    poses_2d = np.zeros((T, J, 3), dtype=np.float32)

    # Use X, Y coordinates (orthographic projection)
    xy = joints_3d[:, :, :2].copy()

    # Normalize to [-1, 1] based on typical human scale (~2m range)
    # Root-relative coords are typically in [-0.5, 0.5] for limbs
    poses_2d[:, :, :2] = xy * 2.0  # Scale to approximately [-1, 1]
    poses_2d[:, :, :2] = np.clip(poses_2d[:, :, :2], -1, 1)
    poses_2d[:, :, 2] = 1.0  # High confidence

    return poses_2d


def generate_pseudo_2d_poses_for_contact(joints_3d: np.ndarray) -> np.ndarray:
    """
    Generate pseudo-2D poses in PIXEL coordinates for contact detection.

    The contact detector model expects pixel coordinates (typically 0-1000 range),
    not normalized [-1, 1] coordinates. This function converts root-relative 3D
    joints (in meters) to pseudo-pixel coordinates.

    Args:
        joints_3d: (T, 17, 3) root-relative joints in meters

    Returns:
        poses_2d: (T, 17, 3) with [x_pixels, y_pixels, confidence=1.0]
                  where coordinates are centered at 500 with ~500px human height
    """
    T, J, _ = joints_3d.shape
    poses_2d = np.zeros((T, J, 3), dtype=np.float32)

    # Use X, Y coordinates (orthographic projection)
    xy = joints_3d[:, :, :2].copy()

    # Convert to pseudo-pixel coordinates:
    # - Root-relative joints are typically in [-0.5, 0.5] meters for limbs
    # - Scale by 500 to get ~500px limb extent (typical human height ~1000px)
    # - Center at (500, 500) to match typical image center
    # - IMPORTANT: Flip Y-axis because contact detector was trained on image coords
    #   where Y increases downward, but 3D coords have Y increasing upward
    poses_2d[:, :, 0] = xy[:, :, 0] * 500 + 500   # X: center at 500
    poses_2d[:, :, 1] = -xy[:, :, 1] * 500 + 500  # Y: FLIP to match image coords
    poses_2d[:, :, 2] = 1.0  # High confidence

    return poses_2d


def extract_windows(
    joints_3d: np.ndarray,
    poses_2d: np.ndarray,
    root_trans: np.ndarray,
    root_orient: np.ndarray,
    contacts: np.ndarray,
    window_size: int,
    stride: int,
) -> List[Dict[str, np.ndarray]]:
    """Extract sliding windows from a sequence."""
    T = joints_3d.shape[0]
    windows = []
    
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        
        w_joints = joints_3d[start:end].copy()
        w_poses_2d = poses_2d[start:end].copy()
        w_trans = root_trans[start:end].copy()
        w_orient = root_orient[start:end].copy()
        w_contacts = contacts[start:end].copy()
        
        # Normalize: frame 0 as reference
        # Translation: simple subtraction (correct for vectors)
        w_trans = w_trans - w_trans[0:1]
        # Orientation: proper rotation composition R_t @ R_0^(-1) (NOT axis-angle subtraction)
        w_orient = make_orient_frame0_relative(w_orient)

        windows.append({
            'joints_3d': w_joints.astype(np.float32),
            'poses_2d': w_poses_2d.astype(np.float32),
            'root_trans': w_trans.astype(np.float32),
            'root_orient': w_orient.astype(np.float32),
            'contacts': w_contacts.astype(np.float32),
        })
    
    return windows


def process_amass_sequence(
    bdata: Dict,
    body_model: BodyModel,
    j_regressor: np.ndarray,
    device: torch.device,
    source_fps: float,
    target_fps: float,
    num_betas: int = 16,
    num_dmpls: int = 8,
    max_frames_per_batch: int = 2000,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Process a single AMASS sequence.

    For long sequences, processes in sub-chunks to avoid GPU/RAM overflow.

    Returns:
        Dictionary with joints_3d, root_trans, root_orient, or None if failed
    """
    T = len(bdata['trans'])
    if T < 10:
        return None

    # Handle missing dmpls
    if 'dmpls' not in bdata or bdata['dmpls'] is None:
        dmpls = np.zeros((T, num_dmpls), dtype=np.float32)
    else:
        dmpls = bdata['dmpls'][:, :num_dmpls]

    # Extract global transform BEFORE running body model
    root_orient_global = bdata['poses'][:, :3].copy()  # (T, 3) axis-angle
    root_trans_global = bdata['trans'].copy()  # (T, 3)

    # Prepare betas (constant across all frames)
    betas_np = bdata['betas'][:num_betas]

    # Process in sub-chunks if sequence is too long
    if T > max_frames_per_batch:
        all_joints = []

        for chunk_start in range(0, T, max_frames_per_batch):
            chunk_end = min(chunk_start + max_frames_per_batch, T)
            chunk_T = chunk_end - chunk_start

            # Prepare chunk data
            body_parms = {
                'root_orient': torch.zeros(chunk_T, 3, device=device),
                'pose_body': torch.tensor(bdata['poses'][chunk_start:chunk_end, 3:66], dtype=torch.float32, device=device),
                'pose_hand': torch.tensor(bdata['poses'][chunk_start:chunk_end, 66:], dtype=torch.float32, device=device) if bdata['poses'].shape[1] > 66 else torch.zeros(chunk_T, 90, device=device),
                'trans': torch.zeros(chunk_T, 3, device=device),
                'betas': torch.tensor(np.repeat(betas_np[np.newaxis], chunk_T, axis=0), dtype=torch.float32, device=device),
                'dmpls': torch.tensor(dmpls[chunk_start:chunk_end], dtype=torch.float32, device=device),
            }

            with torch.no_grad():
                body_output = body_model(**body_parms)
                mesh = body_output.v.cpu().numpy()  # (chunk_T, V, 3)

            # Apply J_regressor
            joints_chunk = np.einsum('jv,tvd->tjd', j_regressor, mesh)  # (chunk_T, 17, 3)
            all_joints.append(joints_chunk)

            # Clear GPU memory
            del body_parms, body_output, mesh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        joints_3d = np.concatenate(all_joints, axis=0)  # (T, 17, 3)
        del all_joints
    else:
        # Process entire sequence at once (original behavior for short sequences)
        body_parms = {
            'root_orient': torch.zeros(T, 3, device=device),
            'pose_body': torch.tensor(bdata['poses'][:, 3:66], dtype=torch.float32, device=device),
            'pose_hand': torch.tensor(bdata['poses'][:, 66:], dtype=torch.float32, device=device) if bdata['poses'].shape[1] > 66 else torch.zeros(T, 90, device=device),
            'trans': torch.zeros(T, 3, device=device),
            'betas': torch.tensor(np.repeat(betas_np[np.newaxis], T, axis=0), dtype=torch.float32, device=device),
            'dmpls': torch.tensor(dmpls, dtype=torch.float32, device=device),
        }

        with torch.no_grad():
            body_output = body_model(**body_parms)

        mesh = body_output.v.cpu().numpy()  # (T, V, 3)
        joints_3d = np.einsum('jv,tvd->tjd', j_regressor, mesh)  # (T, 17, 3)

        del body_parms, body_output, mesh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Make pelvis-centered (should already be close to zero, but ensure it)
    pelvis = joints_3d[:, 0:1, :].copy()
    joints_3d_rel = joints_3d - pelvis

    # Resample if needed
    if abs(source_fps - target_fps) > 0.1:
        joints_3d_rel = resample_sequence(joints_3d_rel, source_fps, target_fps)
        root_trans_global = resample_sequence(root_trans_global, source_fps, target_fps)
        root_orient_global = resample_sequence(root_orient_global, source_fps, target_fps)

    return {
        'joints_3d': joints_3d_rel.astype(np.float32),
        'root_trans': root_trans_global.astype(np.float32),
        'root_orient': root_orient_global.astype(np.float32),
    }


class IncrementalHDF5Writer:
    """
    Writes windows incrementally to HDF5 using resizable datasets.
    This avoids accumulating all data in memory.
    """

    def __init__(
        self,
        output_path: str,
        window_size: int,
        fps: float,
        has_orient: bool = True,
        chunk_cache_size: int = 100,  # Number of windows to buffer before writing
    ):
        self.output_path = output_path
        self.window_size = window_size
        self.fps = fps
        self.has_orient = has_orient
        self.chunk_cache_size = chunk_cache_size

        self.buffer: List[Dict[str, np.ndarray]] = []
        self.total_written = 0
        self.file: Optional[h5py.File] = None

    def open(self):
        """Open the HDF5 file and create resizable datasets."""
        self.file = h5py.File(self.output_path, 'w')

        # Create resizable datasets with chunking for efficient appends
        # maxshape=(None, ...) allows unlimited growth in first dimension
        W = self.window_size

        self.file.create_dataset(
            'joints_3d', shape=(0, W, 17, 3), maxshape=(None, W, 17, 3),
            dtype=np.float32, chunks=(min(100, self.chunk_cache_size), W, 17, 3), compression='gzip'
        )
        self.file.create_dataset(
            'poses_2d', shape=(0, W, 17, 3), maxshape=(None, W, 17, 3),
            dtype=np.float32, chunks=(min(100, self.chunk_cache_size), W, 17, 3), compression='gzip'
        )
        self.file.create_dataset(
            'root_trans', shape=(0, W, 3), maxshape=(None, W, 3),
            dtype=np.float32, chunks=(min(100, self.chunk_cache_size), W, 3), compression='gzip'
        )
        self.file.create_dataset(
            'root_orient', shape=(0, W, 3), maxshape=(None, W, 3),
            dtype=np.float32, chunks=(min(100, self.chunk_cache_size), W, 3), compression='gzip'
        )
        self.file.create_dataset(
            'contacts', shape=(0, W, 4), maxshape=(None, W, 4),
            dtype=np.float32, chunks=(min(100, self.chunk_cache_size), W, 4), compression='gzip'
        )
        self.file.create_dataset(
            'has_orient', shape=(0,), maxshape=(None,),
            dtype=np.float32, chunks=(min(1000, self.chunk_cache_size * 10),), compression='gzip'
        )

        # Set initial attributes
        self.file.attrs['window_size'] = self.window_size
        self.file.attrs['num_joints'] = 17
        self.file.attrs['fps'] = self.fps
        self.file.attrs['has_orient'] = self.has_orient

    def add_windows(self, windows: List[Dict[str, np.ndarray]]):
        """Add windows to buffer and flush if needed."""
        self.buffer.extend(windows)

        if len(self.buffer) >= self.chunk_cache_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered windows to HDF5 and clear buffer."""
        if not self.buffer or self.file is None:
            return

        N = len(self.buffer)

        # Stack buffered windows
        joints_3d = np.stack([w['joints_3d'] for w in self.buffer])
        poses_2d = np.stack([w['poses_2d'] for w in self.buffer])
        root_trans = np.stack([w['root_trans'] for w in self.buffer])
        root_orient = np.stack([w['root_orient'] for w in self.buffer])
        contacts = np.stack([w['contacts'] for w in self.buffer])

        # Sanitize
        joints_3d = replace_non_finite(joints_3d, "joints_3d")
        poses_2d = replace_non_finite(poses_2d, "poses_2d")
        root_trans = replace_non_finite(root_trans, "root_trans")
        root_orient = replace_non_finite(root_orient, "root_orient")
        contacts = replace_non_finite(contacts, "contacts")

        has_orient_mask = np.ones(N, dtype=np.float32) if self.has_orient else np.zeros(N, dtype=np.float32)

        # Resize datasets and append
        old_size = self.total_written
        new_size = old_size + N

        self.file['joints_3d'].resize(new_size, axis=0)
        self.file['poses_2d'].resize(new_size, axis=0)
        self.file['root_trans'].resize(new_size, axis=0)
        self.file['root_orient'].resize(new_size, axis=0)
        self.file['contacts'].resize(new_size, axis=0)
        self.file['has_orient'].resize(new_size, axis=0)

        self.file['joints_3d'][old_size:new_size] = joints_3d
        self.file['poses_2d'][old_size:new_size] = poses_2d
        self.file['root_trans'][old_size:new_size] = root_trans
        self.file['root_orient'][old_size:new_size] = root_orient
        self.file['contacts'][old_size:new_size] = contacts
        self.file['has_orient'][old_size:new_size] = has_orient_mask

        self.total_written = new_size
        self.buffer.clear()

    def close(self):
        """Flush remaining buffer and close file."""
        if self.file is None:
            return

        # Flush any remaining data
        self._flush_buffer()

        # Update final count
        self.file.attrs['num_samples'] = self.total_written

        print(f"Saved {self.total_written} windows to {self.output_path}")
        if self.total_written > 0:
            # Print stats from first few samples
            print(f"  joints_3d: {self.file['joints_3d'].shape}")
            print(f"  root_trans: {self.file['root_trans'].shape}")
            print(f"  has_orient: {self.has_orient}")

        self.file.close()
        self.file = None


def create_split_hdf5(
    source_path: str,
    output_path: str,
    indices: np.ndarray,
    window_size: int,
    fps: float,
    has_orient: bool = True,
):
    """
    Create a new HDF5 file containing only the samples at the given indices.
    Optimized: reads contiguous blocks and selects in memory (much faster than fancy indexing).
    """
    from tqdm import tqdm

    with h5py.File(source_path, 'r') as src, h5py.File(output_path, 'w') as dst:
        N = len(indices)
        W = window_size
        total_src = src['joints_3d'].shape[0]

        # Create output datasets WITHOUT compression for faster writing
        # (use prepare_streaming_data.py later if streaming is needed)
        dst.create_dataset('joints_3d', shape=(N, W, 17, 3), dtype=np.float32)
        dst.create_dataset('poses_2d', shape=(N, W, 17, 3), dtype=np.float32)
        dst.create_dataset('root_trans', shape=(N, W, 3), dtype=np.float32)
        dst.create_dataset('root_orient', shape=(N, W, 3), dtype=np.float32)
        dst.create_dataset('contacts', shape=(N, W, 4), dtype=np.float32)
        dst.create_dataset('has_orient', shape=(N,), dtype=np.float32)

        # Create a mapping: source_idx -> list of (dest_idx)
        # This lets us read source data contiguously and scatter to destinations
        idx_set = set(indices)

        # Read source in contiguous blocks (MUCH faster than fancy indexing on compressed data)
        block_size = 5000  # Read this many contiguous samples at a time
        dest_idx_map = {src_idx: dest_idx for dest_idx, src_idx in enumerate(indices)}

        written = 0
        pbar = tqdm(total=N, desc="Copying samples")

        for block_start in range(0, total_src, block_size):
            block_end = min(block_start + block_size, total_src)

            # Find which indices in this block we need
            needed_in_block = []
            for src_idx in range(block_start, block_end):
                if src_idx in idx_set:
                    needed_in_block.append((src_idx, dest_idx_map[src_idx]))

            if not needed_in_block:
                continue

            # Read the entire block (contiguous read - fast even with compression)
            block_joints = src['joints_3d'][block_start:block_end]
            block_poses = src['poses_2d'][block_start:block_end]
            block_trans = src['root_trans'][block_start:block_end]
            block_orient = src['root_orient'][block_start:block_end]
            block_contacts = src['contacts'][block_start:block_end]
            block_has_orient = src['has_orient'][block_start:block_end]

            # Write needed samples to their destination positions
            for src_idx, dest_idx in needed_in_block:
                local_idx = src_idx - block_start
                dst['joints_3d'][dest_idx] = block_joints[local_idx]
                dst['poses_2d'][dest_idx] = block_poses[local_idx]
                dst['root_trans'][dest_idx] = block_trans[local_idx]
                dst['root_orient'][dest_idx] = block_orient[local_idx]
                dst['contacts'][dest_idx] = block_contacts[local_idx]
                dst['has_orient'][dest_idx] = block_has_orient[local_idx]

            written += len(needed_in_block)
            pbar.update(len(needed_in_block))

        pbar.close()

        # Set attributes
        dst.attrs['num_samples'] = N
        dst.attrs['window_size'] = window_size
        dst.attrs['num_joints'] = 17
        dst.attrs['fps'] = fps
        dst.attrs['has_orient'] = has_orient

    print(f"Created {output_path} with {N} samples")


def main():
    parser = argparse.ArgumentParser(description='Convert AMASS to HDF5 for RootNet')
    parser.add_argument('--motions_pkl', required=True, help='Path to all_motions_fps60.pkl')
    parser.add_argument('--fps_csv', default=None, help='Path to fps.csv (optional, for filenames)')
    parser.add_argument('--j_regressor', required=True, help='Path to J_regressor_h36m_correct.npy')
    parser.add_argument('--body_models_dir', required=True, help='Directory containing smplh/ and dmpls/')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--contact_checkpoint', type=str, required=True,
                        help='Path to contact detection model checkpoint (.pth). Required.')
    parser.add_argument('--source_fps', type=float, default=60.0, help='Source FPS of AMASS data')
    parser.add_argument('--target_fps', type=float, default=50.0, help='Target FPS for output')
    parser.add_argument('--window_size', type=int, default=243, help='Window size in frames')
    parser.add_argument('--stride', type=int, default=81, help='Stride between windows')
    parser.add_argument('--min_seq_length', type=int, default=50, help='Min sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Train/val split ratio')
    parser.add_argument('--max_sequences', type=int, default=None, help='Max sequences to process (for debugging)')
    parser.add_argument('--chunk_size', type=int, default=500, help='Number of sequences to process before flushing to disk (memory optimization)')
    parser.add_argument('--max_frames_per_batch', type=int, default=2000, help='Max frames to process at once through body model (for long sequences)')
    args = parser.parse_args()

    # Validate contact checkpoint (required)
    contact_ckpt_path = Path(args.contact_checkpoint)
    if not contact_ckpt_path.exists():
        raise FileNotFoundError(f"Contact checkpoint not found: {args.contact_checkpoint}")
    if not CONTACT_DETECTOR_AVAILABLE:
        raise RuntimeError(
            "contact_detector module not available. "
            "Model-based contact detection is required."
        )
    print(f"Using model-based contact detection: {args.contact_checkpoint}")

    # Load data
    print(f"Loading {args.motions_pkl}...")
    with open(args.motions_pkl, 'rb') as f:
        motion_data = pickle.load(f)
    print(f"Loaded {len(motion_data)} sequences")
    
    # Load J_regressor
    j_regressor = np.load(args.j_regressor)
    print(f"J_regressor shape: {j_regressor.shape}")
    
    # Setup device and body models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Pre-load body models for all genders
    num_betas = 16
    num_dmpls = 8
    body_models = {}
    for gender in ['male', 'female', 'neutral']:
        bm_path = osp.join(args.body_models_dir, f'smplh/{gender}/model.npz')
        dmpl_path = osp.join(args.body_models_dir, f'dmpls/{gender}/model.npz')
        if osp.exists(bm_path):
            body_models[gender] = BodyModel(
                bm_fname=bm_path,
                num_betas=num_betas,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_path if osp.exists(dmpl_path) else None
            ).to(device)
            print(f"Loaded {gender} body model from {bm_path}")

    print(f"Loaded body models for: {list(body_models.keys())}")
    
    # Setup output directory and incremental writer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary file for all windows, then split into train/val
    temp_all_path = str(output_dir / '_amass_all_temp.h5')

    # Create incremental HDF5 writer
    writer = IncrementalHDF5Writer(
        output_path=temp_all_path,
        window_size=args.window_size,
        fps=args.target_fps,
        has_orient=True,
        chunk_cache_size=100,  # Flush every 100 windows
    )
    writer.open()

    # Process sequences in chunks to manage memory
    num_to_process = len(motion_data) if args.max_sequences is None else min(args.max_sequences, len(motion_data))
    chunk_size = args.chunk_size

    print(f"Processing {num_to_process} sequences in chunks of {chunk_size}...")

    total_windows = 0
    sequences_processed = 0

    for chunk_start in range(0, num_to_process, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_to_process)
        chunk_data = motion_data[chunk_start:chunk_end]

        chunk_windows = []

        for i, bdata in enumerate(tqdm(chunk_data, desc=f"Chunk {chunk_start//chunk_size + 1}/{(num_to_process + chunk_size - 1)//chunk_size}")):
            seq_idx = chunk_start + i

            # Determine gender
            gender = str(bdata.get('gender', 'neutral')).lower()
            if gender not in body_models:
                # Fallback: prefer neutral, then female
                gender = 'neutral' if 'neutral' in body_models else 'female'
            if gender not in body_models:
                print(f"Warning: No body model available for sequence {seq_idx}, skipping")
                continue

            bm = body_models[gender]

            # Process sequence
            try:
                result = process_amass_sequence(
                    bdata, bm, j_regressor, device,
                    source_fps=args.source_fps,
                    target_fps=args.target_fps,
                    num_betas=num_betas,
                    num_dmpls=num_dmpls,
                    max_frames_per_batch=args.max_frames_per_batch,
                )
            except Exception as e:
                print(f"Warning: Failed to process sequence {seq_idx}: {e}")
                continue

            if result is None:
                continue

            T = result['joints_3d'].shape[0]
            if T < args.min_seq_length:
                continue

            # Generate pseudo-2D poses for MotionBERT (normalized [-1, 1])
            poses_2d = generate_pseudo_2d_poses(result['joints_3d'])

            # Generate pseudo-2D poses in pixel coordinates for contact detection
            poses_2d_for_contact = generate_pseudo_2d_poses_for_contact(result['joints_3d'])

            # Estimate contacts using model-based detection (requires pixel coordinates)
            contacts = estimate_contacts(
                poses_2d=poses_2d_for_contact,
                checkpoint_path=args.contact_checkpoint,
                fps=args.target_fps,
            )

            # Extract windows
            windows = extract_windows(
                result['joints_3d'],
                poses_2d,
                result['root_trans'],
                result['root_orient'],
                contacts,
                args.window_size,
                args.stride,
            )
            chunk_windows.extend(windows)
            sequences_processed += 1

        # Write chunk windows to HDF5 incrementally
        if chunk_windows:
            writer.add_windows(chunk_windows)
            total_windows += len(chunk_windows)
            print(f"  Chunk complete: {len(chunk_windows)} windows extracted (total: {total_windows})")

        # Clear chunk data from memory
        del chunk_windows
        del chunk_data
        gc.collect()

        # Also clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Close the writer (flushes remaining buffer)
    writer.close()

    print(f"\nTotal windows extracted: {total_windows}")
    print(f"Sequences processed: {sequences_processed}")

    if total_windows == 0:
        print("No windows extracted! Check input data.")
        # Clean up temp file
        if os.path.exists(temp_all_path):
            os.remove(temp_all_path)
        return

    # Split into train/val by shuffling indices and creating new files
    print("\nCreating train/val split...")
    np.random.seed(42)
    indices = np.random.permutation(total_windows)
    split_idx = int(len(indices) * args.train_ratio)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_path = str(output_dir / 'amass_train.h5')
    val_path = str(output_dir / 'amass_val.h5')

    print(f"Creating training set with {len(train_indices)} samples...")
    create_split_hdf5(
        source_path=temp_all_path,
        output_path=train_path,
        indices=train_indices,
        window_size=args.window_size,
        fps=args.target_fps,
        has_orient=True,
    )

    print(f"Creating validation set with {len(val_indices)} samples...")
    create_split_hdf5(
        source_path=temp_all_path,
        output_path=val_path,
        indices=val_indices,
        window_size=args.window_size,
        fps=args.target_fps,
        has_orient=True,
    )

    # Clean up temporary file
    if os.path.exists(temp_all_path):
        os.remove(temp_all_path)
        print(f"Cleaned up temporary file: {temp_all_path}")

    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Train: {len(train_indices)} windows -> {train_path}")
    print(f"Val: {len(val_indices)} windows -> {val_path}")
    print(f"FPS: {args.target_fps}")
    print(f"has_orient: True (AMASS has ground truth orientation)")
    print(f"Contact detection: Model-based ({args.contact_checkpoint})")


if __name__ == '__main__':
    main()