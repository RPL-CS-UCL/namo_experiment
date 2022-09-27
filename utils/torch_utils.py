
from isaacgym.torch_utils import *
from typing import Dict, Any, Tuple
import torch


def to_torch(x, dtype=torch.float, device='cpu', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def local_to_world_space(pos_offset_local: torch.Tensor, pose_global: torch.Tensor):
    """ Convert a point from the local frame to the global frame
    Args:
        pos_offset_local: Point in local frame. Shape: [N, 3]
        pose_global: The spatial pose of this point. Shape: [N, 7]
    Returns:
        Position in the global frame. Shape: [N, 3]
    """
    quat_pos_local = torch.cat(
        [pos_offset_local, torch.zeros(
            pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device)],
        dim=-1
    )
    quat_global = pose_global[:, 3:7]
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(quat_global, quat_mul(
        quat_pos_local, quat_global_conj))[:, 0:3]

    result_pos_gloal = pos_offset_global + pose_global[:, 0:3]

    return result_pos_gloal


@torch.jit.script
def gen_keypoints(pose: torch.Tensor, num_keypoints: int = 8, size: Tuple[float, float, float] = (0.6, 0.6, 0.6)):

    num_envs = pose.shape[0]

    keypoints_buf = torch.ones(
        num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)

    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32,
                              device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf


def process_box_state(box_state):
    '''
    transforms box state read from sensor to gym state
    '''
    out_buf = torch.zeros((box_state.shape[0], 13), dtype=torch.float32)
    up_vec = to_torch([0.0, 0.0, 1.0])

    out_buf[:, 0] = box_state[:, 0]
    out_buf[:, 1] = box_state[:, 1]
    out_buf[:, 2] = 0.3
    out_buf[:, 3:7] = quat_from_angle_axis(box_state[:, 2], up_vec)
    out_buf[:, 7:] = 0

    return out_buf
