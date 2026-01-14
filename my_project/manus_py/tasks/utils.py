from isaacgymenvs.utils.torch_jit_utils import *
import torch
import os
import numpy as np
import trimesh
import open3d as o3d
import queue
from queue import Empty

def batch_linear_interpolate_poses(
    pose1: torch.Tensor,  # Shape: [B, 7] (x, y, z, qx, qy, qz, qw)
    pose2: torch.Tensor,  # Shape: [B, 7]
    max_trans_step: float,
    max_rot_step: float,
):
    """Batch interpolate between poses with limits on translation/rotation steps.
    
    Args:
        pose1: Starting poses of shape [B, 7]
        pose2: Target poses of shape [B, 7]
        max_trans_step: Maximum translation step between consecutive poses
        max_rot_step: Maximum rotation step (in radians) between consecutive poses
        
    Returns:
        interp_poses: Interpolated poses of shape [B, T_max, 7]
        timesteps: Actual lengths of each sequence in the batch [B]
    """
    B = pose1.shape[0]
    device = pose1.device
    
    # Split into positions and quaternions
    p1, q1 = pose1[:, :3], pose1[:, 3:]  # [B, 3], [B, 4]
    p2, q2 = pose2[:, :3], pose2[:, 3:]  # [B, 3], [B, 4]
    
    # --- Compute required steps for each pair ---
    delta_p = p2 - p1  # [B, 3]
    trans_dist = torch.norm(delta_p, dim=1)  # [B]
    n_trans = torch.ceil(trans_dist / max_trans_step).long().clamp(min=1)  # [B]
    
    theta = quat_diff_rad(q1, q2)  # [B]
    n_rot = torch.ceil(theta / max_rot_step).long().clamp(min=1)  # [B]
    
    # print(n_trans, n_rot)
    n = torch.maximum(n_trans, n_rot)  # [B]
    T_max = n.max().item()
    timesteps = n  # [B]
    
    # Create mask for valid steps [B, T_max+1]
    step_idx = torch.arange(T_max + 1, device=device).expand(B, -1)  # [B, T_max+1]
    valid_mask = step_idx <= n.unsqueeze(1)  # [B, T_max+1]
    
    # Compute interpolation factors t [B, T_max+1]
    t = step_idx.float() / n.unsqueeze(1).clamp(min=1)  # [B, T_max+1]
    t = t * valid_mask.float()  # Zero out invalid steps
    
    # Interpolate positions (LERP) [B, T_max+1, 3]
    interp_p = p1.unsqueeze(1) + t.unsqueeze(-1) * delta_p.unsqueeze(1)
    
    # --- Vectorized SLERP implementation ---
    interp_q = slerp(
        q1.unsqueeze(1).repeat(1,T_max+1,1), 
        q2.unsqueeze(1).repeat(1,T_max+1,1),
        t.unsqueeze(-1)
    ) # [B, T_max+1, 4]

    # Combine into poses [B, T_max+1, 7]
    interp_poses = torch.cat([interp_p, interp_q], dim=-1)
    
    return interp_poses, timesteps


COLORS_DICT = {
    # 基础颜色
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    
    # 复合颜色
    "yellow": [1.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    
    # 灰度色
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "gray": [0.5, 0.5, 0.5],
    "light_gray": [0.75, 0.75, 0.75],
    "dark_gray": [0.25, 0.25, 0.25],
    
    # 常见颜色
    "orange": [1.0, 0.65, 0.0],
    "purple": [0.5, 0.0, 0.5],
    "pink": [1.0, 0.75, 0.8],
    "brown": [0.65, 0.16, 0.16],
    "olive": [0.5, 0.5, 0.0],
    "teal": [0.0, 0.5, 0.5],
    "navy": [0.0, 0.0, 0.5],
    "maroon": [0.5, 0.0, 0.0],
    "lime": [0.75, 1.0, 0.0],
    
    # 金属色
    "gold": [1.0, 0.84, 0.0],
    "silver": [0.75, 0.75, 0.75],
    "bronze": [0.8, 0.5, 0.2],
    
    # 自然色
    "sky_blue": [0.53, 0.81, 0.92],
    "forest_green": [0.13, 0.55, 0.13],
    "violet": [0.93, 0.51, 0.93],
    "coral": [1.0, 0.5, 0.31],
    "salmon": [0.98, 0.5, 0.45],
    "turquoise": [0.25, 0.88, 0.82],
    "indigo": [0.29, 0.0, 0.51],
    "beige": [0.96, 0.96, 0.86],
    "ivory": [1.0, 1.0, 0.94]
}

import matplotlib.pyplot as plt
def vis_success(success_per_object, save_fn, title):
    objects = [item[0] for item in success_per_object]
    success_rates = [float(item[1]) for item in success_per_object]
    #print(objects, success_rates)
    plt.cla()
    plt.figure(figsize=(30, 6))
    plt.bar(objects, success_rates, color='skyblue')
    #plt.xlabel('Object')
    plt.ylabel('Success Rate')
    plt.title(title)
    plt.ylim(0, 1)  # 设定纵轴的范围在0到1之间
    plt.xticks(rotation=60)  # 如果物体名称较长，可以旋转横轴标签
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(save_fn)



#################### point cloud for isaac
def load_object_point_clouds(object_files, asset_root):
    ret = []
    for fn in object_files:
        substrs = fn.split('/')
        assert len(substrs)==3, f"Filename should be ObjDatasetName/urdf/xxx.urdf, got {fn}"
        pc_fn = os.path.join(substrs[0], 'pointclouds', substrs[-1].replace('.urdf','.npy'))
        print("object file: {} -> pcl file: {}".format(fn, pc_fn))
        pc = np.load(os.path.join(asset_root, pc_fn))
        #pc = np.load("vision/real_pcl.npy")
        ret.append(pc)
    return ret

@torch.jit.script
def transform_points(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    if len(pt_new.size()) == 3:
        return pt_new[:,:,:3]
    elif len(pt_new.size()) == 2:
        return pt_new[:,:3]

# def get_pointcloud_from_mesh(mesh_dir, filename, num_sample=4096):
#     all_points = []
#     mesh = trimesh.load_mesh(os.path.join(mesh_dir, filename))
#     #points = mesh.sample(num_sample)
#     points = mesh.vertices
#     points_torch = torch.tensor(points, dtype=torch.float32).to('cuda').unsqueeze(0)
#     indices = farthest_point_sample(points_torch, 512, 'cuda')
#     indices = indices[0].cpu().numpy()
#     points = points[indices]
#     return points


def get_pointcloud_from_mesh(mesh_dir, filename, num_sample=4096, device='cuda'):
    mesh = trimesh.load_mesh(os.path.join(mesh_dir, filename), process=True)
    if isinstance(mesh, trimesh.Scene):
        # 有些OBJ会被解析成Scene，合并成一个Trimesh
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    # 修法线/退化面，提升采样稳定性
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)
    except Exception:
        pass

    # 1) 先在“表面”按面积采样大量点（过采样），避免只在稀疏顶点
    #   - sample_surface返回(pts, face_idx)
    #   - 过采样倍率可以按需改大一些（比如8~16倍）
    oversample_ratio = 8
    target_oversample = max(num_sample * oversample_ratio, 20000)
    surf_pts, _ = trimesh.sample.sample_surface(mesh, target_oversample)

    # 2) （可选）再沿“边”按长度少量采样，帮助保轮廓与锐边（特别是有棱的模型）
    #    对圆柱侧面帮助不大，但对有明显折线的物体更稳
    try:
        edges = mesh.edges_unique
        v = mesh.vertices
        edge_vec = v[edges[:, 1]] - v[edges[:, 0]]
        edge_len = np.linalg.norm(edge_vec, axis=1)
        total_len = edge_len.sum() + 1e-8
        edge_budget = max(num_sample // 2, 1024)  # 控制边采样量，不要太多
        # 每条边按长度占比分配样本数
        edge_counts = np.maximum(
            np.floor(edge_budget * (edge_len / total_len)).astype(int), 0
        )
        # 采样
        edge_pts_list = []
        for (a, b), k in zip(edges, edge_counts):
            if k <= 0:
                continue
            t = np.random.rand(k, 1)
            pts = v[a][None, :] * (1.0 - t) + v[b][None, :] * t
            edge_pts_list.append(pts)
        if edge_pts_list:
            edge_pts = np.concatenate(edge_pts_list, axis=0)
            pts_all = np.concatenate([surf_pts, edge_pts], axis=0)
        else:
            pts_all = surf_pts
    except Exception:
        pts_all = surf_pts

    # 3) 用FPS从过采样点里选出num_sample个代表点
    pts_torch = torch.from_numpy(pts_all.astype(np.float32)).to(device).unsqueeze(0)
    # 保护：若过采样点仍不足，直接补齐/去重
    if pts_torch.shape[1] < num_sample:
        # 重复一些点补齐，尽量不报错
        repeat_factor = int(np.ceil(num_sample / pts_torch.shape[1]))
        pts_torch = pts_torch.repeat(1, repeat_factor, 1)[:, :num_sample, :]

    # 你的farthest_point_sample应返回[B, npoints]的索引
    idx = farthest_point_sample(pts_torch, num_sample, device)[0].cpu().numpy()
    points = pts_all[idx]

    return points


def farthest_point_sample(xyz, npoint, device, init=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if init is not None:
        farthest = torch.tensor(init).long().reshape(B).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx, device):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.size()[0]
    view_shape = list(idx.size())
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.size())
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

# visualize static point cloud
def vis_pointcloud(pc, add_coordinate_frame=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc)
    vis.add_geometry(pointcloud)
    if add_coordinate_frame:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
    while True:
        vis.poll_events()
        vis.update_renderer()

# realtime visualize point cloud with threading
# pc_queue: queue.Queue(1)
def vis_pointcloud_realtime(pc_queue, add_coordinate_frame=True, zoom=1, coord_len=0.5):
    def vis_pointcloud_realtime_thread(q, add_coordinate_frame=True):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600)
        pointcloud = o3d.geometry.PointCloud()
        vis.add_geometry(pointcloud)
        # draw coordinate frame
        if add_coordinate_frame:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_len, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)
        # adjust view point
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])  # 观察点设置为世界坐标原点
        view_control.set_front([1, 0.2, 0.4])  # 前视方向
        view_control.set_up([0, 0, 1])  # 上方向设置为Z轴方向
        view_control.set_zoom(zoom)  # 设置缩放比例以适应场景
        while True:
            try:
                pcd = q.get()
                pointcloud.points = o3d.utility.Vector3dVector(pcd)
                vis.update_geometry(pointcloud)
                vis.poll_events()
                vis.update_renderer()
            except Empty:
                continue

    import threading
    thread = threading.Thread(target=vis_pointcloud_realtime_thread, args=(pc_queue,))
    thread.daemon = True
    thread.start()


if __name__=="__main__":
    #pc = get_pointcloud_from_mesh('../assets/union_ycb_unidex/meshes', 
    #                               '017_orange.stl',
    #                               num_sample=512)
    #print(pc)
    import time

    dataset_name = 'temp'

    fns = os.listdir(f'../assets/{dataset_name}/pointclouds')
    num = len(fns)

    Q = queue.Queue(1)
    vis_pointcloud_realtime(Q, coord_len=0.1)
    for t in range(10000000):
        if t%100==0:
            idx = np.random.randint(num)
            pc = np.load(os.path.join(f'../assets/{dataset_name}/pointclouds', fns[idx]))
            print(fns[idx], pc.shape, pc.dtype)
        Q.put(pc)
        time.sleep(0.01)
