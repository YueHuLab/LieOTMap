import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import os
import datetime
import mrcfile
import re

# --- Atom Parsers ---
def parse_structure_file(file_path):
    """Robustly parses a PDB or CIF file to get structured atom data."""
    atom_data = []
    try:
        with open(file_path, 'r') as f:
            file_lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 结构文件未找到于 {file_path}"); return None

    is_cif = file_path.lower().endswith('.cif')
    if is_cif:
        header, data_lines = [], []
        in_loop, header_done = False, False
        for line in file_lines:
            s = line.strip()
            if s == 'loop_': in_loop = True; header_done = False; header = []
            elif s.startswith('_atom_site.') and in_loop: header.append(s)
            elif in_loop and s and not s.startswith('#') and not s.startswith('_'):
                header_done = True; data_lines.append(s)
        
        if not header or not data_lines: return None
        col_map = {name: i for i, name in enumerate(header)}
        x_col, y_col, z_col = col_map.get('_atom_site.Cartn_x'), col_map.get('_atom_site.Cartn_y'), col_map.get('_atom_site.Cartn_z')
        atom_col, chain_col = col_map.get('_atom_site.label_atom_id'), col_map.get('_atom_site.auth_asym_id')
        res_seq_col, res_name_col = col_map.get('_atom_site.auth_seq_id'), col_map.get('_atom_site.label_comp_id')
        if any(c is None for c in [x_col, y_col, z_col, atom_col, chain_col, res_seq_col, res_name_col]): return None
        
        for line in data_lines:
            try:
                parts = line.split()
                atom_data.append({
                    'chain': parts[chain_col], 'res_seq': int(parts[res_seq_col]), 'res_name': parts[res_name_col],
                    'atom_name': parts[atom_col].strip('"'), 'coords': [float(parts[x_col]), float(parts[y_col]), float(parts[z_col])]
                })
            except (ValueError, IndexError): continue
    else: # PDB
        for line in file_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    atom_data.append({
                        'chain': line[21], 'res_seq': int(line[22:26]), 'res_name': line[17:20].strip(),
                        'atom_name': line[12:16].strip(), 'coords': [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    })
                except (ValueError, IndexError): continue
    
    print(f"从 '{os.path.basename(file_path)}' 解析了 {len(atom_data)} 个原子。")
    if not atom_data: return None
    return atom_data

# --- Map Processing & Other Logic ---
def parse_mrc(file_path):
    try:
        with mrcfile.open(file_path, permissive=True) as mrc: 
            native_dtype = mrc.data.dtype.newbyteorder('='); data = mrc.data.astype(native_dtype)
            map_data = torch.tensor(data, dtype=torch.float32); hdr = mrc.header
            if not (hdr.mx>0 and hdr.my>0 and hdr.mz>0 and hdr.cella.x>0 and hdr.cella.y>0 and hdr.cella.z>0): return None, None, None
            vx=float(hdr.cella.x)/float(hdr.mx); vy=float(hdr.cella.y)/float(hdr.my); vz=float(hdr.cella.z)/float(hdr.mz)
            voxel_size = torch.tensor([vx, vy, vz], dtype=torch.float32)
            origin = torch.tensor([float(hdr.origin.x), float(hdr.origin.y), float(hdr.origin.z)], dtype=torch.float32)
            print(f"密度图 '{os.path.basename(file_path)}' 已加载。 维度: {map_data.shape}, 体素大小: {voxel_size.numpy()} Å, 文件头原点: {origin.numpy()} Å")
            return map_data, voxel_size, origin
    except Exception as e: print(f"解析MRC文件时发生意外错误: {e}"); return None, None, None

def get_sigma_threshold_points(map_data, voxel_size, origin, sigma_level=3.0, max_points=40000, output_pdb_path=None):
    """Generates map points by selecting all voxels above a given sigma level."""
    print(f"正在选取所有高于 {sigma_level}-sigma 水平的密度点...")
    print("  - Sigma值将仅基于图中的正密度值计算。  ")

    positive_densities = map_data[map_data > 0]
    if positive_densities.numel() == 0:
        print("警告: 密度图中找不到任何正值。")
        return torch.empty((0, 3))

    map_mean = positive_densities.mean()
    map_std = positive_densities.std()
    density_threshold = map_mean + sigma_level * map_std
    print(f"  - 正密度均值: {map_mean:.4f}, 正密度标准差 (sigma): {map_std:.4f}")
    print(f"  - {sigma_level}-sigma 阈值: {density_threshold:.4f}")

    mask = map_data > density_threshold
    candidate_indices = mask.nonzero(as_tuple=False)
    
    if candidate_indices.shape[0] == 0:
        print(f"警告: 找不到任何密度高于 {sigma_level}-sigma 水平的点。")
        return torch.empty((0, 3))
    
    print(f"  - 找到 {candidate_indices.shape[0]} 个高于阈值的候选点。")
    
    candidate_densities = map_data[candidate_indices[:, 0], candidate_indices[:, 1], candidate_indices[:, 2]]

    if candidate_indices.shape[0] > max_points:
        print(f"  - 候选点数量超过 {max_points}，将选取密度最高的 {max_points} 个点。")
        top_k_indices = torch.topk(candidate_densities, k=max_points).indices
        final_indices = candidate_indices[top_k_indices]
    else:
        final_indices = candidate_indices

    # Correct the order of coordinates from (z, y, x) to (x, y, z) before calculation
    final_indices_xyz = torch.flip(final_indices, dims=[1])
    final_points_angstrom = final_indices_xyz.float() * voxel_size + origin
    print(f"最终使用 {final_points_angstrom.shape[0]} 个点作为目标点云。")

    if output_pdb_path:
        print(f"  - 正在将目标点云写入PDB文件: {output_pdb_path}")
        with open(output_pdb_path, 'w') as f:
            for i, point in enumerate(final_points_angstrom):
                x, y, z = point
                f.write(f"HETATM{i+1:5d}  C   PTS A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
    
    return final_points_angstrom

# --- Core Algorithm ---
def get_transformation_matrix(params):
    w, u = params[:3], params[3:]; W = torch.zeros((3,3),dtype=params.dtype,device=params.device); W[0,1],W[0,2]=-w[2],w[1]; W[1,0],W[1,2]=w[2],-w[0]; W[2,0],W[2,1]=-w[1],w[0]
    return torch.linalg.matrix_exp(W), u

def get_d0(length):
    return 1.24 * (length - 15)**(1/3) - 1.8 if length > 15 else 0.5

def differentiable_tm_score(coords_a, coords_b, d0):
    """Calculates a differentiable TM-score."""
    d_ij_sq = torch.sum((coords_a.unsqueeze(1) - coords_b.unsqueeze(0))**2, dim=2)
    s_ij = 1.0 / (1.0 + d_ij_sq / (d0**2))
    alignment_probs = torch.softmax(-d_ij_sq, dim=1)
    return torch.mean(torch.sum(alignment_probs * s_ij, dim=1))

def calculate_rmsd(c1, c2): 
    return torch.sqrt(torch.mean(torch.sum((c1 - c2)**2, dim=1)))

# --- Main Execution ---
def main():
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="使用Sinkhorn/TM-score进行多阶段拟合的最终测试版本。")
    parser.add_argument("--mobile_structure", required=True, help="需要拟合的移动结构文件 (.cif 或 .pdb)。")
    parser.add_argument("--target_map", required=True, help="目标Cryo-EM密度图文件 (.mrc)。")
    parser.add_argument("--gold_standard_structure", required=True, help="用于计算最终RMSD的金标准结构文件。")
    parser.add_argument("--output", default=None, help="保存拟合后的PDB结构的路径。")
    parser.add_argument("--sigma_level", type=float, default=3.0, help="用于筛选候选点的Sigma水平。")
    parser.add_argument("--max_points", type=int, default=40000, help="使用的最大点云数。")
    parser.add_argument("--output_points_pdb", type=str, default=None, help="将生成的目标点云保存为PDB文件的路径。")
    parser.add_argument("--lr", type=float, default=0.15, help="优化学习率。")
    parser.add_argument("--steps", type=int, default=100, help="优化步数。")
    args = parser.parse_args()

    print(f"\n--- Cryo-EM 拟合程序 (Sinkhorn 多阶段最终测试) ---\n程序开始于: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    mobile_data = parse_structure_file(args.mobile_structure)
    gold_data = parse_structure_file(args.gold_standard_structure)
    if not mobile_data or not gold_data: print("错误: 结构文件解析失败。"); return

    all_mobile_coords = torch.tensor([a['coords'] for a in mobile_data], dtype=torch.float32)
    mobile_ca_coords = torch.tensor([a['coords'] for a in mobile_data if a['atom_name'] == 'CA'], dtype=torch.float32)
    
    mobile_ca_map = {(a['chain'], a['res_seq']): a['coords'] for a in mobile_data if a['atom_name'] == 'CA'}
    gold_ca_map = {(a['chain'], a['res_seq']): a['coords'] for a in gold_data if a['atom_name'] == 'CA'}
    common_keys = sorted(list(mobile_ca_map.keys() & gold_ca_map.keys()))
    print(f"在两个结构中找到了 {len(common_keys)} 个共同的C-alpha原子用于计算最终RMSD。")
    if not common_keys: print("错误: 找不到任何共同的C-alpha原子!"); return
    common_mobile_ca_coords = torch.tensor([mobile_ca_map[k] for k in common_keys], dtype=torch.float32)
    common_gold_ca_coords = torch.tensor([gold_ca_map[k] for k in common_keys], dtype=torch.float32)

    map_data, voxel_size, origin_from_header = parse_mrc(args.target_map)
    if map_data is None: return

    map_points_angstrom = get_sigma_threshold_points(map_data, voxel_size, origin_from_header, 
                                                     sigma_level=args.sigma_level, 
                                                     max_points=args.max_points, 
                                                     output_pdb_path=args.output_points_pdb)

    if map_points_angstrom.shape[0] == 0:
        print("错误: 点云生成失败，无法继续。")
        return

    # --- Pre-alignment ---
    map_com_angstrom = map_points_angstrom.mean(dim=0)
    struct_center = all_mobile_coords.mean(dim=0)
    initial_t = map_com_angstrom - struct_center
    print(f"--- 预对齐 (结构质心 -> 点云质心) ---\n  - 结构质心 (Å): {struct_center.numpy()}\n  - 点云质心 (Å): {map_com_angstrom.numpy()}\n  - 计算得到的初始平移 (Å): {initial_t.numpy()}")

    transform_params = torch.zeros(6, requires_grad=True)
    with torch.no_grad(): transform_params[3:] = initial_t
    
    d0 = get_d0(len(map_points_angstrom))

    stages = [
        {'name': 'CA-only', 'coords': mobile_ca_coords, 'steps': args.steps, 'lr': args.lr}
    ]

    for stage in stages:
        print(f"\n--- 正在执行阶段: {stage['name']} ({stage['steps']} 步, lr={stage['lr']}) ---")
        optimizer = optim.Adam([transform_params], lr=stage['lr'])
        for step in range(stage['steps']):
            optimizer.zero_grad()
            R, t = get_transformation_matrix(transform_params)
            
            transformed_coords = (R @ stage['coords'].T).T + t
            score = differentiable_tm_score(transformed_coords, map_points_angstrom, d0)
            loss = -score

            if step % 10 == 0 or step == stage['steps'] - 1:
                with torch.no_grad():
                    transformed_common_ca = (R @ common_mobile_ca_coords.T).T + t
                    current_rmsd = calculate_rmsd(transformed_common_ca, common_gold_ca_coords)
                print(f"  步骤 {step:04d}: TM-score = {score.item():.6f}, RMSD = {current_rmsd.item():.4f} Å")
            
            loss.backward(); optimizer.step()

    print("\n--- 所有阶段优化完成 ---")
    R_final, t_final = get_transformation_matrix(transform_params.detach().clone())
    
    final_coords = (R_final @ all_mobile_coords.T).T + t_final
    transformed_common_mobile_coords = (R_final @ common_mobile_ca_coords.T).T + t_final
    final_rmsd = calculate_rmsd(transformed_common_mobile_coords, common_gold_ca_coords)
    print(f"\n最终RMSD (对比 '{os.path.basename(args.gold_standard_structure)}' 的CA原子): {final_rmsd.item():.4f} Å")

    output_filename = args.output if args.output else f"{os.path.splitext(os.path.basename(args.mobile_structure))[0]}_sinkhorn_final_rmsd_{final_rmsd.item():.2f}.pdb"
    print(f"\n--- 正在将拟合后的PDB写入 '{output_filename}' ---")
    with open(output_filename, 'w') as f:
        for i, atom in enumerate(mobile_data):
            x,y,z = final_coords[i]
            f.write(f"ATOM  {i+1:5d} {atom['atom_name']:<4s} {atom['res_name']:<3s} {atom['chain']}{atom['res_seq']:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

    print(f"成功！您现在可以一起查看 '{args.target_map}' 和 '{output_filename}'。")
    print(f"\n程序结束于: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n总执行时间: {datetime.datetime.now() - start_time}")

if __name__ == '__main__':
    main()
