import os
import time
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import gmsh
import meshio
import pyvista as pv

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


# ============================================================
# 全局设置
# ============================================================
FORCE_CPU = False
device = torch.device("cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))

torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

N_JOBS = 1

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

FIGSIZE_HISTORY = (7.2, 8.8)

COLOR_RED = "#C44E52"
COLOR_BLUE = "#4C72B0"
COLOR_GREEN = "#55A868"
COLOR_PURPLE = "#8172B2"
COLOR_BLACK = "#222222"

TITLE_KW = dict(fontsize=12, pad=6)
LABEL_KW = dict(fontsize=12)

PV_FONT_FAMILY = "times"
PV_TITLE_FONT_SIZE = 32
PV_SCALAR_TITLE_FONT_SIZE = 28
PV_SCALAR_LABEL_FONT_SIZE = 24
PV_TOP_TEXT_FONT_SIZE = 28

PV_SCALAR_BAR_POS_X = 0.80
PV_SCALAR_BAR_POS_Y = 0.14
PV_SCALAR_BAR_WIDTH = 0.045
PV_SCALAR_BAR_HEIGHT = 0.68


# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def savefig_paper(fig, save_path_png):
    fig.savefig(save_path_png, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)


def style_axis_paper(ax, equal=False):
    if equal:
        ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    ax.tick_params(
        axis="both", which="major", direction="in",
        width=0.8, length=3.5, top=False, right=False, bottom=True, left=True
    )
    ax.tick_params(
        axis="both", which="minor", direction="in",
        width=0.6, length=2.0, top=False, right=False, bottom=True, left=True
    )
    ax.minorticks_on()


def relative_l2_error(pred, ref, eps=1e-12):
    pred = pred.reshape(-1)
    ref = ref.reshape(-1)
    return float(np.sqrt(np.sum((pred - ref) ** 2)) / (np.sqrt(np.sum(ref ** 2)) + eps))


def relative_energy_error(pred, ref, eps=1e-12):
    return float(abs(pred - ref) / (abs(ref) + eps))


def max_abs_error(pred, ref):
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.max(np.abs(pred - ref)))


def format_seconds(sec):
    return f"{sec:.6f}"


def save_summary_table(table_dicts, save_csv_path, save_txt_path=None):
    df = pd.DataFrame(table_dicts)
    df.to_csv(save_csv_path, index=False)

    if save_txt_path is not None:
        with open(save_txt_path, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))

    print("\n================= SUMMARY TABLE =================")
    print(df.to_string(index=False))
    print("=================================================\n")
    print(f"Summary table saved to: {save_csv_path}")
    if save_txt_path is not None:
        print(f"Summary table txt saved to: {save_txt_path}")


# ============================================================
# Gmsh：三维 Cook 膜
# ============================================================
def generate_gmsh_cook_membrane_3d(
    msh_path,
    L=48.0,
    H1=44.0,
    H2=16.0,
    B=10.0,
    lc=8.0
):
    gmsh.initialize()
    gmsh.model.add("cook_membrane_3d")

    geo = gmsh.model.geo

    p1 = geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = geo.addPoint(L,   0.0, 0.0, lc)
    p3 = geo.addPoint(L,  H2,  0.0, lc)
    p4 = geo.addPoint(0.0, H1, 0.0, lc)

    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p1)

    loop = geo.addCurveLoop([l1, l2, l3, l4])
    surf = geo.addPlaneSurface([loop])

    geo.extrude([(2, surf)], 0.0, 0.0, B)
    geo.synchronize()

    volumes = gmsh.model.getEntities(dim=3)
    if len(volumes) != 1:
        raise RuntimeError("Expected exactly one volume in Cook membrane model.")
    vol_tag = volumes[0][1]

    surfaces = gmsh.model.getBoundary([(3, vol_tag)], oriented=False, recursive=False)

    left_surfs = []
    right_surfs = []
    free_surfs = []

    tol = 1e-6
    for dim, s in surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, s)
        xc = 0.5 * (xmin + xmax)

        if abs(xc - 0.0) < tol:
            left_surfs.append(s)
        elif abs(xc - L) < tol:
            right_surfs.append(s)
        else:
            free_surfs.append(s)

    gmsh.model.addPhysicalGroup(2, left_surfs, 1)
    gmsh.model.setPhysicalName(2, 1, "left")

    gmsh.model.addPhysicalGroup(2, right_surfs, 2)
    gmsh.model.setPhysicalName(2, 2, "right")

    if len(free_surfs) > 0:
        gmsh.model.addPhysicalGroup(2, free_surfs, 3)
        gmsh.model.setPhysicalName(2, 3, "free")

    gmsh.model.addPhysicalGroup(3, [vol_tag], 10)
    gmsh.model.setPhysicalName(3, 10, "domain")

    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    gmsh.finalize()


# ============================================================
# 网格读取
# ============================================================
def load_gmsh_mesh_3d(msh_path):
    mesh = meshio.read(msh_path)

    points = mesh.points[:, :3]
    nodes_np = points.astype(np.float64)
    nodes_torch = torch.tensor(points, dtype=torch.float32)

    if "gmsh:physical" not in mesh.cell_data:
        raise ValueError("No 'gmsh:physical' found in mesh.cell_data")

    physical_data = mesh.cell_data["gmsh:physical"]

    tetra_blocks = []
    tri_blocks = []
    tri_tags_blocks = []

    for i, cell_block in enumerate(mesh.cells):
        ctype = cell_block.type
        data = cell_block.data
        tags = np.array(physical_data[i])

        if ctype == "tetra":
            tetra_blocks.append(data)
        elif ctype == "triangle":
            tri_blocks.append(data)
            tri_tags_blocks.append(tags)

    if len(tetra_blocks) == 0:
        raise ValueError("No tetrahedral cells found in mesh.")
    if len(tri_blocks) == 0:
        raise ValueError("No boundary triangle faces found in mesh.")

    tet_cells = np.vstack(tetra_blocks).astype(np.int64)
    tri_faces = np.vstack(tri_blocks).astype(np.int64)
    tri_tags = np.concatenate(tri_tags_blocks).astype(np.int64)

    elements_torch = torch.tensor(tet_cells, dtype=torch.long)

    boundary_faces_dict_np = {
        "left": [],
        "right": [],
        "free": [],
    }

    for face, tag in zip(tri_faces, tri_tags):
        if tag == 1:
            boundary_faces_dict_np["left"].append(face.tolist())
        elif tag == 2:
            boundary_faces_dict_np["right"].append(face.tolist())
        elif tag == 3:
            boundary_faces_dict_np["free"].append(face.tolist())

    for k in boundary_faces_dict_np:
        if len(boundary_faces_dict_np[k]) == 0:
            boundary_faces_dict_np[k] = np.empty((0, 3), dtype=np.int64)
        else:
            boundary_faces_dict_np[k] = np.array(boundary_faces_dict_np[k], dtype=np.int64)

    boundary_faces_dict_torch = {
        k: torch.tensor(v, dtype=torch.long) if v.shape[0] > 0 else torch.empty((0, 3), dtype=torch.long)
        for k, v in boundary_faces_dict_np.items()
    }

    return nodes_np, nodes_torch, tet_cells, elements_torch, boundary_faces_dict_np, boundary_faces_dict_torch


def boundary_nodes_from_faces_np(boundary_faces_dict):
    boundaries = {}
    for name, faces in boundary_faces_dict.items():
        if faces.size == 0:
            boundaries[name] = np.empty(0, dtype=np.int64)
        else:
            boundaries[name] = np.unique(faces.reshape(-1))
    return boundaries


def boundary_nodes_from_faces_torch(boundary_faces_dict):
    boundaries = {}
    for name, faces in boundary_faces_dict.items():
        if faces.numel() == 0:
            boundaries[name] = torch.empty(0, dtype=torch.long)
        else:
            boundaries[name] = torch.unique(faces.flatten())
    return boundaries


# ============================================================
# 图构建（向量化加速版）
# ============================================================
def build_graph_from_tetrahedra_vectorized(elements):
    """
    elements: torch.LongTensor [ne, 4] or np.ndarray [ne, 4]
    return edge_index: torch.LongTensor [2, num_edges_directed]
    """
    if isinstance(elements, torch.Tensor):
        elem_np = elements.detach().cpu().numpy()
    else:
        elem_np = np.asarray(elements, dtype=np.int64)

    e = elem_np
    edges = np.concatenate([
        e[:, [0, 1]],
        e[:, [0, 2]],
        e[:, [0, 3]],
        e[:, [1, 2]],
        e[:, [1, 3]],
        e[:, [2, 3]],
    ], axis=0)

    edges_sorted = np.sort(edges, axis=1)
    edges_unique = np.unique(edges_sorted, axis=0)

    rev_edges = edges_unique[:, [1, 0]]
    edges_bidir = np.concatenate([edges_unique, rev_edges], axis=0)

    edge_index = torch.tensor(edges_bidir.T, dtype=torch.long)
    return edge_index


def build_node_features_3d(nodes, boundaries, Ls):
    dev = nodes.device
    dtype = nodes.dtype

    Ls_t = torch.as_tensor(Ls, dtype=dtype, device=dev)

    x = nodes[:, 0:1]
    y = nodes[:, 1:2]
    z = nodes[:, 2:3]

    x_norm = x / Ls_t
    y_norm = y / Ls_t
    z_norm = z / Ls_t

    def build_flag(name):
        flag = torch.zeros((nodes.shape[0], 1), dtype=dtype, device=dev)
        if boundaries[name].numel() > 0:
            idx = boundaries[name].to(dev)
            flag[idx] = 1.0
        return flag

    left_flag = build_flag("left")
    right_flag = build_flag("right")
    free_flag = build_flag("free")

    return torch.cat([x_norm, y_norm, z_norm, left_flag, right_flag, free_flag], dim=1)


def build_edge_features_3d(nodes, edge_index, Ls):
    dev = nodes.device
    dtype = nodes.dtype

    edge_index = edge_index.to(dev)
    Ls_t = torch.as_tensor(Ls, dtype=dtype, device=dev)

    src = edge_index[0]
    dst = edge_index[1]

    rel = (nodes[dst] - nodes[src]) / Ls_t
    dist = torch.sqrt(torch.sum(rel ** 2, dim=1, keepdim=True) + 1e-12)

    return torch.cat([rel, dist], dim=1)


def build_pyg_data_3d(nodes_torch, elements_torch, boundaries_torch, Ls):
    edge_index = build_graph_from_tetrahedra_vectorized(elements_torch).to(nodes_torch.device)

    node_feat = build_node_features_3d(
        nodes_torch,
        boundaries_torch,
        Ls
    )

    edge_attr = build_edge_features_3d(
        nodes_torch,
        edge_index,
        Ls
    )

    data = Data(
        x=node_feat,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=nodes_torch
    )
    return data


# ============================================================
# PyG MPNN
# ============================================================
class PyGMPNNLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, dropout=0.0):
        super().__init__(aggr='add')

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        dh = self.update_mlp(torch.cat([x, agg], dim=-1))
        dh = self.dropout(dh)
        return self.norm(x + dh)

    def message(self, x_i, x_j, edge_attr):
        m_in = torch.cat([x_j, x_i, edge_attr], dim=-1)
        return self.message_mlp(m_in)


class PINN_MPNN_PyG(nn.Module):
    def __init__(self, node_in_dim, edge_dim, hidden_dim=64, mpnn_layers=8, dropout=0.0, output_scale=1e-4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.layers = nn.ModuleList([
            PyGMPNNLayer(hidden_dim, edge_dim, dropout=dropout)
            for _ in range(mpnn_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )

        final_layer = self.decoder[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=output_scale)
        nn.init.zeros_(final_layer.bias)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h0 = self.encoder(x)
        h = h0
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        h = h + h0
        return self.decoder(h)


# ============================================================
# 硬边界
# ============================================================
def build_displacement_bc_values_3d(num_nodes, prescribed_node_values):
    bc_ux = torch.full((num_nodes, 1), float("nan"), dtype=torch.float32, device=device)
    bc_uy = torch.full((num_nodes, 1), float("nan"), dtype=torch.float32, device=device)
    bc_uz = torch.full((num_nodes, 1), float("nan"), dtype=torch.float32, device=device)

    for nodes_idx, ux_val, uy_val, uz_val in prescribed_node_values:
        if nodes_idx.numel() == 0:
            continue
        if ux_val is not None:
            bc_ux[nodes_idx, 0] = float(ux_val)
        if uy_val is not None:
            bc_uy[nodes_idx, 0] = float(uy_val)
        if uz_val is not None:
            bc_uz[nodes_idx, 0] = float(uz_val)

    return bc_ux, bc_uy, bc_uz


def build_hard_bc_masks_and_values_3d(num_nodes, bc_ux, bc_uy, bc_uz):
    mask_ux = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
    mask_uy = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
    mask_uz = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

    val_ux = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
    val_uy = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
    val_uz = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

    ux_fixed = ~torch.isnan(bc_ux)
    uy_fixed = ~torch.isnan(bc_uy)
    uz_fixed = ~torch.isnan(bc_uz)

    mask_ux[ux_fixed] = 0.0
    mask_uy[uy_fixed] = 0.0
    mask_uz[uz_fixed] = 0.0

    val_ux[ux_fixed] = bc_ux[ux_fixed]
    val_uy[uy_fixed] = bc_uy[uy_fixed]
    val_uz[uz_fixed] = bc_uz[uz_fixed]

    return mask_ux, mask_uy, mask_uz, val_ux, val_uy, val_uz


def apply_hard_bc_3d(raw_u_tilde, mask_ux, mask_uy, mask_uz, val_ux, val_uy, val_uz):
    ux = raw_u_tilde[:, 0:1] * mask_ux + val_ux
    uy = raw_u_tilde[:, 1:2] * mask_uy + val_uy
    uz = raw_u_tilde[:, 2:3] * mask_uz + val_uz
    return torch.cat([ux, uy, uz], dim=1)


# ============================================================
# FEM：解析切线
# ============================================================
def neo_hookean_P_3d(F, mu, lam):
    J = np.linalg.det(F)
    J = max(J, 1e-12)
    FinvT = np.linalg.inv(F).T
    logJ = np.log(J)
    P = mu * (F - FinvT) + lam * logJ * FinvT
    return P, J


def cauchy_from_P_F_3d(P, F):
    J = np.linalg.det(F)
    J = max(J, 1e-12)
    sigma = (1.0 / J) * P @ F.T
    return sigma, J


def neo_hookean_dPdF_3d(F, mu, lam):
    J = np.linalg.det(F)
    if J <= 0.0:
        raise RuntimeError(f"Invalid deformation gradient with J = {J:.6e}")

    H = np.linalg.inv(F).T
    logJ = np.log(J)

    A = np.zeros((3, 3, 3, 3), dtype=np.float64)
    I = np.eye(3, dtype=np.float64)

    for i in range(3):
        for Jidx in range(3):
            for k in range(3):
                for L in range(3):
                    A[i, Jidx, k, L] = (
                        mu * I[i, k] * I[Jidx, L]
                        + (mu - lam * logJ) * H[i, L] * H[k, Jidx]
                        + lam * H[k, L] * H[i, Jidx]
                    )
    return A, J


def _precompute_element_reference_worker_3d(e, elements, nodes):
    conn = elements[e]
    X = nodes[conn]

    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]

    Dm = np.column_stack([x2 - x1, x3 - x1, x4 - x1])
    detDm = np.linalg.det(Dm)
    V0 = abs(detDm) / 6.0
    invDm = np.linalg.inv(Dm)

    gradN2 = invDm[:, 0]
    gradN3 = invDm[:, 1]
    gradN4 = invDm[:, 2]
    gradN1 = -gradN2 - gradN3 - gradN4
    gradN = np.stack([gradN1, gradN2, gradN3, gradN4], axis=0)

    dofs = []
    for nid in conn.tolist():
        dofs.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])
    dofs = np.array(dofs, dtype=np.int64)

    return conn, V0, gradN, dofs


def precompute_reference_data_parallel_3d(nodes, elements, n_jobs=N_JOBS):
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_precompute_element_reference_worker_3d)(e, elements, nodes)
        for e in range(elements.shape[0])
    )

    conn_all = np.stack([r[0] for r in results], axis=0)
    V0_all = np.array([r[1] for r in results], dtype=np.float64)
    gradN_all = np.stack([r[2] for r in results], axis=0)
    dofs_all = np.stack([r[3] for r in results], axis=0)

    return conn_all, V0_all, gradN_all, dofs_all


def precompute_sparse_pattern_3d(dofs_all):
    ne = dofs_all.shape[0]
    ndofe = dofs_all.shape[1]

    rows_all = np.zeros((ne, ndofe * ndofe), dtype=np.int64)
    cols_all = np.zeros((ne, ndofe * ndofe), dtype=np.int64)

    for e in range(ne):
        dofs = dofs_all[e]
        rr = np.repeat(dofs, ndofe)
        cc = np.tile(dofs, ndofe)
        rows_all[e] = rr
        cols_all[e] = cc

    return rows_all, cols_all


def element_tangent_stiffness_analytic_fast_3d(ue, gradN, V0, mu, lam):
    u_nodes = ue.reshape(4, 3)
    grad_u = u_nodes.T @ gradN
    F = np.eye(3) + grad_u

    P, J = neo_hookean_P_3d(F, mu, lam)
    A, _ = neo_hookean_dPdF_3d(F, mu, lam)

    fe_nodes = V0 * (P @ gradN.T).T
    fe = fe_nodes.reshape(12)

    Kab = V0 * np.einsum('iJkL,bL,aJ->abik', A, gradN, gradN)
    Ke = Kab.transpose(0, 2, 1, 3).reshape(12, 12)

    Ke = 0.5 * (Ke + Ke.T)
    return Ke, fe, F, J


def _assemble_elements_chunk_analytic_3d(
    elements_idx, dofs_all, gradN_all, V0_all, rows_all, cols_all,
    u_global_flat, mu, lam
):
    vals = []
    fint = []
    fint_idx = []

    sigma_chunk = []
    vm_chunk = []
    J_chunk = []

    rows_chunk = []
    cols_chunk = []

    for e in elements_idx:
        dofs = dofs_all[e]
        ue = u_global_flat[dofs]
        gradN = gradN_all[e]
        V0 = V0_all[e]

        Ke, fe, F, J = element_tangent_stiffness_analytic_fast_3d(ue, gradN, V0, mu, lam)

        rows_chunk.append(rows_all[e])
        cols_chunk.append(cols_all[e])
        vals.append(Ke.ravel())

        fint_idx.append(dofs)
        fint.append(fe)

        P, _ = neo_hookean_P_3d(F, mu, lam)
        sigma, J_sigma = cauchy_from_P_F_3d(P, F)

        sxx = sigma[0, 0]
        syy = sigma[1, 1]
        szz = sigma[2, 2]
        sxy = sigma[0, 1]
        syz = sigma[1, 2]
        sxz = sigma[0, 2]

        vm = np.sqrt(
            0.5 * (
                (sxx - syy)**2 +
                (syy - szz)**2 +
                (szz - sxx)**2 +
                6.0 * (sxy**2 + syz**2 + sxz**2)
            )
        )

        sigma_chunk.append([sxx, syy, szz, sxy, syz, sxz])
        vm_chunk.append(vm)
        J_chunk.append(J_sigma)

    rows_chunk = np.concatenate(rows_chunk) if len(rows_chunk) > 0 else np.empty(0, dtype=np.int64)
    cols_chunk = np.concatenate(cols_chunk) if len(cols_chunk) > 0 else np.empty(0, dtype=np.int64)
    vals = np.concatenate(vals) if len(vals) > 0 else np.empty(0, dtype=np.float64)
    fint_idx = np.concatenate(fint_idx) if len(fint_idx) > 0 else np.empty(0, dtype=np.int64)
    fint = np.concatenate(fint) if len(fint) > 0 else np.empty(0, dtype=np.float64)

    sigma_chunk = np.array(sigma_chunk, dtype=np.float64)
    vm_chunk = np.array(vm_chunk, dtype=np.float64)
    J_chunk = np.array(J_chunk, dtype=np.float64)

    return rows_chunk, cols_chunk, vals, fint_idx, fint, sigma_chunk, vm_chunk, J_chunk


def assemble_global_system_parallel_analytic_3d(
    nodes, elements, dofs_all, gradN_all, V0_all, rows_all, cols_all,
    u, mu, lam, n_jobs=N_JOBS
):
    ndof = nodes.shape[0] * 3
    ne = elements.shape[0]
    u_flat = u.reshape(-1)

    chunk_size = int(np.ceil(ne / n_jobs))
    element_chunks = [
        np.arange(i, min(i + chunk_size, ne), dtype=np.int64)
        for i in range(0, ne, chunk_size)
    ]

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_assemble_elements_chunk_analytic_3d)(
            chunk, dofs_all, gradN_all, V0_all, rows_all, cols_all,
            u_flat, mu, lam
        )
        for chunk in element_chunks
    )

    rows = np.concatenate([r[0] for r in results]) if len(results) > 0 else np.empty(0, dtype=np.int64)
    cols = np.concatenate([r[1] for r in results]) if len(results) > 0 else np.empty(0, dtype=np.int64)
    vals = np.concatenate([r[2] for r in results]) if len(results) > 0 else np.empty(0, dtype=np.float64)

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()

    f_int = np.zeros(ndof, dtype=np.float64)
    for r in results:
        np.add.at(f_int, r[3], r[4])

    sigma_all = np.vstack([r[5] for r in results])
    vm_all = np.concatenate([r[6] for r in results])
    J_all = np.concatenate([r[7] for r in results])

    return K, f_int, sigma_all, vm_all, J_all


# ============================================================
# 外载
# ============================================================
def precompute_surface_traction_faces(right_faces, nodes):
    face_data = []
    for face in right_faces:
        X = nodes[face]
        a = X[1] - X[0]
        b = X[2] - X[0]
        area = 0.5 * np.linalg.norm(np.cross(a, b))

        dofs = []
        for nid in face.tolist():
            dofs.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

        face_data.append((face, area, np.array(dofs, dtype=np.int64)))
    return face_data


def assemble_external_force_right_face(nodes, right_face_data, qy):
    ndof = nodes.shape[0] * 3
    f_ext = np.zeros(ndof, dtype=np.float64)
    t = np.array([0.0, qy, 0.0], dtype=np.float64)

    for face, area, dofs in right_face_data:
        fe = np.tile(t, 3) * (area / 3.0)
        np.add.at(f_ext, dofs, fe)
    return f_ext


# ============================================================
# numpy 边界
# ============================================================
def build_dirichlet_bcs_numpy_3d(nodes, boundaries):
    num_nodes = nodes.shape[0]
    bc_ux = np.full(num_nodes, np.nan, dtype=np.float64)
    bc_uy = np.full(num_nodes, np.nan, dtype=np.float64)
    bc_uz = np.full(num_nodes, np.nan, dtype=np.float64)

    bc_ux[boundaries["left"]] = 0.0
    bc_uy[boundaries["left"]] = 0.0
    bc_uz[boundaries["left"]] = 0.0
    return bc_ux, bc_uy, bc_uz


def build_fixed_dofs_and_values_from_bc_numpy_3d(bc_ux, bc_uy, bc_uz):
    fixed_dofs = []
    fixed_vals = []

    num_nodes = bc_ux.shape[0]
    for i in range(num_nodes):
        if not np.isnan(bc_ux[i]):
            fixed_dofs.append(3 * i)
            fixed_vals.append(bc_ux[i])
        if not np.isnan(bc_uy[i]):
            fixed_dofs.append(3 * i + 1)
            fixed_vals.append(bc_uy[i])
        if not np.isnan(bc_uz[i]):
            fixed_dofs.append(3 * i + 2)
            fixed_vals.append(bc_uz[i])

    return np.array(fixed_dofs, dtype=np.int64), np.array(fixed_vals, dtype=np.float64)


def build_free_dofs_numpy(num_nodes, fixed_dofs, ndpn=3):
    ndof = num_nodes * ndpn
    all_dofs = np.arange(ndof, dtype=np.int64)
    mask = np.ones(ndof, dtype=bool)
    mask[fixed_dofs] = False
    return all_dofs[mask]


# ============================================================
# FEM 牛顿求解
# ============================================================
def solve_hyperelastic_newton_parallel_analytic_3d(
    nodes,
    elements,
    dofs_all,
    gradN_all,
    V0_all,
    rows_all,
    cols_all,
    mu,
    lam,
    fixed_dofs,
    fixed_vals,
    f_ext_full,
    n_steps=15,
    newton_max_iter=20,
    newton_tol_res=1e-6,
    newton_tol_du=1e-8,
    adaptive=True,
    min_load_increment=None,
    max_load_increment=None,
    cutback_factor=0.5,
    growth_factor=1.5,
    easy_iter_threshold=5,
    hard_iter_threshold=12,
    max_cutbacks=12,
    n_jobs=N_JOBS,
    verbose=True
):
    """
    非线性超弹性 FEM 的增量 Newton-Raphson 求解器。

    自适应载荷步进规则
    ------------------
    1. 初始载荷增量为 1 / n_steps。
    2. 当前载荷步不收敛、出现非有限数值或单元翻转时：
       回退到上一个已收敛状态，并将载荷增量乘以 cutback_factor。
    3. 若某一步在 easy_iter_threshold 次以内收敛：
       下一步载荷增量乘以 growth_factor。
    4. 若某一步需要不少于 hard_iter_threshold 次迭代：
       下一步载荷增量乘以 cutback_factor。
    5. 载荷增量始终限制在 [min_load_increment, max_load_increment] 内。

    参数 n_steps 为兼容原调用保留；启用 adaptive 时，它只控制初始步长。
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if not (0.0 < cutback_factor < 1.0):
        raise ValueError("cutback_factor must satisfy 0 < cutback_factor < 1.")
    if growth_factor < 1.0:
        raise ValueError("growth_factor must be >= 1.")
    if easy_iter_threshold < 1:
        raise ValueError("easy_iter_threshold must be >= 1.")
    if hard_iter_threshold < easy_iter_threshold:
        raise ValueError(
            "hard_iter_threshold must be greater than or equal to "
            "easy_iter_threshold."
        )
    if max_cutbacks < 0:
        raise ValueError("max_cutbacks must be >= 0.")

    ndof = nodes.shape[0] * 3
    u = np.zeros(ndof, dtype=np.float64)
    u[fixed_dofs] = fixed_vals

    free_dofs = build_free_dofs_numpy(nodes.shape[0], fixed_dofs, ndpn=3)
    step_logs = []

    initial_increment = 1.0 / float(n_steps)

    if min_load_increment is None:
        min_load_increment = initial_increment / 64.0
    if max_load_increment is None:
        max_load_increment = min(0.25, 4.0 * initial_increment)

    min_load_increment = float(min_load_increment)
    max_load_increment = float(max_load_increment)

    if min_load_increment <= 0.0:
        raise ValueError("min_load_increment must be positive.")
    if max_load_increment < min_load_increment:
        raise ValueError(
            "max_load_increment must be greater than or equal to "
            "min_load_increment."
        )

    load_increment = float(np.clip(
        initial_increment,
        min_load_increment,
        max_load_increment
    ))

    load_factor = 0.0
    accepted_step = 0
    total_attempts = 0
    total_cutbacks = 0
    eps_load = 1e-12

    total_fem_solve_start = time.perf_counter()

    while load_factor < 1.0 - eps_load:
        increment_used = min(load_increment, 1.0 - load_factor)
        target_load_factor = load_factor + increment_used

        # 当前已收敛状态：若本次尝试失败，必须完整回退到这里。
        u_committed = u.copy()
        cutbacks_this_step = 0

        while True:
            total_attempts += 1
            u_trial = u_committed.copy()
            u_trial[fixed_dofs] = fixed_vals

            if verbose:
                print(
                    f"\n================ Adaptive load attempt {total_attempts} "
                    f"| accepted step {accepted_step + 1} "
                    f"| factor: {load_factor:.6f} -> {target_load_factor:.6f} "
                    f"| dLambda = {increment_used:.6e} ================"
                )

            converged = False
            failure_reason = ""
            res_norm = np.inf
            du_norm = np.inf
            J_min = np.nan
            J_max = np.nan
            sigma_all = None
            vm_all = None
            J_all = None
            K = None
            f_int = None

            for it in range(1, newton_max_iter + 1):
                try:
                    K, f_int, sigma_all, vm_all, J_all = (
                        assemble_global_system_parallel_analytic_3d(
                            nodes=nodes,
                            elements=elements,
                            dofs_all=dofs_all,
                            gradN_all=gradN_all,
                            V0_all=V0_all,
                            rows_all=rows_all,
                            cols_all=cols_all,
                            u=u_trial.reshape(-1, 3),
                            mu=mu,
                            lam=lam,
                            n_jobs=n_jobs
                        )
                    )
                except (RuntimeError, np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                    failure_reason = f"assembly/material failure: {exc}"
                    break

                if (
                    not np.all(np.isfinite(f_int))
                    or not np.all(np.isfinite(J_all))
                ):
                    failure_reason = "non-finite internal force or Jacobian"
                    break

                J_min = float(np.min(J_all))
                J_max = float(np.max(J_all))

                if J_min <= 0.0:
                    failure_reason = f"element inversion detected, min(J)={J_min:.6e}"
                    break

                R = f_int - target_load_factor * f_ext_full
                Rf = R[free_dofs]
                res_norm = float(np.linalg.norm(Rf))

                if not np.isfinite(res_norm):
                    failure_reason = "non-finite residual norm"
                    break

                Kff = K[free_dofs][:, free_dofs].tocsc()

                try:
                    du_f = spla.spsolve(Kff, -Rf)
                except Exception as exc:
                    failure_reason = f"linear solve failure: {exc}"
                    break

                if not np.all(np.isfinite(du_f)):
                    failure_reason = "linear solve returned non-finite increment"
                    break

                du_norm = float(np.linalg.norm(du_f))

                if verbose:
                    print(
                        f"[Accepted {accepted_step + 1:02d} | "
                        f"Attempt {total_attempts:02d}] Iter {it:02d} | "
                        f"||R_f||={res_norm:.6e} | "
                        f"||du_f||={du_norm:.6e} | "
                        f"J_min={J_min:.6e} | "
                        f"J_max={J_max:.6e}"
                    )

                # 与原程序保持一致：先施加 Newton 修正，再判断本轮收敛。
                u_trial[free_dofs] += du_f
                u_trial[fixed_dofs] = fixed_vals

                if res_norm < newton_tol_res and du_norm < newton_tol_du:
                    converged = True
                    break

            if converged:
                # 正式接受当前载荷步。
                u = u_trial
                load_factor = target_load_factor
                accepted_step += 1

                step_logs.append({
                    "step": accepted_step,
                    "attempt": total_attempts,
                    "load_factor": float(load_factor),
                    "load_increment": float(increment_used),
                    "newton_iterations": int(it),
                    "cutbacks_before_accept": int(cutbacks_this_step),
                    "res_norm": float(res_norm),
                    "du_norm": float(du_norm),
                    "J_min": float(J_min),
                    "J_max": float(J_max),
                })

                old_increment = increment_used

                if adaptive:
                    if it <= easy_iter_threshold:
                        proposed_increment = old_increment * growth_factor
                        adapt_action = "grow"
                    elif it >= hard_iter_threshold:
                        proposed_increment = old_increment * cutback_factor
                        adapt_action = "shrink-next"
                    else:
                        proposed_increment = old_increment
                        adapt_action = "keep"

                    load_increment = float(np.clip(
                        proposed_increment,
                        min_load_increment,
                        max_load_increment
                    ))
                else:
                    load_increment = initial_increment
                    adapt_action = "fixed"

                # 最后一步后无需再给出一个虚假的剩余增量。
                remaining = max(0.0, 1.0 - load_factor)
                if remaining > eps_load:
                    load_increment = min(load_increment, remaining)

                if verbose:
                    print(
                        f"--> Step accepted: lambda={load_factor:.6f}, "
                        f"Newton iterations={it}, action={adapt_action}, "
                        f"next dLambda={load_increment:.6e}"
                    )
                break

            # 当前尝试失败：回退，并缩小当前载荷增量后重试。
            if not failure_reason:
                failure_reason = (
                    f"Newton did not converge within {newton_max_iter} iterations"
                )

            if not adaptive:
                raise RuntimeError(
                    f"Newton failed at target load factor "
                    f"{target_load_factor:.6f}: {failure_reason}"
                )

            cutbacks_this_step += 1
            total_cutbacks += 1

            proposed_increment = increment_used * cutback_factor
            if (
                cutbacks_this_step > max_cutbacks
                or proposed_increment < min_load_increment - eps_load
            ):
                raise RuntimeError(
                    "Adaptive FEM step failed: the load increment cannot be "
                    "reduced further.\n"
                    f"Last converged load factor = {load_factor:.12f}\n"
                    f"Failed target load factor = {target_load_factor:.12f}\n"
                    f"Current increment = {increment_used:.12e}\n"
                    f"Proposed increment = {proposed_increment:.12e}\n"
                    f"Minimum increment = {min_load_increment:.12e}\n"
                    f"Cutbacks for this step = {cutbacks_this_step}\n"
                    f"Failure reason = {failure_reason}"
                )

            # 回退到上一个收敛位移，并用更小的目标载荷重新求解。
            u = u_committed.copy()
            increment_used = max(proposed_increment, min_load_increment)
            increment_used = min(increment_used, 1.0 - load_factor)
            target_load_factor = load_factor + increment_used
            load_increment = increment_used

            if verbose:
                print(
                    f"--> Step rejected: {failure_reason}\n"
                    f"    Roll back to lambda={load_factor:.6f}; "
                    f"cut back dLambda to {increment_used:.6e} and retry."
                )

    # 在最终平衡状态重新组装，保证输出应力、内力和切线均对应 lambda=1。
    K, f_int, sigma_all, vm_all, J_all = (
        assemble_global_system_parallel_analytic_3d(
            nodes=nodes,
            elements=elements,
            dofs_all=dofs_all,
            gradN_all=gradN_all,
            V0_all=V0_all,
            rows_all=rows_all,
            cols_all=cols_all,
            u=u.reshape(-1, 3),
            mu=mu,
            lam=lam,
            n_jobs=n_jobs
        )
    )

    fem_solve_time = time.perf_counter() - total_fem_solve_start

    return {
        "u": u.reshape(nodes.shape[0], 3),
        "K": K,
        "f_int": f_int,
        "sigma": sigma_all,
        "von_mises": vm_all,
        "J": J_all,
        "step_logs": step_logs,
        "fixed_dofs": fixed_dofs,
        "fixed_vals": fixed_vals,
        "free_dofs": free_dofs,
        "fem_solve_time": fem_solve_time,
        "adaptive": bool(adaptive),
        "accepted_steps": int(accepted_step),
        "total_attempts": int(total_attempts),
        "total_cutbacks": int(total_cutbacks),
        "initial_load_increment": float(initial_increment),
        "min_load_increment": float(min_load_increment),
        "max_load_increment": float(max_load_increment),
    }


def summarize_boundary_reactions_numpy_3d(f_int, boundary_nodes_dict):
    out = {}
    for name, nodes_idx in boundary_nodes_dict.items():
        if nodes_idx.size == 0:
            out[f"{name}_Rx"] = 0.0
            out[f"{name}_Ry"] = 0.0
            out[f"{name}_Rz"] = 0.0
            continue
        dofs_x = 3 * nodes_idx
        dofs_y = 3 * nodes_idx + 1
        dofs_z = 3 * nodes_idx + 2
        out[f"{name}_Rx"] = np.sum(f_int[dofs_x])
        out[f"{name}_Ry"] = np.sum(f_int[dofs_y])
        out[f"{name}_Rz"] = np.sum(f_int[dofs_z])
    return out


# ============================================================
# Torch：手写 3x3 det / inv
# ============================================================
def det3x3_torch(F):
    a = F[..., 0, 0]
    b = F[..., 0, 1]
    c = F[..., 0, 2]
    d = F[..., 1, 0]
    e = F[..., 1, 1]
    f = F[..., 1, 2]
    g = F[..., 2, 0]
    h = F[..., 2, 1]
    i = F[..., 2, 2]

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def inv3x3_torch(F, eps=1e-8):
    a = F[..., 0, 0]
    b = F[..., 0, 1]
    c = F[..., 0, 2]
    d = F[..., 1, 0]
    e = F[..., 1, 1]
    f = F[..., 1, 2]
    g = F[..., 2, 0]
    h = F[..., 2, 1]
    i = F[..., 2, 2]

    A11 =  (e * i - f * h)
    A12 = -(d * i - f * g)
    A13 =  (d * h - e * g)

    A21 = -(b * i - c * h)
    A22 =  (a * i - c * g)
    A23 = -(a * h - b * g)

    A31 =  (b * f - c * e)
    A32 = -(a * f - c * d)
    A33 =  (a * e - b * d)

    detF = a * A11 + b * A12 + c * A13
    detF = torch.clamp(detF, min=eps)

    row1 = torch.stack([A11, A21, A31], dim=-1)
    row2 = torch.stack([A12, A22, A32], dim=-1)
    row3 = torch.stack([A13, A23, A33], dim=-1)
    invF = torch.stack([row1, row2, row3], dim=-2) / detF[..., None, None]

    return invF


# ============================================================
# Torch：参考几何预处理（CPU/NumPy）
# ============================================================
def precompute_reference_tetra_data_torch(nodes, elements, Ls):
    nodes_np = nodes.detach().cpu().numpy()
    elements_np = elements.detach().cpu().numpy()

    elem_vol = []
    elem_vol_tilde = []
    elem_gradN = []
    elem_dofs = []

    for e in range(elements_np.shape[0]):
        conn = elements_np[e]
        X = nodes_np[conn]

        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        x4 = X[3]

        Dm = np.column_stack([x2 - x1, x3 - x1, x4 - x1])
        detDm = np.linalg.det(Dm)
        V = abs(detDm) / 6.0
        invDm = np.linalg.inv(Dm)

        gradN2 = invDm[:, 0]
        gradN3 = invDm[:, 1]
        gradN4 = invDm[:, 2]
        gradN1 = -gradN2 - gradN3 - gradN4
        gradN = np.stack([gradN1, gradN2, gradN3, gradN4], axis=0)

        dofs = []
        for nid in conn.tolist():
            dofs.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

        elem_vol.append(V)
        elem_vol_tilde.append(V / (Ls ** 3))
        elem_gradN.append(gradN)
        elem_dofs.append(dofs)

    return (
        torch.tensor(np.array(elem_vol), dtype=torch.float32, device=device),
        torch.tensor(np.array(elem_vol_tilde), dtype=torch.float32, device=device),
        torch.tensor(np.array(elem_gradN), dtype=torch.float32, device=device),
        torch.tensor(np.array(elem_dofs), dtype=torch.long, device=device),
    )


# ============================================================
# Torch：纯能量法（向量化）
# ============================================================
def build_all_F_from_u_tilde_3d(u_pred_tilde, elements, elem_gradN, scales):
    us = scales["us"]
    ue = u_pred_tilde[elements]
    grad_u_phys = us * torch.bmm(ue.transpose(1, 2), elem_gradN)
    I = torch.eye(3, dtype=u_pred_tilde.dtype, device=u_pred_tilde.device).unsqueeze(0)
    F = I + grad_u_phys
    return F


def neo_hookean_energy_density_nondim_3d(F, mu_tilde, lam_tilde):
    J = det3x3_torch(F)
    J = torch.clamp(J, min=1e-8)

    C = F.transpose(-1, -2) @ F
    I1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)

    logJ = torch.log(J)
    W_tilde = (
        0.5 * mu_tilde * (I1 - 3.0 - 2.0 * logJ)
        + 0.5 * lam_tilde * (logJ ** 2)
    )
    return W_tilde, J


def compute_internal_energy_nondim_hyperelastic_3d(
    u_pred_tilde, elements, elem_vol_tilde, elem_gradN, mu_tilde, lam_tilde, scales
):
    F = build_all_F_from_u_tilde_3d(u_pred_tilde, elements, elem_gradN, scales)
    W_tilde, _ = neo_hookean_energy_density_nondim_3d(F, mu_tilde, lam_tilde)
    U_int_tilde = torch.sum(W_tilde * elem_vol_tilde)
    return U_int_tilde


def compute_external_work_nondim_3d(u_pred_tilde, right_faces, nodes, qy, scales):
    us = scales["us"]
    energy_s = scales["energy_s"]

    if right_faces.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    t_phys = torch.tensor([0.0, qy, 0.0], dtype=torch.float32, device=device)

    X = nodes[right_faces]
    u_face = u_pred_tilde[right_faces] * us

    a = X[:, 1, :] - X[:, 0, :]
    b = X[:, 2, :] - X[:, 0, :]
    area = 0.5 * torch.norm(torch.cross(a, b, dim=1), dim=1)

    u_avg = u_face.mean(dim=1)
    work_faces = torch.sum(u_avg * t_phys.unsqueeze(0), dim=1) * area

    W_ext = torch.sum(work_faces)
    return W_ext / energy_s


def compute_potential_energy_nondim_3d(
    u_pred_tilde,
    elements,
    elem_vol_tilde,
    elem_gradN,
    right_faces_torch,
    nodes_torch,
    qy,
    mu_tilde,
    lam_tilde,
    scales
):
    U_int_tilde = compute_internal_energy_nondim_hyperelastic_3d(
        u_pred_tilde=u_pred_tilde,
        elements=elements,
        elem_vol_tilde=elem_vol_tilde,
        elem_gradN=elem_gradN,
        mu_tilde=mu_tilde,
        lam_tilde=lam_tilde,
        scales=scales
    )

    W_ext_tilde = compute_external_work_nondim_3d(
        u_pred_tilde=u_pred_tilde,
        right_faces=right_faces_torch,
        nodes=nodes_torch,
        qy=qy,
        scales=scales
    )

    Pi_tilde = U_int_tilde - W_ext_tilde
    return Pi_tilde, U_int_tilde, W_ext_tilde


def compute_element_stress_neo_hookean_from_tilde_3d(
    u_pred_tilde, elements, elem_gradN, mu_tilde, lam_tilde, scales
):
    sigma_s = scales["sigma_s"]

    F = build_all_F_from_u_tilde_3d(u_pred_tilde, elements, elem_gradN, scales)

    J = det3x3_torch(F)
    J = torch.clamp(J, min=1e-8)

    FinvT = inv3x3_torch(F).transpose(-1, -2)
    logJ = torch.log(J)

    P_tilde = mu_tilde * (F - FinvT) + lam_tilde * logJ[:, None, None] * FinvT
    sigma_tilde = (1.0 / J)[:, None, None] * torch.bmm(P_tilde, F.transpose(1, 2))
    sigma = sigma_s * sigma_tilde

    sxx = sigma[:, 0, 0]
    syy = sigma[:, 1, 1]
    szz = sigma[:, 2, 2]
    sxy = sigma[:, 0, 1]
    syz = sigma[:, 1, 2]
    sxz = sigma[:, 0, 2]

    vm = torch.sqrt(
        0.5 * (
            (sxx - syy)**2 +
            (syy - szz)**2 +
            (szz - sxx)**2 +
            6.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )

    sigma_out = torch.stack([sxx, syy, szz, sxy, syz, sxz], dim=1)
    return sigma_out, vm, J


def train_model_pure_energy_pyg(
    model,
    optimizer,
    scheduler,
    data,
    mask_ux, mask_uy, mask_uz,
    val_ux, val_uy, val_uz,
    elements_torch,
    elem_vol_tilde,
    elem_gradN_torch,
    boundary_faces_torch,
    nodes_torch,
    qy,
    mu_tilde,
    lam_tilde,
    scales,
    u_fem,
    Pi_fem_tilde=None,
    epochs=400
):
    history = {
        "loss": [],
        "U_int_tilde": [],
        "W_ext_tilde": [],
        "potential_tilde": [],
        "ux_rel_l2": [],
        "uy_rel_l2": [],
        "uz_rel_l2": [],
        "umag_rel_l2": [],
        "rel_Pi": [],
        "lr": [],
    }

    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        raw_u_tilde = model(data)
        u_pred_tilde = apply_hard_bc_3d(raw_u_tilde, mask_ux, mask_uy, mask_uz, val_ux, val_uy, val_uz)

        loss, U_int_tilde, W_ext_tilde = compute_potential_energy_nondim_3d(
            u_pred_tilde=u_pred_tilde,
            elements=elements_torch,
            elem_vol_tilde=elem_vol_tilde,
            elem_gradN=elem_gradN_torch,
            right_faces_torch=boundary_faces_torch["right"],
            nodes_torch=nodes_torch,
            qy=qy,
            mu_tilde=mu_tilde,
            lam_tilde=lam_tilde,
            scales=scales
        )

        if torch.isnan(loss):
            print(f"NaN encountered at epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(float(loss.item()))

        with torch.no_grad():
            u_pred = (u_pred_tilde * scales["us"]).detach().cpu().numpy()

            ux_rel = relative_l2_error(u_pred[:, 0], u_fem[:, 0])
            uy_rel = relative_l2_error(u_pred[:, 1], u_fem[:, 1])
            uz_rel = relative_l2_error(u_pred[:, 2], u_fem[:, 2])
            umag_rel = relative_l2_error(
                np.linalg.norm(u_pred, axis=1),
                np.linalg.norm(u_fem, axis=1)
            )

            if Pi_fem_tilde is not None:
                rel_Pi = relative_energy_error(float(loss.item()), Pi_fem_tilde)
            else:
                rel_Pi = np.nan

        current_lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(float(loss.item()))
        history["U_int_tilde"].append(float(U_int_tilde.item()))
        history["W_ext_tilde"].append(float(W_ext_tilde.item()))
        history["potential_tilde"].append(float(loss.item()))
        history["ux_rel_l2"].append(ux_rel)
        history["uy_rel_l2"].append(uy_rel)
        history["uz_rel_l2"].append(uz_rel)
        history["umag_rel_l2"].append(umag_rel)
        history["rel_Pi"].append(rel_Pi)
        history["lr"].append(current_lr)

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d}/{epochs} | "
                f"lr={current_lr:.3e} | "
                f"Pi={float(loss.item()):.6e} | "
                f"U_int={float(U_int_tilde.item()):.6e} | "
                f"W_ext={float(W_ext_tilde.item()):.6e} | "
                f"RelPi={rel_Pi:.6e} | "
                f"ux_rel={ux_rel:.6e} | "
                f"uy_rel={uy_rel:.6e} | "
                f"uz_rel={uz_rel:.6e} | "
                f"umag_rel={umag_rel:.6e}"
            )

    training_time = time.perf_counter() - train_start
    return history, training_time


# ============================================================
# 误差
# ============================================================
def evaluate_errors_3d(u_pred, u_fem, sigma_pred, sigma_fem, vm_pred, vm_fem):
    ux_pred, uy_pred, uz_pred = u_pred[:, 0], u_pred[:, 1], u_pred[:, 2]
    ux_fem, uy_fem, uz_fem = u_fem[:, 0], u_fem[:, 1], u_fem[:, 2]

    umag_pred = np.sqrt(ux_pred**2 + uy_pred**2 + uz_pred**2)
    umag_fem = np.sqrt(ux_fem**2 + uy_fem**2 + uz_fem**2)

    out = {
        "ux_rel_l2": relative_l2_error(ux_pred, ux_fem),
        "uy_rel_l2": relative_l2_error(uy_pred, uy_fem),
        "uz_rel_l2": relative_l2_error(uz_pred, uz_fem),
        "umag_rel_l2": relative_l2_error(umag_pred, umag_fem),
        "vm_rel_l2": relative_l2_error(vm_pred, vm_fem),
        "umag_max_abs_err": max_abs_error(umag_pred, umag_fem),
        "vm_max_abs_err": max_abs_error(vm_pred, vm_fem),
    }

    names = ["sxx", "syy", "szz", "sxy", "syz", "sxz"]
    for i, name in enumerate(names):
        out[f"{name}_rel_l2"] = relative_l2_error(sigma_pred[:, i], sigma_fem[:, i])

    return out


# ============================================================
# 单元场 -> 节点场
# ============================================================
def element_to_nodal_field_3d(elements, elem_field, num_nodes):
    elem_field = np.asarray(elem_field)

    if elem_field.ndim == 1:
        nodal_sum = np.zeros(num_nodes, dtype=np.float64)
        nodal_count = np.zeros(num_nodes, dtype=np.float64)

        for e in range(elements.shape[0]):
            conn = elements[e]
            nodal_sum[conn] += elem_field[e]
            nodal_count[conn] += 1.0

        return nodal_sum / (nodal_count + 1e-12)

    elif elem_field.ndim == 2:
        ncomp = elem_field.shape[1]
        nodal_sum = np.zeros((num_nodes, ncomp), dtype=np.float64)
        nodal_count = np.zeros(num_nodes, dtype=np.float64)

        for e in range(elements.shape[0]):
            conn = elements[e]
            nodal_sum[conn] += elem_field[e]
            nodal_count[conn] += 1.0

        return nodal_sum / (nodal_count[:, None] + 1e-12)

    else:
        raise ValueError("elem_field must be 1D or 2D.")


# ============================================================
# VTU 导出
# ============================================================
def export_to_vtu_3d_smoothed(
    save_path,
    nodes,
    elements,
    u_pred,
    u_fem,
    sigma_pred,
    sigma_fem,
    vm_pred,
    vm_fem,
    J_pred,
    J_fem
):
    points = nodes.copy()
    cells = [("tetra", elements)]

    num_nodes = nodes.shape[0]

    u_error = u_pred - u_fem
    umag_pred = np.linalg.norm(u_pred, axis=1)
    umag_fem = np.linalg.norm(u_fem, axis=1)
    umag_error = np.abs(umag_pred - umag_fem)

    sigma_pred_nodal = element_to_nodal_field_3d(elements, sigma_pred, num_nodes)
    sigma_fem_nodal = element_to_nodal_field_3d(elements, sigma_fem, num_nodes)
    sigma_error_nodal = np.abs(sigma_pred_nodal - sigma_fem_nodal)

    vm_pred_nodal = element_to_nodal_field_3d(elements, vm_pred, num_nodes)
    vm_fem_nodal = element_to_nodal_field_3d(elements, vm_fem, num_nodes)
    vm_error_nodal = np.abs(vm_pred_nodal - vm_fem_nodal)

    J_pred_nodal = element_to_nodal_field_3d(elements, J_pred, num_nodes)
    J_fem_nodal = element_to_nodal_field_3d(elements, J_fem, num_nodes)
    J_error_nodal = np.abs(J_pred_nodal - J_fem_nodal)

    point_data = {
        "u_pred": u_pred,
        "u_fem": u_fem,
        "u_error": u_error,

        "ux_pred": u_pred[:, 0],
        "uy_pred": u_pred[:, 1],
        "uz_pred": u_pred[:, 2],

        "ux_fem": u_fem[:, 0],
        "uy_fem": u_fem[:, 1],
        "uz_fem": u_fem[:, 2],

        "ux_error": np.abs(u_pred[:, 0] - u_fem[:, 0]),
        "uy_error": np.abs(u_pred[:, 1] - u_fem[:, 1]),
        "uz_error": np.abs(u_pred[:, 2] - u_fem[:, 2]),

        "umag_pred": umag_pred,
        "umag_fem": umag_fem,
        "umag_error": umag_error,

        "sxx_pred": sigma_pred_nodal[:, 0],
        "syy_pred": sigma_pred_nodal[:, 1],
        "szz_pred": sigma_pred_nodal[:, 2],
        "sxy_pred": sigma_pred_nodal[:, 3],
        "syz_pred": sigma_pred_nodal[:, 4],
        "sxz_pred": sigma_pred_nodal[:, 5],

        "sxx_fem": sigma_fem_nodal[:, 0],
        "syy_fem": sigma_fem_nodal[:, 1],
        "szz_fem": sigma_fem_nodal[:, 2],
        "sxy_fem": sigma_fem_nodal[:, 3],
        "syz_fem": sigma_fem_nodal[:, 4],
        "sxz_fem": sigma_fem_nodal[:, 5],

        "sxx_error": sigma_error_nodal[:, 0],
        "syy_error": sigma_error_nodal[:, 1],
        "szz_error": sigma_error_nodal[:, 2],
        "sxy_error": sigma_error_nodal[:, 3],
        "syz_error": sigma_error_nodal[:, 4],
        "sxz_error": sigma_error_nodal[:, 5],

        "vm_pred": vm_pred_nodal,
        "vm_fem": vm_fem_nodal,
        "vm_error": vm_error_nodal,

        "J_pred": J_pred_nodal,
        "J_fem": J_fem_nodal,
        "J_error": J_error_nodal,
    }

    cell_data = {
        "sxx_pred_cell": [sigma_pred[:, 0]],
        "syy_pred_cell": [sigma_pred[:, 1]],
        "szz_pred_cell": [sigma_pred[:, 2]],
        "sxy_pred_cell": [sigma_pred[:, 3]],
        "syz_pred_cell": [sigma_pred[:, 4]],
        "sxz_pred_cell": [sigma_pred[:, 5]],
        "vm_pred_cell": [vm_pred],
        "J_pred_cell": [J_pred],

        "sxx_fem_cell": [sigma_fem[:, 0]],
        "syy_fem_cell": [sigma_fem[:, 1]],
        "szz_fem_cell": [sigma_fem[:, 2]],
        "sxy_fem_cell": [sigma_fem[:, 3]],
        "syz_fem_cell": [sigma_fem[:, 4]],
        "sxz_fem_cell": [sigma_fem[:, 5]],
        "vm_fem_cell": [vm_fem],
        "J_fem_cell": [J_fem],
    }

    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data)
    meshio.write(save_path, mesh)
    print(f"Smoothed VTU exported to: {save_path}")


# ============================================================
# PyVista：视角与绘图
# ============================================================
def smooth_warped_grid_for_plot(mesh, vector_name="u_pred", scale=1.0, n_subdiv=1):
    warped = mesh.warp_by_vector(vector_name, factor=scale)
    surf = warped.extract_surface().triangulate()
    if n_subdiv > 0:
        surf = surf.subdivide(n_subdiv, subfilter="linear")
    return surf


def apply_fixed_camera_view(plotter, camera_position):
    plotter.camera_position = camera_position
    plotter.camera.SetParallelProjection(False)


def get_saved_camera_position():
    return [
        (1.0, -1.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]


def save_camera_position_to_txt(camera_position, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("camera_position = [\n")
        f.write(f"    ({camera_position[0][0]}, {camera_position[0][1]}, {camera_position[0][2]}),\n")
        f.write(f"    ({camera_position[1][0]}, {camera_position[1][1]}, {camera_position[1][2]}),\n")
        f.write(f"    ({camera_position[2][0]}, {camera_position[2][1]}, {camera_position[2][2]}),\n")
        f.write("]\n")
    print(f"Camera position saved to: {save_path}")


def pick_camera_interactively(
    vtu_path,
    field="umag_pred",
    vector_name="u_pred",
    scale=1.0,
    smooth_subdiv=1,
    show_edges=False,
    window_size=(1400, 1000)
):
    mesh = pv.read(vtu_path)
    surf = smooth_warped_grid_for_plot(mesh, vector_name=vector_name, scale=scale, n_subdiv=smooth_subdiv)

    p = pv.Plotter(window_size=window_size)
    p.set_background("white")

    p.add_mesh(
        surf,
        scalars=field,
        cmap="jet",
        show_edges=show_edges,
        smooth_shading=True,
        scalar_bar_args=dict(
            title=field,
            vertical=True,
            title_font_size=PV_SCALAR_TITLE_FONT_SIZE,
            label_font_size=PV_SCALAR_LABEL_FONT_SIZE,
            shadow=False,
            n_labels=5,
            italic=False,
            fmt="%.3e",
            font_family=PV_FONT_FAMILY,
            color="black",
            position_x=PV_SCALAR_BAR_POS_X,
            position_y=PV_SCALAR_BAR_POS_Y,
            width=PV_SCALAR_BAR_WIDTH,
            height=PV_SCALAR_BAR_HEIGHT,
        )
    )

    p.add_text(
        "Adjust view manually, then close window",
        position="upper_edge",
        font_size=PV_TOP_TEXT_FONT_SIZE,
        color="black",
        font=PV_FONT_FAMILY
    )

    p.remove_bounds_axes()
    p.show()

    camera_position = p.camera_position
    print("\nSelected camera_position:")
    print(camera_position)
    return camera_position


def save_pyvista_single_paper(
    vtu_path,
    save_png,
    field,
    vector_name,
    scale=1.0,
    title=None,
    clim=None,
    window_size=(2000, 1200),
    font_size_title=PV_TITLE_FONT_SIZE,
    font_size_scalar=PV_SCALAR_TITLE_FONT_SIZE,
    show_edges=False,
    smooth_subdiv=1,
    camera_position=None
):
    mesh = pv.read(vtu_path)
    surf = smooth_warped_grid_for_plot(mesh, vector_name=vector_name, scale=scale, n_subdiv=smooth_subdiv)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")

    p.add_mesh(
        surf,
        scalars=field,
        cmap="jet",
        show_edges=show_edges,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args=dict(
            title="",
            vertical=True,
            title_font_size=font_size_scalar,
            label_font_size=max(12, font_size_scalar - 2),
            shadow=False,
            n_labels=5,
            italic=False,
            fmt="%.3e",
            font_family=PV_FONT_FAMILY,
            color="black",
            position_x=PV_SCALAR_BAR_POS_X,
            position_y=PV_SCALAR_BAR_POS_Y,
            width=PV_SCALAR_BAR_WIDTH,
            height=PV_SCALAR_BAR_HEIGHT,
        )
    )

    if camera_position is not None:
        apply_fixed_camera_view(p, camera_position)

    if title is not None:
        p.add_text(
            title,
            position="upper_edge",
            font_size=font_size_title,
            color="black",
            font=PV_FONT_FAMILY
        )

    p.remove_bounds_axes()
    p.screenshot(save_png)
    p.close()
    print(f"Saved figure: {save_png}")


def save_pyvista_two_panel_paper(
    vtu_path,
    save_png,
    left_field,
    right_field,
    left_vector,
    right_vector,
    left_title,
    right_title,
    scale_left=1.0,
    scale_right=1.0,
    clim_left=None,
    clim_right=None,
    window_size=(2600, 1000),
    font_size_title=PV_TITLE_FONT_SIZE,
    font_size_scalar=PV_SCALAR_TITLE_FONT_SIZE,
    show_edges=False,
    smooth_subdiv=1,
    camera_position=None
):
    mesh = pv.read(vtu_path)
    left_mesh = smooth_warped_grid_for_plot(mesh, vector_name=left_vector, scale=scale_left, n_subdiv=smooth_subdiv)
    right_mesh = smooth_warped_grid_for_plot(mesh, vector_name=right_vector, scale=scale_right, n_subdiv=smooth_subdiv)

    p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=window_size)
    p.set_background("white")

    p.subplot(0, 0)
    p.add_mesh(
        left_mesh,
        scalars=left_field,
        cmap="jet",
        show_edges=show_edges,
        clim=clim_left,
        smooth_shading=True,
        scalar_bar_args=dict(
            title="",
            vertical=True,
            title_font_size=font_size_scalar,
            label_font_size=max(12, font_size_scalar - 2),
            shadow=False,
            n_labels=5,
            italic=False,
            fmt="%.3e",
            font_family=PV_FONT_FAMILY,
            color="black",
            position_x=PV_SCALAR_BAR_POS_X,
            position_y=PV_SCALAR_BAR_POS_Y,
            width=PV_SCALAR_BAR_WIDTH,
            height=PV_SCALAR_BAR_HEIGHT,
        )
    )
    if camera_position is not None:
        apply_fixed_camera_view(p, camera_position)
    p.add_text(left_title, position="upper_edge", font_size=font_size_title, color="black", font=PV_FONT_FAMILY)
    p.remove_bounds_axes()

    p.subplot(0, 1)
    p.add_mesh(
        right_mesh,
        scalars=right_field,
        cmap="jet",
        show_edges=show_edges,
        clim=clim_right,
        smooth_shading=True,
        scalar_bar_args=dict(
            title="",
            vertical=True,
            title_font_size=font_size_scalar,
            label_font_size=max(12, font_size_scalar - 2),
            shadow=False,
            n_labels=5,
            italic=False,
            fmt="%.3e",
            font_family=PV_FONT_FAMILY,
            color="black",
            position_x=PV_SCALAR_BAR_POS_X,
            position_y=PV_SCALAR_BAR_POS_Y,
            width=PV_SCALAR_BAR_WIDTH,
            height=PV_SCALAR_BAR_HEIGHT,
        )
    )
    if camera_position is not None:
        apply_fixed_camera_view(p, camera_position)
    p.add_text(right_title, position="upper_edge", font_size=font_size_title, color="black", font=PV_FONT_FAMILY)
    p.remove_bounds_axes()

    p.screenshot(save_png)
    p.close()
    print(f"Saved paper compare figure: {save_png}")


def save_pyvista_clip_paper(
    vtu_path,
    save_png,
    field,
    vector_name,
    scale=1.0,
    normal="x",
    origin=None,
    title=None,
    window_size=(1600, 1200),
    font_size_title=PV_TITLE_FONT_SIZE,
    font_size_scalar=PV_SCALAR_TITLE_FONT_SIZE,
    smooth_subdiv=1,
    camera_position=None
):
    mesh = pv.read(vtu_path)
    surf = smooth_warped_grid_for_plot(mesh, vector_name=vector_name, scale=scale, n_subdiv=smooth_subdiv)

    if origin is None:
        origin = surf.center

    clipped = surf.clip(normal=normal, origin=origin)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.add_mesh(
        clipped,
        scalars=field,
        cmap="jet",
        smooth_shading=True,
        scalar_bar_args=dict(
            title="",
            vertical=True,
            title_font_size=font_size_scalar,
            label_font_size=max(12, font_size_scalar - 2),
            shadow=False,
            n_labels=5,
            italic=False,
            fmt="%.3e",
            font_family=PV_FONT_FAMILY,
            color="black",
            position_x=PV_SCALAR_BAR_POS_X,
            position_y=PV_SCALAR_BAR_POS_Y,
            width=PV_SCALAR_BAR_WIDTH,
            height=PV_SCALAR_BAR_HEIGHT,
        )
    )

    if camera_position is not None:
        apply_fixed_camera_view(p, camera_position)

    if title is not None:
        p.add_text(title, position="upper_edge", font_size=font_size_title, color="black", font=PV_FONT_FAMILY)

    p.remove_bounds_axes()
    p.screenshot(save_png)
    p.close()
    print(f"Saved paper clip figure: {save_png}")


def save_pyvista_slices_paper(
    vtu_path,
    save_png,
    field,
    vector_name,
    scale=1.0,
    axis="z",
    n=5,
    title=None,
    window_size=(1600, 1200),
    font_size_title=PV_TITLE_FONT_SIZE,
    font_size_scalar=PV_SCALAR_TITLE_FONT_SIZE,
    smooth_subdiv=1,
    camera_position=None
):
    mesh = pv.read(vtu_path)
    surf = smooth_warped_grid_for_plot(mesh, vector_name=vector_name, scale=scale, n_subdiv=smooth_subdiv)
    slices = surf.slice_along_axis(n=n, axis=axis)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.add_mesh(
        slices,
        scalars=field,
        cmap="jet",
        smooth_shading=True,
        scalar_bar_args=dict(
            title="",
            vertical=True,
            title_font_size=font_size_scalar,
            label_font_size=max(12, font_size_scalar - 2),
            shadow=False,
            n_labels=5,
            italic=False,
            fmt="%.3e",
            font_family=PV_FONT_FAMILY,
            color="black",
            position_x=PV_SCALAR_BAR_POS_X,
            position_y=PV_SCALAR_BAR_POS_Y,
            width=PV_SCALAR_BAR_WIDTH,
            height=PV_SCALAR_BAR_HEIGHT,
        )
    )

    if camera_position is not None:
        apply_fixed_camera_view(p, camera_position)

    if title is not None:
        p.add_text(title, position="upper_edge", font_size=font_size_title, color="black", font=PV_FONT_FAMILY)

    p.remove_bounds_axes()
    p.screenshot(save_png)
    p.close()
    print(f"Saved paper slices figure: {save_png}")


def make_displacement_component_figures_paper(vtu_path, fig_dir, scale=1.0, smooth_subdiv=1, camera_position=None):
    ensure_dir(fig_dir)
    mesh = pv.read(vtu_path)

    disp_names = ["ux", "uy", "uz"]
    pretty_map = {"ux": "ux", "uy": "uy", "uz": "uz"}

    for name in disp_names:
        fem_name = f"{name}_fem"
        pred_name = f"{name}_pred"
        err_name = f"{name}_error"

        vmin = min(mesh.point_data[fem_name].min(), mesh.point_data[pred_name].min())
        vmax = max(mesh.point_data[fem_name].max(), mesh.point_data[pred_name].max())

        pretty = pretty_map[name]

        save_pyvista_single_paper(
            vtu_path=vtu_path,
            save_png=os.path.join(fig_dir, f"{name}_fem.png"),
            field=fem_name,
            vector_name="u_fem",
            scale=scale,
            title=f"FEM displacement {pretty}",
            clim=[vmin, vmax],
            smooth_subdiv=smooth_subdiv,
            camera_position=camera_position
        )

        save_pyvista_single_paper(
            vtu_path=vtu_path,
            save_png=os.path.join(fig_dir, f"{name}_pred.png"),
            field=pred_name,
            vector_name="u_pred",
            scale=scale,
            title=f"MPNN displacement {pretty}",
            clim=[vmin, vmax],
            smooth_subdiv=smooth_subdiv,
            camera_position=camera_position
        )

        save_pyvista_single_paper(
            vtu_path=vtu_path,
            save_png=os.path.join(fig_dir, f"{name}_error.png"),
            field=err_name,
            vector_name="u_fem",
            scale=scale,
            title=f"Absolute error in {pretty}",
            smooth_subdiv=smooth_subdiv,
            camera_position=camera_position
        )

        save_pyvista_two_panel_paper(
            vtu_path=vtu_path,
            save_png=os.path.join(fig_dir, f"{name}_compare_fem_vs_pred.png"),
            left_field=fem_name,
            right_field=pred_name,
            left_vector="u_fem",
            right_vector="u_pred",
            left_title=f"FEM displacement {pretty}",
            right_title=f"MPNN displacement {pretty}",
            scale_left=scale,
            scale_right=scale,
            clim_left=[vmin, vmax],
            clim_right=[vmin, vmax],
            smooth_subdiv=smooth_subdiv,
            camera_position=camera_position
        )


def make_stress_component_figures_paper(vtu_path, fig_dir, scale=1.0, smooth_subdiv=1, camera_position=None):
    ensure_dir(fig_dir)
    mesh = pv.read(vtu_path)

    stress_names = ["sxx", "syy", "szz", "sxy", "syz", "sxz"]
    pretty_map = {
        "sxx": "sigma_xx",
        "syy": "sigma_yy",
        "szz": "sigma_zz",
        "sxy": "sigma_xy",
        "syz": "sigma_yz",
        "sxz": "sigma_xz",
    }

    for name in stress_names:
        fem_name = f"{name}_fem"
        pred_name = f"{name}_pred"
        err_name = f"{name}_error"

        vmin = min(mesh.point_data[fem_name].min(), mesh.point_data[pred_name].min())
        vmax = max(mesh.point_data[fem_name].max(), mesh.point_data[pred_name].max())

        pretty = pretty_map[name]

        save_pyvista_single_paper(
            vtu_path=vtu_path,
            save_png=os.path.join(fig_dir, f"{name}_fem.png"),
            field=fem_name,
            vector_name="u_fem",
            scale=scale,
            title=f"FEM stress {pretty}",
            clim=[vmin, vmax],
            smooth_subdiv=smooth_subdiv,
            camera_position=camera_position
        )

        save_pyvista_single_paper(
            vtu_path=vtu_path,
            save_png=os.path.join(fig_dir, f"{name}_pred.png"),
            field=pred_name,
            vector_name="u_pred",
            scale=scale,
            title=f"MPNN stress {pretty}",
            clim=[vmin, vmax],
            smooth_subdiv=smooth_subdiv,
            camera_position=camera_position
        )

        save_pyvista_single_paper(
            vtu_path=vtu_path,
            save_png=os.path.join(fig_dir, f"{name}_error.png"),
            field=err_name,
            vector_name="u_fem",
            scale=scale,
            title=f"Absolute error in {pretty}",
            smooth_subdiv=smooth_subdiv,
            camera_position=camera_position
        )

        save_pyvista_two_panel_paper(
            vtu_path=vtu_path,
            save_png=os.path.join(fig_dir, f"{name}_compare_fem_vs_pred.png"),
            left_field=fem_name,
            right_field=pred_name,
            left_vector="u_fem",
            right_vector="u_pred",
            left_title=f"FEM stress {pretty}",
            right_title=f"MPNN stress {pretty}",
            scale_left=scale,
            scale_right=scale,
            clim_left=[vmin, vmax],
            clim_right=[vmin, vmax],
            smooth_subdiv=smooth_subdiv,
            camera_position=camera_position
        )


def make_all_pyvista_figures_paper(vtu_path, fig_dir, scale=1.0, smooth_subdiv=1, camera_position=None):
    ensure_dir(fig_dir)
    mesh = pv.read(vtu_path)

    umag_min = min(mesh.point_data["umag_fem"].min(), mesh.point_data["umag_pred"].min())
    umag_max = max(mesh.point_data["umag_fem"].max(), mesh.point_data["umag_pred"].max())
    vm_min = min(mesh.point_data["vm_fem"].min(), mesh.point_data["vm_pred"].min())
    vm_max = max(mesh.point_data["vm_fem"].max(), mesh.point_data["vm_pred"].max())

    save_pyvista_single_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "fem_displacement_umag.png"),
        field="umag_fem",
        vector_name="u_fem",
        scale=scale,
        title="FEM displacement magnitude",
        clim=[umag_min, umag_max],
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_single_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "pred_displacement_umag.png"),
        field="umag_pred",
        vector_name="u_pred",
        scale=scale,
        title="MPNN displacement magnitude",
        clim=[umag_min, umag_max],
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_single_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "error_displacement_umag.png"),
        field="umag_error",
        vector_name="u_fem",
        scale=scale,
        title="Absolute error in displacement magnitude",
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_two_panel_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "compare_displacement_fem_vs_pred.png"),
        left_field="umag_fem",
        right_field="umag_pred",
        left_vector="u_fem",
        right_vector="u_pred",
        left_title="FEM displacement magnitude",
        right_title="MPNN displacement magnitude",
        scale_left=scale,
        scale_right=scale,
        clim_left=[umag_min, umag_max],
        clim_right=[umag_min, umag_max],
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_single_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "fem_stress_vm.png"),
        field="vm_fem",
        vector_name="u_fem",
        scale=scale,
        title="FEM von Mises stress",
        clim=[vm_min, vm_max],
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_single_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "pred_stress_vm.png"),
        field="vm_pred",
        vector_name="u_pred",
        scale=scale,
        title="MPNN von Mises stress",
        clim=[vm_min, vm_max],
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_single_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "error_stress_vm.png"),
        field="vm_error",
        vector_name="u_fem",
        scale=scale,
        title="Absolute error in von Mises stress",
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_two_panel_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "compare_stress_fem_vs_pred.png"),
        left_field="vm_fem",
        right_field="vm_pred",
        left_vector="u_fem",
        right_vector="u_pred",
        left_title="FEM von Mises stress",
        right_title="MPNN von Mises stress",
        scale_left=scale,
        scale_right=scale,
        clim_left=[vm_min, vm_max],
        clim_right=[vm_min, vm_max],
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_clip_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "clip_vm_pred.png"),
        field="vm_pred",
        vector_name="u_pred",
        scale=scale,
        normal="x",
        title="Clipped view of predicted von Mises stress",
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    save_pyvista_slices_paper(
        vtu_path=vtu_path,
        save_png=os.path.join(fig_dir, "slices_vm_pred.png"),
        field="vm_pred",
        vector_name="u_pred",
        scale=scale,
        axis="z",
        n=5,
        title="Slices of predicted von Mises stress",
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    make_displacement_component_figures_paper(
        vtu_path=vtu_path,
        fig_dir=os.path.join(fig_dir, "displacement_components"),
        scale=scale,
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )

    make_stress_component_figures_paper(
        vtu_path=vtu_path,
        fig_dir=os.path.join(fig_dir, "stress_components"),
        scale=scale,
        smooth_subdiv=smooth_subdiv,
        camera_position=camera_position
    )


# ============================================================
# 主程序
# ============================================================
def main():
    output_dir = "cook_membrane_3d_pure_energy_mpnn_pyg_fullfig_pickcamera"
    ensure_dir(output_dir)

    fig_dir = os.path.join(output_dir, "pyvista_figures")
    ensure_dir(fig_dir)

    msh_path = os.path.join(output_dir, "cook_membrane_3d.msh")
    vtu_path = os.path.join(output_dir, "results_cook_membrane_3d_smoothed.vtu")
    camera_txt_path = os.path.join(output_dir, "selected_camera_position.txt")
    summary_csv_path = os.path.join(output_dir, "summary_table.csv")
    summary_txt_path = os.path.join(output_dir, "summary_table.txt")

    # --------------------------------------------------------
    # 几何 / 材料 / 载荷
    # --------------------------------------------------------
    L = 48.0
    H1 = 44.0
    H2 = 16.0
    B = 10.0

    lam = 100.0
    mu = 40.0
    qy = -40.0

    # --------------------------------------------------------
    # 参数
    # --------------------------------------------------------
    lc = 3.0

    # FEM 自适应载荷步进参数
    # n_steps 仅用于给出初始载荷增量：initial dLambda = 1 / n_steps
    n_steps = 15
    newton_max_iter = 20
    newton_tol_res = 1e-6
    newton_tol_du = 1e-8

    fem_adaptive = True
    fem_min_load_increment = 1.0e-3
    fem_max_load_increment = 0.20
    fem_cutback_factor = 0.5
    fem_growth_factor = 1.5
    fem_easy_iter_threshold = 5
    fem_hard_iter_threshold = 12
    fem_max_cutbacks = 12

    adamw_epochs = 1000
    adamw_lr = 1e-3
    adamw_weight_decay = 1e-6
    adamw_min_lr = 1e-7

    scheduler_factor = 0.5
    scheduler_patience = 40
    scheduler_threshold = 1e-5
    scheduler_cooldown = 10

    pyvista_scale = 1.0
    pyvista_smooth_subdiv = 1

    pick_camera_first = True

    # --------------------------------------------------------
    # 无量纲尺度
    # --------------------------------------------------------
    Ls = max(L, H1, B)
    us = 1.0
    sigma_s = max(lam, mu)
    fs = sigma_s * (Ls ** 2)
    energy_s = sigma_s * (Ls ** 3)

    scales = {
        "Ls": torch.tensor(Ls, dtype=torch.float32, device=device),
        "us": torch.tensor(us, dtype=torch.float32, device=device),
        "sigma_s": torch.tensor(sigma_s, dtype=torch.float32, device=device),
        "fs": torch.tensor(fs, dtype=torch.float32, device=device),
        "energy_s": torch.tensor(energy_s, dtype=torch.float32, device=device),
    }

    mu_tilde = mu / sigma_s
    lam_tilde = lam / sigma_s

    print(f"Using device = {device}")
    print(f"Using N_JOBS = {N_JOBS}")

    # --------------------------------------------------------
    # 网格
    # --------------------------------------------------------
    generate_gmsh_cook_membrane_3d(
        msh_path=msh_path,
        L=L,
        H1=H1,
        H2=H2,
        B=B,
        lc=lc
    )

    nodes_np, nodes_torch, elements_np, elements_torch, boundary_faces_np, boundary_faces_torch = load_gmsh_mesh_3d(msh_path)
    boundaries_np = boundary_nodes_from_faces_np(boundary_faces_np)
    boundaries_torch = boundary_nodes_from_faces_torch(boundary_faces_torch)

    print("num_nodes =", nodes_np.shape[0])
    print("num_elements =", elements_np.shape[0])
    print("num_left_faces =", boundary_faces_np["left"].shape[0])
    print("num_right_faces =", boundary_faces_np["right"].shape[0])

    # --------------------------------------------------------
    # FEM 参考解
    # --------------------------------------------------------
    print("\nPrecomputing FEM reference data in parallel...")
    conn_all, V0_all, gradN_all_np, dofs_all_np = precompute_reference_data_parallel_3d(
        nodes_np, elements_np, n_jobs=N_JOBS
    )

    print("Precomputing sparse pattern...")
    rows_all, cols_all = precompute_sparse_pattern_3d(dofs_all_np)

    bc_ux_np, bc_uy_np, bc_uz_np = build_dirichlet_bcs_numpy_3d(nodes_np, boundaries_np)
    fixed_dofs_np, fixed_vals_np = build_fixed_dofs_and_values_from_bc_numpy_3d(bc_ux_np, bc_uy_np, bc_uz_np)

    right_face_data = precompute_surface_traction_faces(boundary_faces_np["right"], nodes_np)
    f_ext_full = assemble_external_force_right_face(nodes_np, right_face_data, qy=qy)

    print("Solving nonlinear hyperelastic FEM reference with analytic tangent...")
    fem_result = solve_hyperelastic_newton_parallel_analytic_3d(
        nodes=nodes_np,
        elements=elements_np,
        dofs_all=dofs_all_np,
        gradN_all=gradN_all_np,
        V0_all=V0_all,
        rows_all=rows_all,
        cols_all=cols_all,
        mu=mu,
        lam=lam,
        fixed_dofs=fixed_dofs_np,
        fixed_vals=fixed_vals_np,
        f_ext_full=f_ext_full,
        n_steps=n_steps,
        newton_max_iter=newton_max_iter,
        newton_tol_res=newton_tol_res,
        newton_tol_du=newton_tol_du,
        adaptive=fem_adaptive,
        min_load_increment=fem_min_load_increment,
        max_load_increment=fem_max_load_increment,
        cutback_factor=fem_cutback_factor,
        growth_factor=fem_growth_factor,
        easy_iter_threshold=fem_easy_iter_threshold,
        hard_iter_threshold=fem_hard_iter_threshold,
        max_cutbacks=fem_max_cutbacks,
        n_jobs=N_JOBS,
        verbose=True
    )

    u_fem = fem_result["u"]
    sigma_fem = fem_result["sigma"]
    vm_fem = fem_result["von_mises"]
    J_fem = fem_result["J"]
    f_int_fem = fem_result["f_int"]
    fem_solve_time = fem_result["fem_solve_time"]

    print("\nFEM adaptive stepping summary:")
    print(f"  accepted_steps = {fem_result['accepted_steps']}")
    print(f"  total_attempts = {fem_result['total_attempts']}")
    print(f"  total_cutbacks = {fem_result['total_cutbacks']}")
    print(f"  initial dLambda = {fem_result['initial_load_increment']:.6e}")
    print(f"  min dLambda = {fem_result['min_load_increment']:.6e}")
    print(f"  max dLambda = {fem_result['max_load_increment']:.6e}")

    reaction_stats_fem = summarize_boundary_reactions_numpy_3d(
        f_int_fem,
        {
            "left": boundaries_np["left"],
            "right": boundaries_np["right"],
            "free": boundaries_np["free"],
        }
    )

    print("\nFEM reaction summary:")
    for k, v in reaction_stats_fem.items():
        print(f"{k:15s}: {v:.6e}")

    # 计算 FEM 参考势能（无量纲）
    nodes_torch_tmp = torch.tensor(nodes_np, dtype=torch.float32, device=device)
    elements_torch_tmp = torch.tensor(elements_np, dtype=torch.long, device=device)
    boundary_faces_right_tmp = torch.tensor(boundary_faces_np["right"], dtype=torch.long, device=device)

    _, elem_vol_tilde_tmp, elem_gradN_tmp, _ = precompute_reference_tetra_data_torch(
        nodes_torch_tmp, elements_torch_tmp, Ls
    )

    u_fem_tilde_torch = torch.tensor(u_fem / us, dtype=torch.float32, device=device)

    with torch.no_grad():
        Pi_fem_tilde, U_fem_tilde, W_fem_tilde = compute_potential_energy_nondim_3d(
            u_pred_tilde=u_fem_tilde_torch,
            elements=elements_torch_tmp,
            elem_vol_tilde=elem_vol_tilde_tmp,
            elem_gradN=elem_gradN_tmp,
            right_faces_torch=boundary_faces_right_tmp,
            nodes_torch=nodes_torch_tmp,
            qy=qy,
            mu_tilde=mu_tilde,
            lam_tilde=lam_tilde,
            scales=scales
        )
        Pi_fem_tilde = float(Pi_fem_tilde.item())

    print(f"\nFEM nondimensional potential energy Pi_fem_tilde = {Pi_fem_tilde:.12e}")
    print(f"FEM solve time = {fem_solve_time:.6f} s")

    # --------------------------------------------------------
    # PyG 数据与网络
    # --------------------------------------------------------
    nodes_torch = nodes_torch.to(device)
    elements_torch = elements_torch.to(device)
    boundaries_torch = {k: v.to(device) for k, v in boundaries_torch.items()}
    boundary_faces_torch = {k: v.to(device) for k, v in boundary_faces_torch.items()}

    graph_build_start = time.perf_counter()
    data = build_pyg_data_3d(
        nodes_torch,
        elements_torch,
        boundaries_torch,
        Ls
    )
    graph_build_time = time.perf_counter() - graph_build_start
    data = data.to(device)

    print(f"Vectorized graph construction time = {graph_build_time:.6f} s")
    print(f"num_graph_edges_directed = {data.edge_index.shape[1]}")

    elem_vol, elem_vol_tilde, elem_gradN_torch, elem_dofs_torch = precompute_reference_tetra_data_torch(
        nodes_torch, elements_torch, Ls
    )

    bc_ux_tilde, bc_uy_tilde, bc_uz_tilde = build_displacement_bc_values_3d(
        num_nodes=nodes_torch.shape[0],
        prescribed_node_values=[
            (boundaries_torch["left"], 0.0, 0.0, 0.0),
        ]
    )

    mask_ux, mask_uy, mask_uz, val_ux, val_uy, val_uz = build_hard_bc_masks_and_values_3d(
        nodes_torch.shape[0], bc_ux_tilde, bc_uy_tilde, bc_uz_tilde
    )

    model = PINN_MPNN_PyG(
        node_in_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        hidden_dim=128,
        mpnn_layers=4,
        dropout=0.0,
        output_scale=1e-4
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=adamw_lr,
        weight_decay=adamw_weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        threshold_mode='rel',
        cooldown=scheduler_cooldown,
        min_lr=adamw_min_lr
    )

    print("\n================= Training MPNN by pure potential minimization =================")
    history, training_time = train_model_pure_energy_pyg(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        mask_ux=mask_ux,
        mask_uy=mask_uy,
        mask_uz=mask_uz,
        val_ux=val_ux,
        val_uy=val_uy,
        val_uz=val_uz,
        elements_torch=elements_torch,
        elem_vol_tilde=elem_vol_tilde,
        elem_gradN_torch=elem_gradN_torch,
        boundary_faces_torch=boundary_faces_torch,
        nodes_torch=nodes_torch,
        qy=qy,
        mu_tilde=mu_tilde,
        lam_tilde=lam_tilde,
        scales=scales,
        u_fem=u_fem,
        Pi_fem_tilde=Pi_fem_tilde,
        epochs=adamw_epochs
    )
    print("================= Training finished =================\n")
    print(f"Training time = {training_time:.6f} s")

    # --------------------------------------------------------
    # 最终预测 + 推理时间
    # --------------------------------------------------------
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()
    with torch.no_grad():
        raw_u_tilde = model(data)
        u_pred_tilde = apply_hard_bc_3d(raw_u_tilde, mask_ux, mask_uy, mask_uz, val_ux, val_uy, val_uz)
        u_pred = (u_pred_tilde * scales["us"]).detach().cpu().numpy()

        sigma_pred_t, vm_pred_t, J_pred_t = compute_element_stress_neo_hookean_from_tilde_3d(
            u_pred_tilde=u_pred_tilde,
            elements=elements_torch,
            elem_gradN=elem_gradN_torch,
            mu_tilde=mu_tilde,
            lam_tilde=lam_tilde,
            scales=scales
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - inference_start

    print(f"Inference time = {inference_time:.6f} s")

    sigma_pred = sigma_pred_t.detach().cpu().numpy()
    vm_pred = vm_pred_t.detach().cpu().numpy()
    J_pred = J_pred_t.detach().cpu().numpy()

    # --------------------------------------------------------
    # 误差评估
    # --------------------------------------------------------
    metrics = evaluate_errors_3d(
        u_pred=u_pred,
        u_fem=u_fem,
        sigma_pred=sigma_pred,
        sigma_fem=sigma_fem,
        vm_pred=vm_pred,
        vm_fem=vm_fem
    )

    with torch.no_grad():
        Pi_pred_tilde, _, _ = compute_potential_energy_nondim_3d(
            u_pred_tilde=u_pred_tilde,
            elements=elements_torch,
            elem_vol_tilde=elem_vol_tilde,
            elem_gradN=elem_gradN_torch,
            right_faces_torch=boundary_faces_torch["right"],
            nodes_torch=nodes_torch,
            qy=qy,
            mu_tilde=mu_tilde,
            lam_tilde=lam_tilde,
            scales=scales
        )
        Pi_pred_tilde = float(Pi_pred_tilde.item())

    rel_Pi_final = relative_energy_error(Pi_pred_tilde, Pi_fem_tilde)

    print("\n================= FINAL ERROR METRICS =================")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6e}")
    print(f"{'rel_Pi':20s}: {rel_Pi_final:.6e}")
    print("=======================================================\n")

    # --------------------------------------------------------
    # 结果表格
    # --------------------------------------------------------
    summary_rows = [{
        "Case": "Cook membrane 3D / PureEnergy-MPNN-PyG",
        "Rel. L2 err. (|u|)": metrics["umag_rel_l2"],
        "Rel. L2 err. (|σvm|)": metrics["vm_rel_l2"],
        "Rel(Π)": rel_Pi_final,
        "Max err. (|u|)": metrics["umag_max_abs_err"],
        "Max err. (|σvm|)": metrics["vm_max_abs_err"],
        "Training time": training_time,
        "Inference time": inference_time,
        "FEM solve time": fem_solve_time,
    }]

    save_summary_table(
        summary_rows,
        save_csv_path=summary_csv_path,
        save_txt_path=summary_txt_path
    )

    # --------------------------------------------------------
    # 训练曲线
    # --------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 11.0))

    ax = axes[0]
    ax.plot(history["potential_tilde"], label=r"Potential $\tilde{\Pi}$", linewidth=1.5, color=COLOR_BLACK)
    ax.plot(history["U_int_tilde"], label=r"$\tilde{U}_{int}$", linewidth=1.2, color=COLOR_RED)
    ax.plot(history["W_ext_tilde"], label=r"$\tilde{W}_{ext}$", linewidth=1.2, color=COLOR_BLUE)
    ax.set_xlabel("Epoch", **LABEL_KW)
    ax.set_ylabel("Value", **LABEL_KW)
    ax.set_title("Pure energy minimization history", **TITLE_KW)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75", facecolor="white", loc="best")
    leg.get_frame().set_linewidth(0.8)
    style_axis_paper(ax, equal=False)

    ax = axes[1]
    ax.plot(history["ux_rel_l2"], label=r"$u_x$ rel-L2", linewidth=1.2, color=COLOR_BLUE)
    ax.plot(history["uy_rel_l2"], label=r"$u_y$ rel-L2", linewidth=1.2, color=COLOR_RED)
    ax.plot(history["uz_rel_l2"], label=r"$u_z$ rel-L2", linewidth=1.2, color=COLOR_GREEN)
    ax.plot(history["umag_rel_l2"], label=r"$|\mathbf{u}|$ rel-L2", linewidth=1.2, color=COLOR_PURPLE)
    ax.set_xlabel("Epoch", **LABEL_KW)
    ax.set_ylabel("Relative error", **LABEL_KW)
    ax.set_title("Relative displacement errors against FEM", **TITLE_KW)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75", facecolor="white", loc="best")
    leg.get_frame().set_linewidth(0.8)
    style_axis_paper(ax, equal=False)

    ax = axes[2]
    ax.plot(history["rel_Pi"], label=r"Rel($\tilde{\Pi}$)", linewidth=1.3, color=COLOR_BLACK)
    ax.set_xlabel("Epoch", **LABEL_KW)
    ax.set_ylabel("Relative error", **LABEL_KW)
    ax.set_title("Relative potential-energy error against FEM", **TITLE_KW)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75", facecolor="white", loc="best")
    leg.get_frame().set_linewidth(0.8)
    style_axis_paper(ax, equal=False)

    ax = axes[3]
    ax.plot(history["lr"], label="Learning rate", linewidth=1.2, color=COLOR_PURPLE)
    ax.set_xlabel("Epoch", **LABEL_KW)
    ax.set_ylabel("LR", **LABEL_KW)
    ax.set_title("Learning-rate schedule", **TITLE_KW)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75", facecolor="white", loc="best")
    leg.get_frame().set_linewidth(0.8)
    style_axis_paper(ax, equal=False)

    fig.tight_layout()
    savefig_paper(fig, os.path.join(output_dir, "training_history.png"))

    # --------------------------------------------------------
    # 导出 VTU
    # --------------------------------------------------------
    export_to_vtu_3d_smoothed(
        save_path=vtu_path,
        nodes=nodes_np,
        elements=elements_np,
        u_pred=u_pred,
        u_fem=u_fem,
        sigma_pred=sigma_pred,
        sigma_fem=sigma_fem,
        vm_pred=vm_pred,
        vm_fem=vm_fem,
        J_pred=J_pred,
        J_fem=J_fem
    )

    # --------------------------------------------------------
    # 交互式选视角 / 默认视角
    # --------------------------------------------------------
    if pick_camera_first:
        camera_position = pick_camera_interactively(
            vtu_path=vtu_path,
            field="umag_pred",
            vector_name="u_pred",
            scale=pyvista_scale,
            smooth_subdiv=pyvista_smooth_subdiv,
            show_edges=False
        )
        save_camera_position_to_txt(camera_position, camera_txt_path)
    else:
        camera_position = get_saved_camera_position()
        save_camera_position_to_txt(camera_position, camera_txt_path)

    # --------------------------------------------------------
    # 完整出图
    # --------------------------------------------------------
    print("Generating full PyVista paper-style figures with fixed camera...")
    make_all_pyvista_figures_paper(
        vtu_path=vtu_path,
        fig_dir=fig_dir,
        scale=pyvista_scale,
        smooth_subdiv=pyvista_smooth_subdiv,
        camera_position=camera_position
    )

    # --------------------------------------------------------
    # 保存文本与模型
    # --------------------------------------------------------
    with open(os.path.join(output_dir, "error_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("FINAL ERROR METRICS\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.12e}\n")
        f.write(f"rel_Pi: {rel_Pi_final:.12e}\n")
        f.write(f"Pi_pred_tilde: {Pi_pred_tilde:.12e}\n")
        f.write(f"Pi_fem_tilde: {Pi_fem_tilde:.12e}\n")
        f.write(f"training_time: {training_time:.12e}\n")
        f.write(f"inference_time: {inference_time:.12e}\n")
        f.write(f"fem_solve_time: {fem_solve_time:.12e}\n")
        f.write(f"graph_build_time: {graph_build_time:.12e}\n")

    with open(os.path.join(output_dir, "fem_reaction_summary.txt"), "w", encoding="utf-8") as f:
        f.write("FEM REACTION SUMMARY\n")
        for k, v in reaction_stats_fem.items():
            f.write(f"{k}: {v:.12e}\n")

    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": metrics,
        "summary_table": summary_rows,
        "Pi_pred_tilde": Pi_pred_tilde,
        "Pi_fem_tilde": Pi_fem_tilde,
        "rel_Pi_final": rel_Pi_final,
        "training_time": training_time,
        "inference_time": inference_time,
        "fem_solve_time": fem_solve_time,
        "graph_build_time": graph_build_time,
        "u_pred": torch.tensor(u_pred, dtype=torch.float32),
        "u_fem": torch.tensor(u_fem, dtype=torch.float32),
        "sigma_pred": torch.tensor(sigma_pred, dtype=torch.float32),
        "sigma_fem": torch.tensor(sigma_fem, dtype=torch.float32),
        "vm_pred": torch.tensor(vm_pred, dtype=torch.float32),
        "vm_fem": torch.tensor(vm_fem, dtype=torch.float32),
        "J_pred": torch.tensor(J_pred, dtype=torch.float32),
        "J_fem": torch.tensor(J_fem, dtype=torch.float32),
        "nodes": torch.tensor(nodes_np, dtype=torch.float32),
        "elements": torch.tensor(elements_np, dtype=torch.long),
        "fem_step_logs": fem_result["step_logs"],
        "reaction_stats_fem": reaction_stats_fem,
        "camera_position": camera_position,
        "L": L,
        "H1": H1,
        "H2": H2,
        "B": B,
        "lam": lam,
        "mu": mu,
        "qy": qy,
        "training_mode": "pure_potential_energy_minimization",
        "graph_backend": "PyTorch_Geometric_vectorized_graph_build",
        "fem_backend": "SciPy_sparse_analytic_tangent",
        "torch_reference_geom_backend": "NumPy_CPU_precompute",
        "torch_energy_backend": "vectorized_manual_det_inv",
        "view_mode": "interactive_picked_camera"
    }, os.path.join(output_dir, "cook_membrane_3d_pure_energy_model_fullfig_pickcamera.pt"))

    print(f"All results saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
