import os
import time
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import gmsh
import meshio
import imageio.v2 as imageio
from PIL import Image

import torch
import torch.nn as nn

import pyvista as pv

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# 如在无图形界面服务器上运行，可取消下一行注释
# pv.start_xvfb()
pv.OFF_SCREEN = True


# ============================================================
# 全局 Matplotlib 论文风格
# ============================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 130,
    "savefig.dpi": 320,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# ============================================================
# PyVista 科研绘图全局参数
# ============================================================
PV_THEME_BACKGROUND = "white"
PV_THEME_CMAP = "turbo"
PV_WINDOW_SIZE = (2600, 1900)

PV_CAMERA_POSITION = [
    (3.8, -2.6, 2.4),
    (1.0, 0.2, 0.2),
    (0.0, 0.0, 1.0)
]

PV_LINE_WIDTH = 0.6
PV_FONT_FAMILY = "times"

PV_FONT_SIZE_TITLE = 50
PV_FONT_SIZE_LABEL = 50
PV_FONT_SIZE_TEXT = 50
PV_FONT_SIZE_FRAME = 50
PV_FONT_SIZE_FRAME_TOP = 50

PV_SCALAR_BAR_WIDTH = 0.12
PV_SCALAR_BAR_HEIGHT = 0.78
PV_SCALAR_BAR_POS_X = 0.84
PV_SCALAR_BAR_POS_Y = 0.11


# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def savefig_paper(fig, save_path_png):
    fig.savefig(save_path_png, bbox_inches="tight", pad_inches=0.03, dpi=320)
    plt.close(fig)


def style_axis_paper(ax, equal=False):
    if equal:
        ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=0.8,
        length=3.5,
        top=False,
        right=False,
        bottom=True,
        left=True
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        width=0.6,
        length=2.2,
        top=False,
        right=False,
        bottom=True,
        left=True
    )
    ax.minorticks_on()


def relative_l2_error(pred, ref, eps=1e-12):
    return (torch.sqrt(torch.sum((pred - ref) ** 2)) /
            (torch.sqrt(torch.sum(ref ** 2)) + eps)).item()


def max_abs_error(pred, ref):
    return torch.max(torch.abs(pred - ref)).item()


def resize_image_to_target(img, target_w, target_h):
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil_img)


def build_training_video(frames_dir, output_video_path, fps=10):
    frame_files = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".png")
    ])

    if len(frame_files) == 0:
        print("No frames found, skip video generation.")
        return

    try:
        first_img = imageio.imread(frame_files[0])
        h0, w0 = first_img.shape[:2]

        target_h = int(np.ceil(h0 / 16) * 16)
        target_w = int(np.ceil(w0 / 16) * 16)

        with imageio.get_writer(
            output_video_path,
            fps=fps,
            codec="libx264",
            macro_block_size=16
        ) as writer:
            for f in frame_files:
                img = imageio.imread(f)
                img = resize_image_to_target(img, target_w, target_h)
                writer.append_data(img)

        print(f"Training video saved to: {output_video_path}")
    except Exception as e:
        print(f"Failed to build MP4 video: {e}")


def build_training_gif(frames_dir, output_gif_path, fps=10):
    frame_files = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".png")
    ])

    if len(frame_files) == 0:
        print("No frames found, skip GIF generation.")
        return

    duration = 1.0 / fps
    images = []

    try:
        first_img = imageio.imread(frame_files[0])
        target_h, target_w = first_img.shape[:2]

        for f in frame_files:
            img = imageio.imread(f)
            img = resize_image_to_target(img, target_w, target_h)
            images.append(img)

        imageio.mimsave(output_gif_path, images, duration=duration, loop=0)
        print(f"Training GIF saved to: {output_gif_path}")
    except Exception as e:
        print(f"Failed to build GIF: {e}")


# ============================================================
# 1. Gmsh 生成三维悬臂梁网格
# ============================================================
def generate_gmsh_cantilever_beam_3d(
    msh_path,
    Lx=2.0,
    Ly=0.4,
    Lz=0.4,
    lc=0.15
):
    gmsh.initialize()
    gmsh.model.add("cantilever_beam_3d")

    gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)

    left_surfs = []
    right_surfs = []
    top_surfs = []
    bottom_surfs = []
    front_surfs = []
    back_surfs = []

    tol = 1e-8
    for dim, tag in surfaces:
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)

        if abs(x - 0.0) < tol:
            left_surfs.append(tag)
        elif abs(x - Lx) < tol:
            right_surfs.append(tag)
        elif abs(y - 0.0) < tol:
            front_surfs.append(tag)
        elif abs(y - Ly) < tol:
            back_surfs.append(tag)
        elif abs(z - 0.0) < tol:
            bottom_surfs.append(tag)
        elif abs(z - Lz) < tol:
            top_surfs.append(tag)

    volumes = gmsh.model.getEntities(dim=3)
    vol_tags = [v[1] for v in volumes]

    gmsh.model.addPhysicalGroup(2, left_surfs, 1)
    gmsh.model.setPhysicalName(2, 1, "left")

    gmsh.model.addPhysicalGroup(2, right_surfs, 2)
    gmsh.model.setPhysicalName(2, 2, "right")

    gmsh.model.addPhysicalGroup(2, top_surfs, 3)
    gmsh.model.setPhysicalName(2, 3, "top")

    gmsh.model.addPhysicalGroup(2, bottom_surfs, 4)
    gmsh.model.setPhysicalName(2, 4, "bottom")

    gmsh.model.addPhysicalGroup(2, front_surfs, 5)
    gmsh.model.setPhysicalName(2, 5, "front")

    gmsh.model.addPhysicalGroup(2, back_surfs, 6)
    gmsh.model.setPhysicalName(2, 6, "back")

    gmsh.model.addPhysicalGroup(3, vol_tags, 10)
    gmsh.model.setPhysicalName(3, 10, "domain")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    gmsh.finalize()


# ============================================================
# 2. 读取 Gmsh 三维网格
# ============================================================
def load_gmsh_mesh_3d(msh_path):
    mesh = meshio.read(msh_path)

    points = mesh.points[:, :3]
    nodes = torch.tensor(points, dtype=torch.float32)

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
        raise ValueError("No tetra blocks found in msh file.")
    if len(tri_blocks) == 0:
        raise ValueError("No boundary triangle blocks found in msh file.")

    tet_cells = np.vstack(tetra_blocks)
    tri_cells = np.vstack(tri_blocks)
    tri_tags = np.concatenate(tri_tags_blocks)

    elements = torch.tensor(tet_cells, dtype=torch.long)

    boundary_faces_dict = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": [],
        "front": [],
        "back": [],
    }

    for face, tag in zip(tri_cells, tri_tags):
        face = face.tolist()
        if tag == 1:
            boundary_faces_dict["left"].append(face)
        elif tag == 2:
            boundary_faces_dict["right"].append(face)
        elif tag == 3:
            boundary_faces_dict["top"].append(face)
        elif tag == 4:
            boundary_faces_dict["bottom"].append(face)
        elif tag == 5:
            boundary_faces_dict["front"].append(face)
        elif tag == 6:
            boundary_faces_dict["back"].append(face)

    for k in boundary_faces_dict:
        if len(boundary_faces_dict[k]) == 0:
            boundary_faces_dict[k] = torch.empty((0, 3), dtype=torch.long)
        else:
            boundary_faces_dict[k] = torch.tensor(boundary_faces_dict[k], dtype=torch.long)

    print("---- Gmsh boundary face statistics ----")
    for k in boundary_faces_dict:
        print(f"{k:10s}: {boundary_faces_dict[k].shape[0]}")
    print("---------------------------------------")

    return nodes, elements, boundary_faces_dict


def boundary_nodes_from_faces(boundary_faces_dict):
    boundaries = {}
    for name, faces in boundary_faces_dict.items():
        if faces.numel() == 0:
            boundaries[name] = torch.empty(0, dtype=torch.long)
        else:
            boundaries[name] = torch.unique(faces.flatten())
    return boundaries


# ============================================================
# 3. 图构建
# ============================================================
def build_graph_from_tetra(elements):
    edge_set = set()
    for e in elements.tolist():
        pairs = [
            (e[0], e[1]), (e[0], e[2]), (e[0], e[3]),
            (e[1], e[2]), (e[1], e[3]), (e[2], e[3])
        ]
        for i, j in pairs:
            a, b = min(i, j), max(i, j)
            edge_set.add((a, b))

    senders, receivers = [], []
    for i, j in edge_set:
        senders += [i, j]
        receivers += [j, i]

    return torch.tensor([senders, receivers], dtype=torch.long)


def build_node_features_3d(nodes, boundaries, Ls):
    x = nodes[:, 0:1] / Ls
    y = nodes[:, 1:2] / Ls
    z = nodes[:, 2:3] / Ls

    def build_flag(name):
        flag = torch.zeros((nodes.shape[0], 1), dtype=torch.float32)
        if name in boundaries and boundaries[name].numel() > 0:
            flag[boundaries[name]] = 1.0
        return flag

    left_flag = build_flag("left")
    right_flag = build_flag("right")
    top_flag = build_flag("top")
    bottom_flag = build_flag("bottom")
    front_flag = build_flag("front")
    back_flag = build_flag("back")

    return torch.cat([
        x, y, z,
        left_flag, right_flag, top_flag, bottom_flag, front_flag, back_flag
    ], dim=1).to(device)


def build_edge_features_3d(nodes, edge_index, Ls):
    src = edge_index[0]
    dst = edge_index[1]

    dx = (nodes[dst, 0] - nodes[src, 0]).unsqueeze(1) / Ls
    dy = (nodes[dst, 1] - nodes[src, 1]).unsqueeze(1) / Ls
    dz = (nodes[dst, 2] - nodes[src, 2]).unsqueeze(1) / Ls
    dist = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + 1e-12)

    return torch.cat([dx, dy, dz, dist], dim=1).to(device)


# ============================================================
# 4. MPNN
# ============================================================
class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.0):
        super().__init__()
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

    def forward(self, h, edge_index, edge_attr):
        src = edge_index[0]
        dst = edge_index[1]

        m_in = torch.cat([h[src], h[dst], edge_attr], dim=1)
        m = self.message_mlp(m_in)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)

        dh = self.update_mlp(torch.cat([h, agg], dim=1))
        dh = self.dropout(dh)

        return self.norm(h + dh)


class PINN_MPNN_Local_3D(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_dim,
        hidden_dim=128,
        mpnn_layers=6,
        dropout=0.0
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim, dropout=dropout)
            for _ in range(mpnn_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, node_feat, edge_index, edge_attr):
        h0 = self.encoder(node_feat)
        h = h0

        for layer in self.mpnn_layers:
            h = layer(h, edge_index, edge_attr)

        h = h + h0
        return self.decoder(h)


# ============================================================
# 5. 三维线弹性 FEM
# ============================================================
def constitutive_matrix_3d(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    D = torch.tensor([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]
    ], dtype=torch.float32, device=device)

    return D


def constitutive_matrix_3d_nondim(nu):
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1.0 / (2 * (1 + nu))

    D = torch.tensor([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]
    ], dtype=torch.float32, device=device)

    return D


def tet4_B_matrix_and_volume(xe):
    dev = xe.device
    dtype = xe.dtype

    x1, y1, z1 = xe[0]
    x2, y2, z2 = xe[1]
    x3, y3, z3 = xe[2]
    x4, y4, z4 = xe[3]

    one = torch.tensor(1.0, dtype=dtype, device=dev)

    M = torch.stack([
        torch.stack([one, x1, y1, z1]),
        torch.stack([one, x2, y2, z2]),
        torch.stack([one, x3, y3, z3]),
        torch.stack([one, x4, y4, z4]),
    ], dim=0)

    detM = torch.det(M)
    V = torch.abs(detM) / 6.0

    # 用 linalg.inv 替代 inverse
    invM = torch.linalg.inv(M)

    b = invM[1, :]
    c = invM[2, :]
    d = invM[3, :]

    B = torch.zeros((6, 12), dtype=dtype, device=dev)

    for i in range(4):
        B[0, 3 * i + 0] = b[i]
        B[1, 3 * i + 1] = c[i]
        B[2, 3 * i + 2] = d[i]

        B[3, 3 * i + 0] = c[i]
        B[3, 3 * i + 1] = b[i]

        B[4, 3 * i + 1] = d[i]
        B[4, 3 * i + 2] = c[i]

        B[5, 3 * i + 0] = d[i]
        B[5, 3 * i + 2] = b[i]

    return B, V


def tet4_element_stiffness(xe, D):
    B, V = tet4_B_matrix_and_volume(xe)
    Ke = V * (B.T @ D @ B)
    return Ke, B, V


def tet4_element_stiffness_nondim(xe, D_tilde, Ls):
    B, V = tet4_B_matrix_and_volume(xe)
    B_tilde = Ls * B
    V_tilde = V / (Ls ** 3)
    Ke_tilde = V_tilde * (B_tilde.T @ D_tilde @ B_tilde)
    return Ke_tilde, B_tilde, V_tilde


def precompute_fem_quantities_3d(nodes, elements, E, nu, Ls):
    # FEM 预处理统一放在 CPU，避免 CUDA 小矩阵求逆导致 cusolver 报错
    cpu_device = torch.device("cpu")

    nodes_cpu = nodes.detach().to(cpu_device)
    elements_cpu = elements.detach().to(cpu_device)

    D_cpu = constitutive_matrix_3d(E, nu).to(cpu_device)
    D_tilde_cpu = constitutive_matrix_3d_nondim(nu).to(cpu_device)

    elem_dofs = []
    elem_B = []
    elem_V = []
    elem_Ke = []

    elem_B_tilde = []
    elem_V_tilde = []
    elem_Ke_tilde = []

    for e in range(elements_cpu.shape[0]):
        conn = elements_cpu[e]
        xe = nodes_cpu[conn]

        Ke, B, V = tet4_element_stiffness(xe, D_cpu)
        Ke_tilde, B_tilde, V_tilde = tet4_element_stiffness_nondim(xe, D_tilde_cpu, Ls)

        dofs = []
        for nid in conn.tolist():
            dofs.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

        elem_dofs.append(torch.tensor(dofs, dtype=torch.long, device=cpu_device))
        elem_B.append(B)
        elem_V.append(V)
        elem_Ke.append(Ke)

        elem_B_tilde.append(B_tilde)
        elem_V_tilde.append(V_tilde)
        elem_Ke_tilde.append(Ke_tilde)

    # 预处理结束后再搬到全局 device
    elem_dofs = torch.stack(elem_dofs, dim=0).to(device)
    elem_B = torch.stack(elem_B, dim=0).to(device)
    elem_V = torch.stack(elem_V, dim=0).to(device)
    elem_Ke = torch.stack(elem_Ke, dim=0).to(device)

    elem_B_tilde = torch.stack(elem_B_tilde, dim=0).to(device)
    elem_V_tilde = torch.stack(elem_V_tilde, dim=0).to(device)
    elem_Ke_tilde = torch.stack(elem_Ke_tilde, dim=0).to(device)

    D = D_cpu.to(device)
    D_tilde = D_tilde_cpu.to(device)

    return (
        elem_dofs,
        elem_B,
        elem_V,
        elem_Ke,
        elem_B_tilde,
        elem_V_tilde,
        elem_Ke_tilde,
        D,
        D_tilde
    )


# ============================================================
# 6. 外力组装（向量化）
# ============================================================
def build_external_force_from_faces(nodes, boundary_tractions):
    ndof = nodes.shape[0] * 3
    f = torch.zeros(ndof, dtype=torch.float32, device=device)

    for faces, tx, ty, tz in boundary_tractions:
        if faces.numel() == 0:
            continue

        n1 = faces[:, 0]
        n2 = faces[:, 1]
        n3 = faces[:, 2]

        x1 = nodes[n1]
        x2 = nodes[n2]
        x3 = nodes[n3]

        area = 0.5 * torch.linalg.norm(torch.cross(x2 - x1, x3 - x1, dim=1), dim=1)
        tvec = torch.tensor([tx, ty, tz], dtype=torch.float32, device=device).view(1, 3)

        fe_face = (area / 3.0).unsqueeze(1) * tvec
        fe_all = torch.cat([fe_face, fe_face, fe_face], dim=1)  # [nf, 9]

        dofs = torch.stack([
            3 * n1, 3 * n1 + 1, 3 * n1 + 2,
            3 * n2, 3 * n2 + 1, 3 * n2 + 2,
            3 * n3, 3 * n3 + 1, 3 * n3 + 2
        ], dim=1).reshape(-1)

        vals = fe_all.reshape(-1)
        f.index_add_(0, dofs, vals)

    return f


def build_external_force_nondim_from_faces(nodes, boundary_tractions_tilde, Ls=1.0):
    ndof = nodes.shape[0] * 3
    f_tilde = torch.zeros(ndof, dtype=torch.float32, device=device)

    for faces, tx, ty, tz in boundary_tractions_tilde:
        if faces.numel() == 0:
            continue

        n1 = faces[:, 0]
        n2 = faces[:, 1]
        n3 = faces[:, 2]

        x1 = nodes[n1]
        x2 = nodes[n2]
        x3 = nodes[n3]

        area = 0.5 * torch.linalg.norm(torch.cross(x2 - x1, x3 - x1, dim=1), dim=1)
        area_tilde = area / (Ls ** 2)

        tvec = torch.tensor([tx, ty, tz], dtype=torch.float32, device=device).view(1, 3)
        fe_face = (area_tilde / 3.0).unsqueeze(1) * tvec
        fe_all = torch.cat([fe_face, fe_face, fe_face], dim=1)  # [nf, 9]

        dofs = torch.stack([
            3 * n1, 3 * n1 + 1, 3 * n1 + 2,
            3 * n2, 3 * n2 + 1, 3 * n2 + 2,
            3 * n3, 3 * n3 + 1, 3 * n3 + 2
        ], dim=1).reshape(-1)

        vals = fe_all.reshape(-1)
        f_tilde.index_add_(0, dofs, vals)

    return f_tilde


# ============================================================
# 7. 硬边界条件
# ============================================================
def build_hard_bc_masks_3d(num_nodes, left_nodes):
    mask_ux = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
    mask_uy = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
    mask_uz = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

    mask_ux[left_nodes, 0] = 0.0
    mask_uy[left_nodes, 0] = 0.0
    mask_uz[left_nodes, 0] = 0.0

    return mask_ux, mask_uy, mask_uz


def apply_hard_bc_3d(raw_u_tilde, mask_ux, mask_uy, mask_uz):
    ux = raw_u_tilde[:, 0:1] * mask_ux
    uy = raw_u_tilde[:, 1:2] * mask_uy
    uz = raw_u_tilde[:, 2:3] * mask_uz
    return torch.cat([ux, uy, uz], dim=1)


def build_free_dofs_3d(num_nodes, fixed_dofs):
    ndof = num_nodes * 3
    all_dofs = torch.arange(ndof, dtype=torch.long, device=device)
    mask = torch.ones(ndof, dtype=torch.bool, device=device)
    mask[fixed_dofs] = False
    return all_dofs[mask]


# ============================================================
# 8. 物理量计算（向量化）
# ============================================================
def compute_physics_nondim_3d(
    u_pred_tilde,
    elem_dofs,
    elem_B_tilde,
    elem_Ke_tilde,
    D_tilde
):
    ndof = u_pred_tilde.shape[0] * 3
    u_flat_tilde = u_pred_tilde.reshape(-1)

    ue = u_flat_tilde[elem_dofs]  # [ne, 12]

    strains_tilde = torch.bmm(elem_B_tilde, ue.unsqueeze(-1)).squeeze(-1)  # [ne, 6]
    Dt = D_tilde.unsqueeze(0).expand(strains_tilde.shape[0], -1, -1)
    stresses_tilde = torch.bmm(Dt, strains_tilde.unsqueeze(-1)).squeeze(-1)  # [ne, 6]
    fe_tilde = torch.bmm(elem_Ke_tilde, ue.unsqueeze(-1)).squeeze(-1)  # [ne, 12]

    f_int_tilde = torch.zeros(ndof, dtype=torch.float32, device=device)
    f_int_tilde.index_add_(0, elem_dofs.reshape(-1), fe_tilde.reshape(-1))

    return f_int_tilde, strains_tilde, stresses_tilde


def compute_total_potential_energy_nondim(u_pred_tilde, elem_dofs, elem_Ke_tilde, f_ext_tilde):
    u_flat = u_pred_tilde.reshape(-1)
    ue = u_flat[elem_dofs]  # [ne, 12]
    Ke_ue = torch.bmm(elem_Ke_tilde, ue.unsqueeze(-1)).squeeze(-1)
    U_int = 0.5 * torch.sum(ue * Ke_ue)
    W_ext = u_flat @ f_ext_tilde
    Pi = U_int - W_ext
    return Pi, U_int, W_ext


def physics_hardbc_energy_loss_3d(
    u_pred_tilde,
    f_ext_tilde,
    elem_dofs,
    elem_B_tilde,
    elem_Ke_tilde,
    D_tilde,
    free_dofs,
    weights
):
    Pi, U_int, W_ext = compute_total_potential_energy_nondim(
        u_pred_tilde, elem_dofs, elem_Ke_tilde, f_ext_tilde
    )

    f_int_tilde, strains_tilde, stresses_tilde = compute_physics_nondim_3d(
        u_pred_tilde, elem_dofs, elem_B_tilde, elem_Ke_tilde, D_tilde
    )

    residual_tilde = f_int_tilde - f_ext_tilde
    residual_free = residual_tilde[free_dofs]
    loss_weak = torch.mean(residual_free ** 2)

    loss = weights["energy"] * Pi + weights["weak"] * loss_weak

    logs = {
        "loss": float(loss.item()),
        "energy": float(Pi.item()),
        "U_int": float(U_int.item()),
        "W_ext": float(W_ext.item()),
        "weak": float(loss_weak.item()),
    }

    return loss, logs, strains_tilde, stresses_tilde


def compute_element_average_stress_vm_from_tilde_3d(strains_tilde, stresses_tilde, scales):
    sigma_s = scales["sigma_s"]
    stresses = stresses_tilde * sigma_s

    sxx = stresses[:, 0]
    syy = stresses[:, 1]
    szz = stresses[:, 2]
    txy = stresses[:, 3]
    tyz = stresses[:, 4]
    tzx = stresses[:, 5]

    von_mises = torch.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)
    )

    return stresses, von_mises


# ============================================================
# 9. FEM 稀疏参考解
# ============================================================
def assemble_global_stiffness_sparse(num_nodes, elem_dofs, elem_Ke):
    ndof = num_nodes * 3

    rows = elem_dofs[:, :, None].expand(-1, 12, 12).reshape(-1).detach().cpu().numpy()
    cols = elem_dofs[:, None, :].expand(-1, 12, 12).reshape(-1).detach().cpu().numpy()
    vals = elem_Ke.reshape(-1).detach().cpu().numpy()

    if SCIPY_AVAILABLE:
        K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    else:
        K = torch.zeros((ndof, ndof), dtype=torch.float32, device=device)
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=device)
        vals_t = torch.tensor(vals, dtype=torch.float32, device=device)
        K.index_put_((rows_t, cols_t), vals_t, accumulate=True)

    return K


def solve_fem_reference_3d_sparse(num_nodes, elem_dofs, elem_Ke, f_ext, fixed_dofs, fixed_vals):
    ndof = num_nodes * 3

    t0 = time.perf_counter()
    K = assemble_global_stiffness_sparse(num_nodes, elem_dofs, elem_Ke)

    all_dofs = np.arange(ndof)
    fixed_np = fixed_dofs.detach().cpu().numpy()
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[fixed_np] = False
    free_dofs = all_dofs[free_mask]

    f_ext_np = f_ext.detach().cpu().numpy()
    fixed_vals_np = fixed_vals.detach().cpu().numpy()

    if SCIPY_AVAILABLE:
        K_ff = K[free_dofs][:, free_dofs]
        K_fc = K[free_dofs][:, fixed_np]
        rhs = f_ext_np[free_dofs] - K_fc @ fixed_vals_np

        u = np.zeros(ndof, dtype=np.float32)
        u[fixed_np] = fixed_vals_np
        u_free = spla.spsolve(K_ff, rhs)
        u[free_dofs] = u_free.astype(np.float32)

        fem_solve_time = time.perf_counter() - t0
        u_torch = torch.tensor(u, dtype=torch.float32, device=device).reshape(num_nodes, 3)
        return u_torch, K, fem_solve_time
    else:
        print("Warning: scipy not available, fallback to dense solver.")
        all_dofs_t = torch.arange(ndof, dtype=torch.long, device=device)
        fixed_dofs_t = fixed_dofs
        mask = torch.ones(ndof, dtype=torch.bool, device=device)
        mask[fixed_dofs_t] = False
        free_dofs_t = all_dofs_t[mask]

        K_ff = K[free_dofs_t][:, free_dofs_t]
        K_fc = K[free_dofs_t][:, fixed_dofs_t]
        rhs = f_ext[free_dofs_t] - K_fc @ fixed_vals

        u = torch.zeros(ndof, dtype=torch.float32, device=device)
        u[fixed_dofs_t] = fixed_vals
        u_free = torch.linalg.solve(K_ff, rhs)
        u[free_dofs_t] = u_free

        fem_solve_time = time.perf_counter() - t0
        return u.reshape(num_nodes, 3), K, fem_solve_time


def compute_energy_from_physical_solution_3d(u_phys, elem_dofs, elem_Ke, f_ext_phys):
    u_flat = u_phys.reshape(-1)
    ue = u_flat[elem_dofs]
    Ke_ue = torch.bmm(elem_Ke, ue.unsqueeze(-1)).squeeze(-1)
    U_int = 0.5 * torch.sum(ue * Ke_ue)
    W_ext = u_flat @ f_ext_phys
    Pi = U_int - W_ext

    return {
        "Pi": float(Pi.item()),
        "U_int": float(U_int.item()),
        "W_ext": float(W_ext.item()),
    }


# ============================================================
# 10. 误差评估
# ============================================================
def evaluate_errors_3d(u_pred, u_fem, elem_stress_pred, elem_stress_fem, vm_pred, vm_fem):
    ux_pred, uy_pred, uz_pred = u_pred[:, 0], u_pred[:, 1], u_pred[:, 2]
    ux_fem, uy_fem, uz_fem = u_fem[:, 0], u_fem[:, 1], u_fem[:, 2]

    umag_pred = torch.sqrt(ux_pred ** 2 + uy_pred ** 2 + uz_pred ** 2)
    umag_fem = torch.sqrt(ux_fem ** 2 + uy_fem ** 2 + uz_fem ** 2)

    return {
        "ux_rel_l2": relative_l2_error(ux_pred, ux_fem),
        "uy_rel_l2": relative_l2_error(uy_pred, uy_fem),
        "uz_rel_l2": relative_l2_error(uz_pred, uz_fem),
        "umag_rel_l2": relative_l2_error(umag_pred, umag_fem),
        "sxx_rel_l2": relative_l2_error(elem_stress_pred[:, 0], elem_stress_fem[:, 0]),
        "syy_rel_l2": relative_l2_error(elem_stress_pred[:, 1], elem_stress_fem[:, 1]),
        "szz_rel_l2": relative_l2_error(elem_stress_pred[:, 2], elem_stress_fem[:, 2]),
        "txy_rel_l2": relative_l2_error(elem_stress_pred[:, 3], elem_stress_fem[:, 3]),
        "tyz_rel_l2": relative_l2_error(elem_stress_pred[:, 4], elem_stress_fem[:, 4]),
        "tzx_rel_l2": relative_l2_error(elem_stress_pred[:, 5], elem_stress_fem[:, 5]),
        "vm_rel_l2": relative_l2_error(vm_pred, vm_fem),
        "umag_max_abs": max_abs_error(umag_pred, umag_fem),
        "vm_max_abs": max_abs_error(vm_pred, vm_fem),
    }


# ============================================================
# 11. 导出 VTU
# ============================================================
def export_to_vtu_3d(
    save_path,
    nodes,
    elements,
    u_pred_phys,
    u_fem,
    elem_stress_pred,
    von_mises_pred,
    elem_stress_fem,
    von_mises_fem
):
    points = nodes.detach().cpu().numpy().astype(np.float64)
    cells = [("tetra", elements.detach().cpu().numpy())]

    u_pred = u_pred_phys.detach().cpu().numpy()
    u_fem_np = u_fem.detach().cpu().numpy()
    u_error = u_pred - u_fem_np

    umag_pred = np.linalg.norm(u_pred, axis=1)
    umag_fem = np.linalg.norm(u_fem_np, axis=1)

    point_data = {
        "u_pred": u_pred,
        "u_fem": u_fem_np,
        "u_error": u_error,
        "umag_pred": umag_pred,
        "umag_fem": umag_fem,
        "umag_error": np.abs(umag_pred - umag_fem),
    }

    cell_data = {
        "sxx_pred": [elem_stress_pred[:, 0].detach().cpu().numpy()],
        "syy_pred": [elem_stress_pred[:, 1].detach().cpu().numpy()],
        "szz_pred": [elem_stress_pred[:, 2].detach().cpu().numpy()],
        "txy_pred": [elem_stress_pred[:, 3].detach().cpu().numpy()],
        "tyz_pred": [elem_stress_pred[:, 4].detach().cpu().numpy()],
        "tzx_pred": [elem_stress_pred[:, 5].detach().cpu().numpy()],
        "von_mises_pred": [von_mises_pred.detach().cpu().numpy()],
        "sxx_fem": [elem_stress_fem[:, 0].detach().cpu().numpy()],
        "syy_fem": [elem_stress_fem[:, 1].detach().cpu().numpy()],
        "szz_fem": [elem_stress_fem[:, 2].detach().cpu().numpy()],
        "txy_fem": [elem_stress_fem[:, 3].detach().cpu().numpy()],
        "tyz_fem": [elem_stress_fem[:, 4].detach().cpu().numpy()],
        "tzx_fem": [elem_stress_fem[:, 5].detach().cpu().numpy()],
        "von_mises_fem": [von_mises_fem.detach().cpu().numpy()],
    }

    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data)
    meshio.write(save_path, mesh)
    print(f"VTU exported to: {save_path}")


# ============================================================
# 12. 单元场转节点场（向量化）
# ============================================================
def element_to_nodal_field_tetra(elements, elem_field, num_nodes):
    device_local = elem_field.device
    nodal_sum = torch.zeros(num_nodes, dtype=elem_field.dtype, device=device_local)
    nodal_count = torch.zeros(num_nodes, dtype=elem_field.dtype, device=device_local)

    conn_flat = elements.reshape(-1)
    vals = elem_field[:, None].repeat(1, 4).reshape(-1)
    ones = torch.ones_like(vals)

    nodal_sum.index_add_(0, conn_flat, vals)
    nodal_count.index_add_(0, conn_flat, ones)

    nodal_field = nodal_sum / (nodal_count + 1e-12)
    return nodal_field


# ============================================================
# 13. PyVista 论文风格可视化
# ============================================================
def create_paper_plotter(shape=(1, 1), off_screen=True, window_size=PV_WINDOW_SIZE):
    plotter = pv.Plotter(shape=shape, off_screen=off_screen, window_size=window_size)
    plotter.set_background(PV_THEME_BACKGROUND)
    return plotter


def build_pyvista_unstructured_grid(nodes, elements):
    points = nodes.detach().cpu().numpy().astype(np.float64)
    elems = elements.detach().cpu().numpy().astype(np.int64)

    n_cells = elems.shape[0]
    cells = np.hstack([
        np.full((n_cells, 1), 4, dtype=np.int64),
        elems
    ]).ravel()

    celltypes = np.full(n_cells, 10, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    return grid


def make_deformed_grid(grid, vector_name="u_pred", scale=1.0):
    deformed = grid.copy()
    vec = deformed.point_data[vector_name]
    deformed.points = deformed.points + scale * vec
    return deformed


def attach_fields_to_grid_3d(
    grid,
    elements,
    u_pred_phys,
    u_fem,
    elem_stress_pred,
    von_mises_pred,
    elem_stress_fem,
    von_mises_fem
):
    u_pred = u_pred_phys.detach().cpu().numpy()
    u_fem_np = u_fem.detach().cpu().numpy()
    u_error = u_pred - u_fem_np

    umag_pred = np.linalg.norm(u_pred, axis=1)
    umag_fem = np.linalg.norm(u_fem_np, axis=1)
    umag_error = np.abs(umag_pred - umag_fem)

    grid.point_data["u_pred"] = u_pred
    grid.point_data["u_fem"] = u_fem_np
    grid.point_data["u_error"] = u_error

    grid.point_data["ux_pred"] = u_pred[:, 0]
    grid.point_data["uy_pred"] = u_pred[:, 1]
    grid.point_data["uz_pred"] = u_pred[:, 2]

    grid.point_data["ux_fem"] = u_fem_np[:, 0]
    grid.point_data["uy_fem"] = u_fem_np[:, 1]
    grid.point_data["uz_fem"] = u_fem_np[:, 2]

    grid.point_data["ux_error"] = np.abs(u_pred[:, 0] - u_fem_np[:, 0])
    grid.point_data["uy_error"] = np.abs(u_pred[:, 1] - u_fem_np[:, 1])
    grid.point_data["uz_error"] = np.abs(u_pred[:, 2] - u_fem_np[:, 2])

    grid.point_data["umag_pred"] = umag_pred
    grid.point_data["umag_fem"] = umag_fem
    grid.point_data["umag_error"] = umag_error

    spred = elem_stress_pred.detach().cpu().numpy()
    sfem = elem_stress_fem.detach().cpu().numpy()
    vm_p = von_mises_pred.detach().cpu().numpy()
    vm_f = von_mises_fem.detach().cpu().numpy()

    grid.cell_data["sxx_pred"] = spred[:, 0]
    grid.cell_data["syy_pred"] = spred[:, 1]
    grid.cell_data["szz_pred"] = spred[:, 2]
    grid.cell_data["txy_pred"] = spred[:, 3]
    grid.cell_data["tyz_pred"] = spred[:, 4]
    grid.cell_data["tzx_pred"] = spred[:, 5]
    grid.cell_data["von_mises_pred"] = vm_p

    grid.cell_data["sxx_fem"] = sfem[:, 0]
    grid.cell_data["syy_fem"] = sfem[:, 1]
    grid.cell_data["szz_fem"] = sfem[:, 2]
    grid.cell_data["txy_fem"] = sfem[:, 3]
    grid.cell_data["tyz_fem"] = sfem[:, 4]
    grid.cell_data["tzx_fem"] = sfem[:, 5]
    grid.cell_data["von_mises_fem"] = vm_f

    grid.cell_data["sxx_error"] = np.abs(spred[:, 0] - sfem[:, 0])
    grid.cell_data["syy_error"] = np.abs(spred[:, 1] - sfem[:, 1])
    grid.cell_data["szz_error"] = np.abs(spred[:, 2] - sfem[:, 2])
    grid.cell_data["txy_error"] = np.abs(spred[:, 3] - sfem[:, 3])
    grid.cell_data["tyz_error"] = np.abs(spred[:, 4] - sfem[:, 4])
    grid.cell_data["tzx_error"] = np.abs(spred[:, 5] - sfem[:, 5])
    grid.cell_data["von_mises_error"] = np.abs(vm_p - vm_f)

    num_nodes = grid.n_points

    sxx_pred_nodal = element_to_nodal_field_tetra(elements, elem_stress_pred[:, 0], num_nodes)
    syy_pred_nodal = element_to_nodal_field_tetra(elements, elem_stress_pred[:, 1], num_nodes)
    szz_pred_nodal = element_to_nodal_field_tetra(elements, elem_stress_pred[:, 2], num_nodes)
    txy_pred_nodal = element_to_nodal_field_tetra(elements, elem_stress_pred[:, 3], num_nodes)
    tyz_pred_nodal = element_to_nodal_field_tetra(elements, elem_stress_pred[:, 4], num_nodes)
    tzx_pred_nodal = element_to_nodal_field_tetra(elements, elem_stress_pred[:, 5], num_nodes)
    vm_pred_nodal = element_to_nodal_field_tetra(elements, von_mises_pred, num_nodes)

    sxx_fem_nodal = element_to_nodal_field_tetra(elements, elem_stress_fem[:, 0], num_nodes)
    syy_fem_nodal = element_to_nodal_field_tetra(elements, elem_stress_fem[:, 1], num_nodes)
    szz_fem_nodal = element_to_nodal_field_tetra(elements, elem_stress_fem[:, 2], num_nodes)
    txy_fem_nodal = element_to_nodal_field_tetra(elements, elem_stress_fem[:, 3], num_nodes)
    tyz_fem_nodal = element_to_nodal_field_tetra(elements, elem_stress_fem[:, 4], num_nodes)
    tzx_fem_nodal = element_to_nodal_field_tetra(elements, elem_stress_fem[:, 5], num_nodes)
    vm_fem_nodal = element_to_nodal_field_tetra(elements, von_mises_fem, num_nodes)

    grid.point_data["sxx_pred_nodal"] = sxx_pred_nodal.detach().cpu().numpy()
    grid.point_data["syy_pred_nodal"] = syy_pred_nodal.detach().cpu().numpy()
    grid.point_data["szz_pred_nodal"] = szz_pred_nodal.detach().cpu().numpy()
    grid.point_data["txy_pred_nodal"] = txy_pred_nodal.detach().cpu().numpy()
    grid.point_data["tyz_pred_nodal"] = tyz_pred_nodal.detach().cpu().numpy()
    grid.point_data["tzx_pred_nodal"] = tzx_pred_nodal.detach().cpu().numpy()
    grid.point_data["von_mises_pred_nodal"] = vm_pred_nodal.detach().cpu().numpy()

    grid.point_data["sxx_fem_nodal"] = sxx_fem_nodal.detach().cpu().numpy()
    grid.point_data["syy_fem_nodal"] = syy_fem_nodal.detach().cpu().numpy()
    grid.point_data["szz_fem_nodal"] = szz_fem_nodal.detach().cpu().numpy()
    grid.point_data["txy_fem_nodal"] = txy_fem_nodal.detach().cpu().numpy()
    grid.point_data["tyz_fem_nodal"] = tyz_fem_nodal.detach().cpu().numpy()
    grid.point_data["tzx_fem_nodal"] = tzx_fem_nodal.detach().cpu().numpy()
    grid.point_data["von_mises_fem_nodal"] = vm_fem_nodal.detach().cpu().numpy()

    grid.point_data["sxx_error_nodal"] = np.abs(
        grid.point_data["sxx_pred_nodal"] - grid.point_data["sxx_fem_nodal"]
    )
    grid.point_data["syy_error_nodal"] = np.abs(
        grid.point_data["syy_pred_nodal"] - grid.point_data["syy_fem_nodal"]
    )
    grid.point_data["szz_error_nodal"] = np.abs(
        grid.point_data["szz_pred_nodal"] - grid.point_data["szz_fem_nodal"]
    )
    grid.point_data["txy_error_nodal"] = np.abs(
        grid.point_data["txy_pred_nodal"] - grid.point_data["txy_fem_nodal"]
    )
    grid.point_data["tyz_error_nodal"] = np.abs(
        grid.point_data["tyz_pred_nodal"] - grid.point_data["tyz_fem_nodal"]
    )
    grid.point_data["tzx_error_nodal"] = np.abs(
        grid.point_data["tzx_pred_nodal"] - grid.point_data["tzx_fem_nodal"]
    )
    grid.point_data["von_mises_error_nodal"] = np.abs(
        grid.point_data["von_mises_pred_nodal"] - grid.point_data["von_mises_fem_nodal"]
    )

    return grid


def pv_plot_and_screenshot_paper(
    mesh,
    scalar_name,
    save_path,
    title=None,
    cmap=PV_THEME_CMAP,
    show_edges=False,
    clim=None,
    scalar_bar_title=None,
    window_size=PV_WINDOW_SIZE,
    smooth_shading=True,
    lighting=False,
    edge_color="black",
    line_width=PV_LINE_WIDTH
):
    plotter = create_paper_plotter(off_screen=True, window_size=window_size)

    scalar_bar_args = {
        "title": scalar_bar_title if scalar_bar_title is not None else scalar_name,
        "vertical": True,
        "title_font_size": PV_FONT_SIZE_TITLE,
        "label_font_size": PV_FONT_SIZE_LABEL,
        "shadow": False,
        "fmt": "%.2e",
        "font_family": PV_FONT_FAMILY,
        "color": "black",
        "position_x": PV_SCALAR_BAR_POS_X,
        "position_y": PV_SCALAR_BAR_POS_Y,
        "width": PV_SCALAR_BAR_WIDTH,
        "height": PV_SCALAR_BAR_HEIGHT,
        "n_labels": 5,
        "italic": False,
        "bold": False,
    }

    plotter.add_mesh(
        mesh,
        scalars=scalar_name,
        cmap=cmap,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=line_width,
        clim=clim,
        scalar_bar_args=scalar_bar_args,
        smooth_shading=smooth_shading,
        lighting=lighting
    )

    plotter.camera_position = PV_CAMERA_POSITION

    if title is not None:
        plotter.add_text(
            title,
            position=(0.03, 0.93),
            font_size=PV_FONT_SIZE_TEXT,
            color="black",
            font=PV_FONT_FAMILY
        )

    plotter.screenshot(save_path, return_img=False)
    plotter.close()


def pv_plot_slice_and_screenshot_paper(
    mesh,
    scalar_name,
    save_path,
    normal="z",
    origin=None,
    title=None,
    cmap=PV_THEME_CMAP,
    clim=None,
    scalar_bar_title=None,
    window_size=PV_WINDOW_SIZE
):
    if origin is None:
        bounds = mesh.bounds
        xc = 0.5 * (bounds[0] + bounds[1])
        yc = 0.5 * (bounds[2] + bounds[3])
        zc = 0.5 * (bounds[4] + bounds[5])
        origin = (xc, yc, zc)

    slc = mesh.slice(normal=normal, origin=origin)

    plotter = create_paper_plotter(off_screen=True, window_size=window_size)

    scalar_bar_args = {
        "title": scalar_bar_title if scalar_bar_title is not None else scalar_name,
        "vertical": True,
        "title_font_size": PV_FONT_SIZE_TITLE,
        "label_font_size": PV_FONT_SIZE_LABEL,
        "shadow": False,
        "fmt": "%.2e",
        "font_family": PV_FONT_FAMILY,
        "color": "black",
        "position_x": PV_SCALAR_BAR_POS_X,
        "position_y": PV_SCALAR_BAR_POS_Y,
        "width": PV_SCALAR_BAR_WIDTH,
        "height": PV_SCALAR_BAR_HEIGHT,
        "n_labels": 5,
        "italic": False,
        "bold": False,
    }

    plotter.add_mesh(
        slc,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        scalar_bar_args=scalar_bar_args,
        smooth_shading=True,
        lighting=False
    )

    if normal == "x":
        plotter.view_yz()
    elif normal == "y":
        plotter.view_xz()
    elif normal == "z":
        plotter.view_xy()
    else:
        plotter.camera_position = PV_CAMERA_POSITION

    if title is not None:
        plotter.add_text(
            title,
            position=(0.03, 0.93),
            font_size=PV_FONT_SIZE_TEXT,
            color="black",
            font=PV_FONT_FAMILY
        )

    plotter.screenshot(save_path, return_img=False)
    plotter.close()


def pv_plot_mesh_surface_paper(mesh_surface, save_path, title="3D tetrahedral mesh surface"):
    plotter = create_paper_plotter(off_screen=True, window_size=PV_WINDOW_SIZE)

    plotter.add_mesh(
        mesh_surface,
        color="white",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        smooth_shading=False,
        lighting=False
    )

    plotter.camera_position = PV_CAMERA_POSITION

    plotter.add_text(
        title,
        position=(0.03, 0.93),
        font_size=PV_FONT_SIZE_TEXT,
        color="black",
        font=PV_FONT_FAMILY
    )

    plotter.screenshot(save_path, return_img=False)
    plotter.close()


def save_pyvista_screenshots_3d(
    output_dir,
    nodes,
    elements,
    u_pred_phys,
    u_fem,
    elem_stress_pred,
    von_mises_pred,
    elem_stress_fem,
    von_mises_fem,
    Lx,
    Ly,
    Lz
):
    ensure_dir(output_dir)

    grid = build_pyvista_unstructured_grid(nodes, elements)
    grid = attach_fields_to_grid_3d(
        grid,
        elements=elements,
        u_pred_phys=u_pred_phys,
        u_fem=u_fem,
        elem_stress_pred=elem_stress_pred,
        von_mises_pred=von_mises_pred,
        elem_stress_fem=elem_stress_fem,
        von_mises_fem=von_mises_fem
    )

    umag_pred = grid.point_data["umag_pred"]
    umag_fem = grid.point_data["umag_fem"]

    max_disp = max(np.max(umag_pred), np.max(umag_fem), 1e-12)
    deform_scale = 0.12 * max(Lx, Ly, Lz) / max_disp

    grid_pred_def = make_deformed_grid(grid, "u_pred", deform_scale)
    grid_fem_def = make_deformed_grid(grid, "u_fem", deform_scale)

    surface_pred = grid_pred_def.extract_surface().triangulate().compute_normals(inplace=False)
    surface_fem = grid_fem_def.extract_surface().triangulate().compute_normals(inplace=False)
    surface_ref = grid.extract_surface().triangulate()

    point_clims = {
        "ux": (
            min(np.min(grid.point_data["ux_pred"]), np.min(grid.point_data["ux_fem"])),
            max(np.max(grid.point_data["ux_pred"]), np.max(grid.point_data["ux_fem"]))
        ),
        "uy": (
            min(np.min(grid.point_data["uy_pred"]), np.min(grid.point_data["uy_fem"])),
            max(np.max(grid.point_data["uy_pred"]), np.max(grid.point_data["uy_fem"]))
        ),
        "uz": (
            min(np.min(grid.point_data["uz_pred"]), np.min(grid.point_data["uz_fem"])),
            max(np.max(grid.point_data["uz_pred"]), np.max(grid.point_data["uz_fem"]))
        ),
        "umag": (
            0.0,
            max(np.max(grid.point_data["umag_pred"]), np.max(grid.point_data["umag_fem"]))
        ),
    }

    stress_clims = {
        "sxx": (
            min(np.min(grid.point_data["sxx_pred_nodal"]), np.min(grid.point_data["sxx_fem_nodal"])),
            max(np.max(grid.point_data["sxx_pred_nodal"]), np.max(grid.point_data["sxx_fem_nodal"]))
        ),
        "syy": (
            min(np.min(grid.point_data["syy_pred_nodal"]), np.min(grid.point_data["syy_fem_nodal"])),
            max(np.max(grid.point_data["syy_pred_nodal"]), np.max(grid.point_data["syy_fem_nodal"]))
        ),
        "szz": (
            min(np.min(grid.point_data["szz_pred_nodal"]), np.min(grid.point_data["szz_fem_nodal"])),
            max(np.max(grid.point_data["szz_pred_nodal"]), np.max(grid.point_data["szz_fem_nodal"]))
        ),
        "txy": (
            min(np.min(grid.point_data["txy_pred_nodal"]), np.min(grid.point_data["txy_fem_nodal"])),
            max(np.max(grid.point_data["txy_pred_nodal"]), np.max(grid.point_data["txy_fem_nodal"]))
        ),
        "tyz": (
            min(np.min(grid.point_data["tyz_pred_nodal"]), np.min(grid.point_data["tyz_fem_nodal"])),
            max(np.max(grid.point_data["tyz_pred_nodal"]), np.max(grid.point_data["tyz_fem_nodal"]))
        ),
        "tzx": (
            min(np.min(grid.point_data["tzx_pred_nodal"]), np.min(grid.point_data["tzx_fem_nodal"])),
            max(np.max(grid.point_data["tzx_pred_nodal"]), np.max(grid.point_data["tzx_fem_nodal"]))
        ),
        "vm": (
            0.0,
            max(np.max(grid.point_data["von_mises_pred_nodal"]), np.max(grid.point_data["von_mises_fem_nodal"]))
        ),
    }

    pv_plot_mesh_surface_paper(
        surface_ref,
        os.path.join(output_dir, "mesh_surface.png"),
        title="3D tetrahedral mesh surface"
    )

    point_plot_list = [
        ("ux_pred", surface_pred, point_clims["ux"], "ux pred", "ux pred"),
        ("uy_pred", surface_pred, point_clims["uy"], "uy pred", "uy pred"),
        ("uz_pred", surface_pred, point_clims["uz"], "uz pred", "uz pred"),
        ("umag_pred", surface_pred, point_clims["umag"], "|u| pred", "|u| pred"),

        ("ux_fem", surface_fem, point_clims["ux"], "ux FEM", "ux FEM"),
        ("uy_fem", surface_fem, point_clims["uy"], "uy FEM", "uy FEM"),
        ("uz_fem", surface_fem, point_clims["uz"], "uz FEM", "uz FEM"),
        ("umag_fem", surface_fem, point_clims["umag"], "|u| FEM", "|u| FEM"),

        ("ux_error", surface_pred, None, "|ux - ux_FEM|", "ux error"),
        ("uy_error", surface_pred, None, "|uy - uy_FEM|", "uy error"),
        ("uz_error", surface_pred, None, "|uz - uz_FEM|", "uz error"),
        ("umag_error", surface_pred, None, "||u| - |u|_FEM|", "|u| error")
    ]

    for scalar_name, mesh_use, clim, title, cbar_title in point_plot_list:
        pv_plot_and_screenshot_paper(
            mesh=mesh_use,
            scalar_name=scalar_name,
            save_path=os.path.join(output_dir, f"{scalar_name}_surface.png"),
            title=title,
            cmap=PV_THEME_CMAP,
            show_edges=False,
            clim=clim,
            scalar_bar_title=cbar_title
        )

    stress_plot_list = [
        ("sxx_pred_nodal", surface_pred, stress_clims["sxx"], "sxx pred", "sxx pred"),
        ("syy_pred_nodal", surface_pred, stress_clims["syy"], "syy pred", "syy pred"),
        ("szz_pred_nodal", surface_pred, stress_clims["szz"], "szz pred", "szz pred"),
        ("txy_pred_nodal", surface_pred, stress_clims["txy"], "txy pred", "txy pred"),
        ("tyz_pred_nodal", surface_pred, stress_clims["tyz"], "tyz pred", "tyz pred"),
        ("tzx_pred_nodal", surface_pred, stress_clims["tzx"], "tzx pred", "tzx pred"),
        ("von_mises_pred_nodal", surface_pred, stress_clims["vm"], "von Mises pred", "von Mises pred"),

        ("sxx_fem_nodal", surface_fem, stress_clims["sxx"], "sxx FEM", "sxx FEM"),
        ("syy_fem_nodal", surface_fem, stress_clims["syy"], "syy FEM", "syy FEM"),
        ("szz_fem_nodal", surface_fem, stress_clims["szz"], "szz FEM", "szz FEM"),
        ("txy_fem_nodal", surface_fem, stress_clims["txy"], "txy FEM", "txy FEM"),
        ("tyz_fem_nodal", surface_fem, stress_clims["tyz"], "tyz FEM", "tyz FEM"),
        ("tzx_fem_nodal", surface_fem, stress_clims["tzx"], "tzx FEM", "tzx FEM"),
        ("von_mises_fem_nodal", surface_fem, stress_clims["vm"], "von Mises FEM", "von Mises FEM"),

        ("sxx_error_nodal", surface_pred, None, "|sxx - sxx_FEM|", "sxx error"),
        ("syy_error_nodal", surface_pred, None, "|syy - syy_FEM|", "syy error"),
        ("szz_error_nodal", surface_pred, None, "|szz - szz_FEM|", "szz error"),
        ("txy_error_nodal", surface_pred, None, "|txy - txy_FEM|", "txy error"),
        ("tyz_error_nodal", surface_pred, None, "|tyz - tyz_FEM|", "tyz error"),
        ("tzx_error_nodal", surface_pred, None, "|tzx - tzx_FEM|", "tzx error"),
        ("von_mises_error_nodal", surface_pred, None, "|von Mises - von Mises FEM|", "von Mises error")
    ]

    for scalar_name, mesh_use, clim, title, cbar_title in stress_plot_list:
        pv_plot_and_screenshot_paper(
            mesh=mesh_use,
            scalar_name=scalar_name,
            save_path=os.path.join(output_dir, f"{scalar_name}_surface.png"),
            title=title,
            cmap=PV_THEME_CMAP,
            show_edges=False,
            clim=clim,
            scalar_bar_title=cbar_title
        )

    slice_specs = [
        ("z", (Lx / 2, Ly / 2, Lz / 2), "mid_z"),
        ("y", (Lx / 2, Ly / 2, Lz / 2), "mid_y"),
        ("x", (Lx / 2, Ly / 2, Lz / 2), "mid_x"),
    ]

    slice_fields = [
        ("umag_pred", point_clims["umag"], "|u| pred", "|u| pred"),
        ("umag_fem", point_clims["umag"], "|u| FEM", "|u| FEM"),
        ("umag_error", None, "|u| error", "|u| error"),
        ("ux_pred", point_clims["ux"], "ux pred", "ux pred"),
        ("uy_pred", point_clims["uy"], "uy pred", "uy pred"),
        ("uz_pred", point_clims["uz"], "uz pred", "uz pred"),
        ("von_mises_pred_nodal", stress_clims["vm"], "von Mises pred", "von Mises pred"),
        ("von_mises_fem_nodal", stress_clims["vm"], "von Mises FEM", "von Mises FEM"),
        ("von_mises_error_nodal", None, "von Mises error", "von Mises error"),
        ("sxx_pred_nodal", stress_clims["sxx"], "sxx pred", "sxx pred"),
        ("sxx_fem_nodal", stress_clims["sxx"], "sxx FEM", "sxx FEM"),
        ("sxx_error_nodal", None, "sxx error", "sxx error"),
    ]

    for normal, origin, suffix in slice_specs:
        for scalar_name, clim, title_base, cbar_title in slice_fields:
            if "_fem" in scalar_name:
                mesh_use = grid_fem_def
            else:
                mesh_use = grid_pred_def

            pv_plot_slice_and_screenshot_paper(
                mesh=mesh_use,
                scalar_name=scalar_name,
                save_path=os.path.join(output_dir, f"{scalar_name}_{suffix}.png"),
                normal=normal,
                origin=origin,
                title=f"{title_base} ({suffix})",
                cmap=PV_THEME_CMAP,
                clim=clim,
                scalar_bar_title=cbar_title
            )


# ============================================================
# 14. 训练过程论文风格动画帧
# ============================================================
def save_training_frame_pyvista(
    frame_path,
    epoch,
    logs,
    nodes,
    elements,
    u_pred_phys,
    u_fem,
    elem_stress_pred,
    von_mises_pred,
    elem_stress_fem,
    von_mises_fem,
    Lx,
    Ly,
    Lz,
    window_size=PV_WINDOW_SIZE
):
    grid = build_pyvista_unstructured_grid(nodes, elements)
    grid = attach_fields_to_grid_3d(
        grid,
        elements=elements,
        u_pred_phys=u_pred_phys,
        u_fem=u_fem,
        elem_stress_pred=elem_stress_pred,
        von_mises_pred=von_mises_pred,
        elem_stress_fem=elem_stress_fem,
        von_mises_fem=von_mises_fem
    )

    umag_pred = grid.point_data["umag_pred"]
    umag_fem = grid.point_data["umag_fem"]
    vm_pred = grid.point_data["von_mises_pred_nodal"]
    vm_fem = grid.point_data["von_mises_fem_nodal"]

    umag_clim = (0.0, max(np.max(umag_pred), np.max(umag_fem), 1e-12))
    vm_clim = (0.0, max(np.max(vm_pred), np.max(vm_fem), 1e-12))

    max_disp = max(np.max(umag_pred), np.max(umag_fem), 1e-12)
    deform_scale = 0.12 * max(Lx, Ly, Lz) / max_disp

    grid_pred_def = make_deformed_grid(grid, "u_pred", deform_scale)
    surface_pred = grid_pred_def.extract_surface().triangulate().compute_normals(inplace=False)

    plotter = create_paper_plotter(shape=(2, 2), off_screen=True, window_size=window_size)

    scalar_bar_args = dict(
        vertical=True,
        title_font_size=30,
        label_font_size=24,
        shadow=False,
        fmt="%.2e",
        font_family=PV_FONT_FAMILY,
        color="black",
        position_x=0.80,
        position_y=0.15,
        width=0.14,
        height=0.68,
        n_labels=4,
        italic=False,
        bold=False,
    )

    plotter.subplot(0, 0)
    plotter.add_mesh(
        surface_pred,
        scalars="umag_pred",
        cmap=PV_THEME_CMAP,
        clim=umag_clim,
        scalar_bar_args={**scalar_bar_args, "title": "|u| pred"},
        smooth_shading=True,
        lighting=False
    )
    plotter.camera_position = PV_CAMERA_POSITION
    plotter.add_text(
        "|u| pred",
        position=(0.03, 0.92),
        font_size=PV_FONT_SIZE_FRAME,
        color="black",
        font=PV_FONT_FAMILY
    )

    plotter.subplot(0, 1)
    plotter.add_mesh(
        surface_pred,
        scalars="umag_error",
        cmap=PV_THEME_CMAP,
        scalar_bar_args={**scalar_bar_args, "title": "|u| error"},
        smooth_shading=True,
        lighting=False
    )
    plotter.camera_position = PV_CAMERA_POSITION
    plotter.add_text(
        "|u| error",
        position=(0.03, 0.92),
        font_size=PV_FONT_SIZE_FRAME,
        color="black",
        font=PV_FONT_FAMILY
    )

    plotter.subplot(1, 0)
    plotter.add_mesh(
        surface_pred,
        scalars="von_mises_pred_nodal",
        cmap=PV_THEME_CMAP,
        clim=vm_clim,
        scalar_bar_args={**scalar_bar_args, "title": "von Mises pred"},
        smooth_shading=True,
        lighting=False
    )
    plotter.camera_position = PV_CAMERA_POSITION
    plotter.add_text(
        "von Mises pred",
        position=(0.03, 0.92),
        font_size=PV_FONT_SIZE_FRAME,
        color="black",
        font=PV_FONT_FAMILY
    )

    plotter.subplot(1, 1)
    plotter.add_mesh(
        surface_pred,
        scalars="von_mises_error_nodal",
        cmap=PV_THEME_CMAP,
        scalar_bar_args={**scalar_bar_args, "title": "von Mises error"},
        smooth_shading=True,
        lighting=False
    )
    plotter.camera_position = PV_CAMERA_POSITION
    plotter.add_text(
        "von Mises error",
        position=(0.03, 0.92),
        font_size=PV_FONT_SIZE_FRAME,
        color="black",
        font=PV_FONT_FAMILY
    )

    plotter.add_text(
        f"Epoch {epoch} | loss = {logs['loss']:.3e} | "
        f"Pi = {logs['energy']:.3e} | "
        f"U_int = {logs['U_int']:.3e} | "
        f"W_ext = {logs['W_ext']:.3e}",
        position="upper_edge",
        font_size=PV_FONT_SIZE_FRAME_TOP,
        color="black",
        font=PV_FONT_FAMILY
    )

    plotter.screenshot(frame_path, return_img=False)
    plotter.close()


# ============================================================
# 15. 训练曲线
# ============================================================
def save_training_curves(output_dir, history):
    if len(history["loss"]) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(history["loss"], label="loss", linewidth=1.5)
    ax.plot(history["energy"], label=r"$\Pi$", linewidth=1.2)
    ax.plot(history["U_int"], label=r"$U_{\mathrm{int}}$", linewidth=1.2)
    ax.plot(history["W_ext"], label=r"$W_{\mathrm{ext}}$", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training history")
    ax.legend(frameon=False)
    style_axis_paper(ax, equal=False)
    fig.tight_layout()
    savefig_paper(fig, os.path.join(output_dir, "training_history.png"))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(history["ux_rel_l2"], label=r"$u_x$", linewidth=1.2)
    ax.plot(history["uy_rel_l2"], label=r"$u_y$", linewidth=1.2)
    ax.plot(history["uz_rel_l2"], label=r"$u_z$", linewidth=1.2)
    ax.plot(history["umag_rel_l2"], label=r"$|\mathbf{u}|$", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("Displacement relative errors")
    ax.legend(frameon=False)
    style_axis_paper(ax, equal=False)
    fig.tight_layout()
    savefig_paper(fig, os.path.join(output_dir, "displacement_errors.png"))


# ============================================================
# 16. 结果表格
# ============================================================
def save_summary_table(output_dir, case_name, metrics, energy_model, energy_torch_fem,
                       training_time, inference_time, fem_solve_time):
    Pi_fem = energy_torch_fem["Pi"]
    Pi_model = energy_model["Pi"]
    rel_Pi = abs(Pi_model - Pi_fem) / (abs(Pi_fem) + 1e-12)

    row = {
        "Case": case_name,
        "Rel. L2 err. (|u|)": metrics["umag_rel_l2"],
        "Rel. L2 err. (|σvm|)": metrics["vm_rel_l2"],
        "Rel(Π)": rel_Pi,
        "Max err. (|u|)": metrics["umag_max_abs"],
        "Max err. (|σvm|)": metrics["vm_max_abs"],
        "Training time": training_time,
        "Inference time": inference_time,
        "FEM solve time": fem_solve_time,
    }

    txt_path = os.path.join(output_dir, "summary_table.txt")
    csv_path = os.path.join(output_dir, "summary_table.csv")

    headers = list(row.keys())

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        f.write("\t".join([
            str(row["Case"]),
            f"{row['Rel. L2 err. (|u|)']:.6e}",
            f"{row['Rel. L2 err. (|σvm|)']:.6e}",
            f"{row['Rel(Π)']:.6e}",
            f"{row['Max err. (|u|)']:.6e}",
            f"{row['Max err. (|σvm|)']:.6e}",
            f"{row['Training time']:.6f}",
            f"{row['Inference time']:.6f}",
            f"{row['FEM solve time']:.6f}",
        ]) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)

    print("\n================= SUMMARY TABLE =================")
    print("\t".join(headers))
    print("\t".join([
        str(row["Case"]),
        f"{row['Rel. L2 err. (|u|)']:.6e}",
        f"{row['Rel. L2 err. (|σvm|)']:.6e}",
        f"{row['Rel(Π)']:.6e}",
        f"{row['Max err. (|u|)']:.6e}",
        f"{row['Max err. (|σvm|)']:.6e}",
        f"{row['Training time']:.6f}",
        f"{row['Inference time']:.6f}",
        f"{row['FEM solve time']:.6f}",
    ]))
    print("=================================================\n")

    return row


# ============================================================
# 主程序
# ============================================================
def main():
    output_dir = "cantilever_beam_3d_pyvista_paper_sparse_vectorized_final"
    ensure_dir(output_dir)

    msh_path = os.path.join(output_dir, "cantilever_beam_3d.msh")
    vtu_path = os.path.join(output_dir, "results_cantilever_beam_3d_pyvista_paper_sparse_vectorized_final.vtu")
    pyvista_dir = os.path.join(output_dir, "pyvista_screenshots")
    ensure_dir(pyvista_dir)

    frames_dir = os.path.join(output_dir, "training_frames")
    ensure_dir(frames_dir)
    for fname in os.listdir(frames_dir):
        if fname.endswith(".png"):
            os.remove(os.path.join(frames_dir, fname))

    video_path = os.path.join(output_dir, "training_evolution.mp4")
    gif_path = os.path.join(output_dir, "training_evolution.gif")

    # -------------------------
    # 几何/材料/载荷参数
    # -------------------------
    Lx = 2.0
    Ly = 0.4
    Lz = 0.4

    E = 210e3
    nu = 0.3

    traction_x = 0.0
    traction_y = 0.0
    traction_z = -100.0

    lc = 0.18

    # -------------------------
    # 训练参数
    # -------------------------
    adamw_epochs = 2000
    lbfgs_epochs = 0

    adamw_lr = 1e-2
    adamw_weight_decay = 1e-6
    adamw_min_lr = 1e-7

    lbfgs_lr = 1.0
    lbfgs_max_iter = 20
    lbfgs_max_eval = 25
    lbfgs_tolerance_grad = 1e-9
    lbfgs_tolerance_change = 1e-12
    lbfgs_history_size = 100

    frame_every = 50
    video_fps = 10
    gif_fps = 10

    loss_weights = {
        "energy": 1.0,
        "weak": 0.0
    }

    case_name = "CantileverBeam3D_SparseFEM_VectorizedGNN"

    # -------------------------
    # 无量纲尺度
    # -------------------------
    Ls = max(Lx, Ly, Lz)
    sigma_s = abs(traction_x) + abs(traction_y) + abs(traction_z) + 1e-12
    eps_s = sigma_s / E
    us = Ls * eps_s
    fs = sigma_s * (Ls ** 2)

    scales = {
        "Ls": torch.tensor(Ls, dtype=torch.float32, device=device),
        "sigma_s": torch.tensor(sigma_s, dtype=torch.float32, device=device),
        "eps_s": torch.tensor(eps_s, dtype=torch.float32, device=device),
        "us": torch.tensor(us, dtype=torch.float32, device=device),
        "fs": torch.tensor(fs, dtype=torch.float32, device=device),
    }

    print("Scales:")
    print("Ls =", Ls)
    print("sigma_s =", sigma_s)
    print("eps_s =", eps_s)
    print("us =", us)
    print("fs =", fs)

    # -------------------------
    # 网格
    # -------------------------
    generate_gmsh_cantilever_beam_3d(
        msh_path=msh_path,
        Lx=Lx, Ly=Ly, Lz=Lz,
        lc=lc
    )

    nodes, elements, boundary_faces_dict = load_gmsh_mesh_3d(msh_path)
    nodes = nodes.to(device)
    elements = elements.to(device)
    boundary_faces_dict = {k: v.to(device) for k, v in boundary_faces_dict.items()}

    boundaries = boundary_nodes_from_faces(boundary_faces_dict)
    boundaries = {k: v.to(device) for k, v in boundaries.items()}

    print("num_nodes =", nodes.shape[0])
    print("num_elements =", elements.shape[0])

    print("---- 3D Cantilever beam BC ----")
    print(f"Left nodes (ux=uy=uz=0): {boundaries['left'].shape[0]}")
    print(f"Right boundary loaded: {boundary_faces_dict['right'].shape[0]} faces")

    # -------------------------
    # 图与特征
    # -------------------------
    edge_index = build_graph_from_tetra(elements).to(device)
    edge_attr = build_edge_features_3d(nodes, edge_index, Ls)
    node_feat = build_node_features_3d(
        nodes.cpu(),
        {k: v.cpu() for k, v in boundaries.items()},
        Ls
    ).to(device)

    print("num_graph_edges =", edge_index.shape[1])

    # -------------------------
    # FEM 预处理
    # -------------------------
    (
        elem_dofs,
        elem_B,
        elem_V,
        elem_Ke,
        elem_B_tilde,
        elem_V_tilde,
        elem_Ke_tilde,
        D_phys,
        D_tilde
    ) = precompute_fem_quantities_3d(nodes, elements, E, nu, Ls)

    left_nodes = boundaries["left"]
    mask_ux, mask_uy, mask_uz = build_hard_bc_masks_3d(nodes.shape[0], left_nodes)

    fixed_dofs = torch.unique(torch.cat([
        3 * left_nodes,
        3 * left_nodes + 1,
        3 * left_nodes + 2
    ]))
    fixed_vals_phys = torch.zeros(fixed_dofs.shape[0], dtype=torch.float32, device=device)
    free_dofs = build_free_dofs_3d(nodes.shape[0], fixed_dofs)

    f_ext_phys = build_external_force_from_faces(
        nodes,
        boundary_tractions=[
            (boundary_faces_dict["right"], traction_x, traction_y, traction_z),
        ]
    )

    f_ext_tilde = build_external_force_nondim_from_faces(
        nodes,
        boundary_tractions_tilde=[
            (
                boundary_faces_dict["right"],
                traction_x / sigma_s,
                traction_y / sigma_s,
                traction_z / sigma_s
            ),
        ],
        Ls=Ls
    )

    # -------------------------
    # FEM参考解（稀疏）
    # -------------------------
    with torch.no_grad():
        u_fem, K_global_sparse, fem_solve_time = solve_fem_reference_3d_sparse(
            nodes.shape[0],
            elem_dofs,
            elem_Ke,
            f_ext_phys,
            fixed_dofs,
            fixed_vals_phys
        )
        u_fem_tilde = u_fem / scales["us"]

        _, strains_tilde_fem, stresses_tilde_fem = compute_physics_nondim_3d(
            u_fem_tilde, elem_dofs, elem_B_tilde, elem_Ke_tilde, D_tilde
        )
        elem_stress_fem, von_mises_fem = compute_element_average_stress_vm_from_tilde_3d(
            strains_tilde_fem, stresses_tilde_fem, scales
        )

    print(f"FEM reference solved. FEM solve time = {fem_solve_time:.6f} s")

    # -------------------------
    # 模型
    # -------------------------
    model = PINN_MPNN_Local_3D(
        node_in_dim=node_feat.shape[1],
        edge_dim=edge_attr.shape[1],
        hidden_dim=96,
        mpnn_layers=8,
        dropout=0.0
    ).to(device)

    optimizer_adamw = torch.optim.AdamW(
        model.parameters(),
        lr=adamw_lr,
        weight_decay=adamw_weight_decay
    )

    scheduler_adamw = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adamw,
        mode='min',
        factor=0.5,
        patience=20,
        threshold=1e-5,
        threshold_mode='rel',
        cooldown=10,
        min_lr=adamw_min_lr
    )

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        max_eval=lbfgs_max_eval,
        tolerance_grad=lbfgs_tolerance_grad,
        tolerance_change=lbfgs_tolerance_change,
        history_size=lbfgs_history_size,
        line_search_fn="strong_wolfe"
    )

    history = {
        "loss": [],
        "energy": [],
        "U_int": [],
        "W_ext": [],
        "weak": [],
        "ux_rel_l2": [],
        "uy_rel_l2": [],
        "uz_rel_l2": [],
        "umag_rel_l2": [],
        "lr": [],
        "stage": [],
    }

    total_epochs = adamw_epochs + lbfgs_epochs

    def evaluate_current_model():
        with torch.no_grad():
            raw_u_tilde = model(node_feat, edge_index, edge_attr)
            u_pred_tilde = apply_hard_bc_3d(raw_u_tilde, mask_ux, mask_uy, mask_uz)

            loss, logs, strains_tilde_pred, stresses_tilde_pred = physics_hardbc_energy_loss_3d(
                u_pred_tilde=u_pred_tilde,
                f_ext_tilde=f_ext_tilde,
                elem_dofs=elem_dofs,
                elem_B_tilde=elem_B_tilde,
                elem_Ke_tilde=elem_Ke_tilde,
                D_tilde=D_tilde,
                free_dofs=free_dofs,
                weights=loss_weights
            )

            u_pred_phys = u_pred_tilde * scales["us"]
            elem_stress_pred, von_mises_pred = compute_element_average_stress_vm_from_tilde_3d(
                strains_tilde_pred, stresses_tilde_pred, scales
            )

            ux_rel = relative_l2_error(u_pred_phys[:, 0], u_fem[:, 0])
            uy_rel = relative_l2_error(u_pred_phys[:, 1], u_fem[:, 1])
            uz_rel = relative_l2_error(u_pred_phys[:, 2], u_fem[:, 2])
            umag_rel = relative_l2_error(
                torch.linalg.norm(u_pred_phys, dim=1),
                torch.linalg.norm(u_fem, dim=1)
            )

        return (
            u_pred_tilde,
            u_pred_phys,
            elem_stress_pred,
            von_mises_pred,
            loss,
            logs,
            ux_rel,
            uy_rel,
            uz_rel,
            umag_rel
        )

    # -------------------------
    # 训练计时
    # -------------------------
    train_t0 = time.perf_counter()

    print("\n================= Stage 1: AdamW pretraining =================")
    for epoch in range(1, adamw_epochs + 1):
        model.train()
        optimizer_adamw.zero_grad()

        raw_u_tilde = model(node_feat, edge_index, edge_attr)
        u_pred_tilde = apply_hard_bc_3d(raw_u_tilde, mask_ux, mask_uy, mask_uz)

        loss, logs, _, _ = physics_hardbc_energy_loss_3d(
            u_pred_tilde=u_pred_tilde,
            f_ext_tilde=f_ext_tilde,
            elem_dofs=elem_dofs,
            elem_B_tilde=elem_B_tilde,
            elem_Ke_tilde=elem_Ke_tilde,
            D_tilde=D_tilde,
            free_dofs=free_dofs,
            weights=loss_weights
        )

        if torch.isnan(loss):
            print(f"NaN encountered in AdamW stage at epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adamw.step()
        scheduler_adamw.step(logs["loss"])

        (
            u_pred_tilde_eval,
            u_pred_phys,
            elem_stress_pred,
            von_mises_pred,
            loss_eval,
            logs_eval,
            ux_rel,
            uy_rel,
            uz_rel,
            umag_rel
        ) = evaluate_current_model()

        current_lr = optimizer_adamw.param_groups[0]["lr"]
        global_epoch = epoch

        history["loss"].append(logs_eval["loss"])
        history["energy"].append(logs_eval["energy"])
        history["U_int"].append(logs_eval["U_int"])
        history["W_ext"].append(logs_eval["W_ext"])
        history["weak"].append(logs_eval["weak"])
        history["ux_rel_l2"].append(ux_rel)
        history["uy_rel_l2"].append(uy_rel)
        history["uz_rel_l2"].append(uz_rel)
        history["umag_rel_l2"].append(umag_rel)
        history["lr"].append(current_lr)
        history["stage"].append("adamw")

        if global_epoch % frame_every == 0 or global_epoch == 1 or global_epoch == total_epochs:
            frame_path = os.path.join(frames_dir, f"frame_{global_epoch:06d}.png")
            save_training_frame_pyvista(
                frame_path=frame_path,
                epoch=global_epoch,
                logs=logs_eval,
                nodes=nodes,
                elements=elements,
                u_pred_phys=u_pred_phys,
                u_fem=u_fem,
                elem_stress_pred=elem_stress_pred,
                von_mises_pred=von_mises_pred,
                elem_stress_fem=elem_stress_fem,
                von_mises_fem=von_mises_fem,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz
            )

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"[AdamW] Epoch {epoch:5d}/{adamw_epochs} | "
                f"lr={current_lr:.3e} | "
                f"loss={logs_eval['loss']:.6e} | "
                f"Pi={logs_eval['energy']:.6e} | "
                f"U_int={logs_eval['U_int']:.6e} | "
                f"W_ext={logs_eval['W_ext']:.6e} | "
                f"ux_rel={ux_rel:.6e} | "
                f"uy_rel={uy_rel:.6e} | "
                f"uz_rel={uz_rel:.6e} | "
                f"umag_rel={umag_rel:.6e}"
            )

    print("================= AdamW pretraining finished =================\n")

    print("================= Stage 2: LBFGS finetuning =================")
    for epoch in range(1, lbfgs_epochs + 1):
        model.train()
        global_epoch = adamw_epochs + epoch

        def closure():
            optimizer_lbfgs.zero_grad()

            raw_u_tilde_local = model(node_feat, edge_index, edge_attr)
            u_pred_tilde_local = apply_hard_bc_3d(raw_u_tilde_local, mask_ux, mask_uy, mask_uz)

            loss_local, _, _, _ = physics_hardbc_energy_loss_3d(
                u_pred_tilde=u_pred_tilde_local,
                f_ext_tilde=f_ext_tilde,
                elem_dofs=elem_dofs,
                elem_B_tilde=elem_B_tilde,
                elem_Ke_tilde=elem_Ke_tilde,
                D_tilde=D_tilde,
                free_dofs=free_dofs,
                weights=loss_weights
            )

            if torch.isnan(loss_local):
                raise RuntimeError(f"NaN encountered in LBFGS closure at epoch {epoch}")

            loss_local.backward()
            return loss_local

        try:
            optimizer_lbfgs.step(closure)
        except RuntimeError as e:
            print(f"LBFGS failed at epoch {epoch}: {e}")
            break

        (
            u_pred_tilde_eval,
            u_pred_phys,
            elem_stress_pred,
            von_mises_pred,
            loss_eval,
            logs_eval,
            ux_rel,
            uy_rel,
            uz_rel,
            umag_rel
        ) = evaluate_current_model()

        current_lr = optimizer_lbfgs.param_groups[0]["lr"]

        history["loss"].append(logs_eval["loss"])
        history["energy"].append(logs_eval["energy"])
        history["U_int"].append(logs_eval["U_int"])
        history["W_ext"].append(logs_eval["W_ext"])
        history["weak"].append(logs_eval["weak"])
        history["ux_rel_l2"].append(ux_rel)
        history["uy_rel_l2"].append(uy_rel)
        history["uz_rel_l2"].append(uz_rel)
        history["umag_rel_l2"].append(umag_rel)
        history["lr"].append(current_lr)
        history["stage"].append("lbfgs")

        if global_epoch % frame_every == 0 or global_epoch == 1 or global_epoch == total_epochs:
            frame_path = os.path.join(frames_dir, f"frame_{global_epoch:06d}.png")
            save_training_frame_pyvista(
                frame_path=frame_path,
                epoch=global_epoch,
                logs=logs_eval,
                nodes=nodes,
                elements=elements,
                u_pred_phys=u_pred_phys,
                u_fem=u_fem,
                elem_stress_pred=elem_stress_pred,
                von_mises_pred=von_mises_pred,
                elem_stress_fem=elem_stress_fem,
                von_mises_fem=von_mises_fem,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz
            )

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[LBFGS] Epoch {epoch:5d}/{lbfgs_epochs} | "
                f"lr={current_lr:.3e} | "
                f"loss={logs_eval['loss']:.6e} | "
                f"Pi={logs_eval['energy']:.6e} | "
                f"U_int={logs_eval['U_int']:.6e} | "
                f"W_ext={logs_eval['W_ext']:.6e} | "
                f"ux_rel={ux_rel:.6e} | "
                f"uy_rel={uy_rel:.6e} | "
                f"uz_rel={uz_rel:.6e} | "
                f"umag_rel={umag_rel:.6e}"
            )

    print("================= LBFGS finetuning finished =================\n")

    training_time = time.perf_counter() - train_t0

    # -------------------------
    # 推理计时
    # -------------------------
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_t0 = time.perf_counter()

    with torch.no_grad():
        raw_u_tilde = model(node_feat, edge_index, edge_attr)
        u_pred_tilde = apply_hard_bc_3d(raw_u_tilde, mask_ux, mask_uy, mask_uz)
        u_pred_phys = u_pred_tilde * scales["us"]

        _, strains_tilde_pred, stresses_tilde_pred = compute_physics_nondim_3d(
            u_pred_tilde, elem_dofs, elem_B_tilde, elem_Ke_tilde, D_tilde
        )
        elem_stress_pred, von_mises_pred = compute_element_average_stress_vm_from_tilde_3d(
            strains_tilde_pred, stresses_tilde_pred, scales
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - infer_t0

    metrics = evaluate_errors_3d(
        u_pred_phys, u_fem,
        elem_stress_pred, elem_stress_fem,
        von_mises_pred, von_mises_fem
    )

    print("\n================= FINAL ERROR METRICS =================")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6e}")
    print("=======================================================\n")

    with open(os.path.join(output_dir, "error_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("FINAL ERROR METRICS\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.12e}\n")

    with torch.no_grad():
        energy_torch_fem = compute_energy_from_physical_solution_3d(
            u_fem, elem_dofs, elem_Ke, f_ext_phys
        )
        energy_model = compute_energy_from_physical_solution_3d(
            u_pred_phys, elem_dofs, elem_Ke, f_ext_phys
        )

    print("\n================= ENERGY COMPARISON =================")
    print("[Torch FEM]")
    print(f"  Pi    = {energy_torch_fem['Pi']:.12e}")
    print(f"  U_int = {energy_torch_fem['U_int']:.12e}")
    print(f"  W_ext = {energy_torch_fem['W_ext']:.12e}")
    print("[Model]")
    print(f"  Pi    = {energy_model['Pi']:.12e}")
    print(f"  U_int = {energy_model['U_int']:.12e}")
    print(f"  W_ext = {energy_model['W_ext']:.12e}")
    print("=====================================================\n")

    with open(os.path.join(output_dir, "energy_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("ENERGY COMPARISON\n\n")
        f.write("[Torch FEM]\n")
        f.write(f"Pi    = {energy_torch_fem['Pi']:.12e}\n")
        f.write(f"U_int = {energy_torch_fem['U_int']:.12e}\n")
        f.write(f"W_ext = {energy_torch_fem['W_ext']:.12e}\n\n")
        f.write("[Model]\n")
        f.write(f"Pi    = {energy_model['Pi']:.12e}\n")
        f.write(f"U_int = {energy_model['U_int']:.12e}\n")
        f.write(f"W_ext = {energy_model['W_ext']:.12e}\n\n")
        f.write("[Difference: Model - Torch FEM]\n")
        f.write(f"dPi    = {energy_model['Pi'] - energy_torch_fem['Pi']:.12e}\n")
        f.write(f"dU_int = {energy_model['U_int'] - energy_torch_fem['U_int']:.12e}\n")
        f.write(f"dW_ext = {energy_model['W_ext'] - energy_torch_fem['W_ext']:.12e}\n")

    summary_row = save_summary_table(
        output_dir=output_dir,
        case_name=case_name,
        metrics=metrics,
        energy_model=energy_model,
        energy_torch_fem=energy_torch_fem,
        training_time=training_time,
        inference_time=inference_time,
        fem_solve_time=fem_solve_time
    )

    export_to_vtu_3d(
        save_path=vtu_path,
        nodes=nodes,
        elements=elements,
        u_pred_phys=u_pred_phys,
        u_fem=u_fem,
        elem_stress_pred=elem_stress_pred,
        von_mises_pred=von_mises_pred,
        elem_stress_fem=elem_stress_fem,
        von_mises_fem=von_mises_fem
    )

    save_pyvista_screenshots_3d(
        output_dir=pyvista_dir,
        nodes=nodes,
        elements=elements,
        u_pred_phys=u_pred_phys,
        u_fem=u_fem,
        elem_stress_pred=elem_stress_pred,
        von_mises_pred=von_mises_pred,
        elem_stress_fem=elem_stress_fem,
        von_mises_fem=von_mises_fem,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz
    )

    save_training_curves(output_dir, history)

    build_training_video(frames_dir, video_path, fps=video_fps)
    build_training_gif(frames_dir, gif_path, fps=gif_fps)

    save_path = os.path.join(output_dir, "gpt_elastic_cantilever_beam_3d_sparse_vectorized_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": metrics,
        "summary_row": summary_row,
        "loss_weights": loss_weights,
        "scales": {k: v.cpu() for k, v in scales.items()},
        "u_pred_tilde": u_pred_tilde.cpu(),
        "u_pred_phys": u_pred_phys.cpu(),
        "u_fem": u_fem.cpu(),
        "u_fem_tilde": u_fem_tilde.cpu(),
        "nodes": nodes.cpu(),
        "elements": elements.cpu(),
        "boundary_faces_dict": {k: v.cpu() for k, v in boundary_faces_dict.items()},
        "energy_torch_fem": energy_torch_fem,
        "energy_model": energy_model,
        "training_time": training_time,
        "inference_time": inference_time,
        "fem_solve_time": fem_solve_time,
        "adamw_epochs": adamw_epochs,
        "lbfgs_epochs": lbfgs_epochs,
        "frame_every": frame_every,
        "video_fps": video_fps,
        "gif_fps": gif_fps,
        "lc": lc,
        "SCIPY_AVAILABLE": SCIPY_AVAILABLE,
        "PV_THEME_BACKGROUND": PV_THEME_BACKGROUND,
        "PV_THEME_CMAP": PV_THEME_CMAP,
        "PV_WINDOW_SIZE": PV_WINDOW_SIZE,
        "PV_CAMERA_POSITION": PV_CAMERA_POSITION,
    }, save_path)

    print(f"Saved model/results to: {save_path}")
    print(f"All outputs, VTU and PyVista screenshots saved to {output_dir}/")
    print(f"Training video saved to: {video_path}")
    print(f"Training GIF saved to: {gif_path}")
    print(f"Training time   : {training_time:.6f} s")
    print(f"Inference time  : {inference_time:.6f} s")
    print(f"FEM solve time  : {fem_solve_time:.6f} s")
    print("Done.")


if __name__ == "__main__":
    main()
