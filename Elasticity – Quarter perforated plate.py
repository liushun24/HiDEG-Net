import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import gmsh
import meshio

import torch
import torch.nn as nn

import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================
# Optional torch_scatter
# ============================================================
try:
    from torch_scatter import scatter_add
    HAS_TORCH_SCATTER = True
except Exception:
    HAS_TORCH_SCATTER = False
    print("[Warning] torch_scatter not found. Fallback to torch.index_add_.")


# ============================================================
# Global settings
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

FIELD_CMAP = "jet"

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
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
})


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def savefig_paper(fig, save_path_png):
    fig.savefig(save_path_png, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)


def style_axis_paper(ax, equal=True):
    if equal:
        ax.set_aspect("equal")

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=0.8,
        length=3,
        top=False,
        right=False,
        bottom=True,
        left=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        width=0.6,
        length=2,
        top=False,
        right=False,
        bottom=True,
        left=True,
    )
    ax.minorticks_on()


def add_paper_colorbar(fig, mappable, ax, label=None):
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.025)
    if label is not None:
        cbar.set_label(label)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(direction="in", width=0.7, length=3)
    return cbar


# ============================================================
# 1. Gmsh mesh
# ============================================================
def generate_gmsh_plate_with_hole(
    msh_path,
    Lx=1.0,
    Ly=0.5,
    R=0.2,
    lc_min=0.02,
    lc_max=0.08,
):
    gmsh.initialize()
    gmsh.model.add("quarter_plate_with_hole")

    p_rb = gmsh.model.geo.addPoint(R, 0.0, 0.0, lc_min)
    p1 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc_max)
    p2 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc_max)
    p3 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc_max)
    p_lt = gmsh.model.geo.addPoint(0.0, R, 0.0, lc_min)
    pc = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_min)

    l_bottom = gmsh.model.geo.addLine(p_rb, p1)
    l_right = gmsh.model.geo.addLine(p1, p2)
    l_top = gmsh.model.geo.addLine(p2, p3)
    l_left = gmsh.model.geo.addLine(p3, p_lt)
    c1 = gmsh.model.geo.addCircleArc(p_lt, pc, p_rb)

    loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left, c1])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l_left], 1)
    gmsh.model.setPhysicalName(1, 1, "left")

    gmsh.model.addPhysicalGroup(1, [l_right], 2)
    gmsh.model.setPhysicalName(1, 2, "right")

    gmsh.model.addPhysicalGroup(1, [l_top], 3)
    gmsh.model.setPhysicalName(1, 3, "top")

    gmsh.model.addPhysicalGroup(1, [l_bottom], 4)
    gmsh.model.setPhysicalName(1, 4, "bottom")

    gmsh.model.addPhysicalGroup(1, [c1], 5)
    gmsh.model.setPhysicalName(1, 5, "hole")

    gmsh.model.addPhysicalGroup(2, [surf], 10)
    gmsh.model.setPhysicalName(2, 10, "domain")

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

    gmsh.model.mesh.generate(2)
    gmsh.write(msh_path)
    gmsh.finalize()


def load_gmsh_mesh(msh_path):
    mesh = meshio.read(msh_path)

    points = mesh.points[:, :2]
    nodes = torch.tensor(points, dtype=torch.float32)

    if "gmsh:physical" not in mesh.cell_data:
        raise ValueError("No 'gmsh:physical' found in mesh.cell_data.")

    physical_data = mesh.cell_data["gmsh:physical"]

    triangle_blocks = []
    line_blocks = []
    line_tags_blocks = []

    for i, cell_block in enumerate(mesh.cells):
        ctype = cell_block.type
        data = cell_block.data
        tags = np.array(physical_data[i])

        if ctype == "triangle":
            triangle_blocks.append(data)
        elif ctype == "line":
            line_blocks.append(data)
            line_tags_blocks.append(tags)

    if len(triangle_blocks) == 0:
        raise ValueError("No triangle blocks found.")
    if len(line_blocks) == 0:
        raise ValueError("No line blocks found.")

    tri_cells = np.vstack(triangle_blocks)
    line_cells = np.vstack(line_blocks)
    line_tags = np.concatenate(line_tags_blocks)

    elements = torch.tensor(tri_cells, dtype=torch.long)

    boundary_edges_dict = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": [],
        "hole": [],
    }

    for edge, tag in zip(line_cells, line_tags):
        edge = edge.tolist()
        if tag == 1:
            boundary_edges_dict["left"].append(edge)
        elif tag == 2:
            boundary_edges_dict["right"].append(edge)
        elif tag == 3:
            boundary_edges_dict["top"].append(edge)
        elif tag == 4:
            boundary_edges_dict["bottom"].append(edge)
        elif tag == 5:
            boundary_edges_dict["hole"].append(edge)

    for k in boundary_edges_dict:
        if len(boundary_edges_dict[k]) == 0:
            boundary_edges_dict[k] = torch.empty((0, 2), dtype=torch.long)
        else:
            boundary_edges_dict[k] = torch.tensor(boundary_edges_dict[k], dtype=torch.long)

    print("---- Gmsh boundary edge statistics ----")
    for k, v in boundary_edges_dict.items():
        print(f"{k:10s}: {v.shape[0]}")
    print("---------------------------------------")

    return nodes, elements, boundary_edges_dict


def boundary_nodes_from_edges(boundary_edges_dict):
    boundaries = {}
    for name, edges in boundary_edges_dict.items():
        if edges.numel() == 0:
            boundaries[name] = torch.empty(0, dtype=torch.long)
        else:
            boundaries[name] = torch.unique(edges.flatten())
    return boundaries


# ============================================================
# 2. Graph
# ============================================================
def build_graph_from_triangles(elements):
    edge_set = set()
    elems = elements.detach().cpu().numpy()

    for e in elems:
        pairs = [(e[0], e[1]), (e[1], e[2]), (e[2], e[0])]
        for i, j in pairs:
            a, b = min(int(i), int(j)), max(int(i), int(j))
            edge_set.add((a, b))

    senders, receivers = [], []
    for i, j in edge_set:
        senders.extend([i, j])
        receivers.extend([j, i])

    return torch.tensor([senders, receivers], dtype=torch.long)


def build_node_features(nodes, boundaries, Ls):
    x = nodes[:, 0:1]
    y = nodes[:, 1:2]

    x_norm = x / Ls
    y_norm = y / Ls

    def build_flag(name):
        flag = torch.zeros((nodes.shape[0], 1), dtype=torch.float32)
        if boundaries[name].numel() > 0:
            flag[boundaries[name]] = 1.0
        return flag

    left_flag = build_flag("left")
    right_flag = build_flag("right")
    top_flag = build_flag("top")
    bottom_flag = build_flag("bottom")
    hole_flag = build_flag("hole")

    return torch.cat(
        [
            x_norm,
            y_norm,
            left_flag,
            right_flag,
            top_flag,
            bottom_flag,
            hole_flag,
        ],
        dim=1,
    ).to(device)


def build_edge_features(nodes, edge_index, Ls):
    src = edge_index[0]
    dst = edge_index[1]

    dx = (nodes[dst, 0] - nodes[src, 0]).unsqueeze(1) / Ls
    dy = (nodes[dst, 1] - nodes[src, 1]).unsqueeze(1) / Ls
    dist = torch.sqrt(dx**2 + dy**2 + 1e-12)

    return torch.cat([dx, dy, dist], dim=1).to(device)


# ============================================================
# 3. FEM element matrices
# ============================================================
def constitutive_matrix_plane_stress(E, nu):
    coef = E / (1.0 - nu**2)
    return coef * torch.tensor(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def constitutive_matrix_plane_stress_nondim(nu):
    coef = 1.0 / (1.0 - nu**2)
    return coef * torch.tensor(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def t3_B_matrix_and_area(xe, ye):
    x1, x2, x3 = xe[0], xe[1], xe[2]
    y1, y2, y3 = ye[0], ye[1], ye[2]

    A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    A = torch.abs(A)

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    B = (1.0 / (2.0 * A)) * torch.tensor(
        [
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3],
        ],
        dtype=torch.float32,
        device=device,
    )

    return B, A


def t3_element_stiffness(xe, ye, D, thickness=1.0):
    B, A = t3_B_matrix_and_area(xe, ye)
    Ke = thickness * A * (B.T @ D @ B)
    return Ke, B, A


def t3_element_stiffness_nondim(xe, ye, D_tilde, Ls, thickness=1.0):
    B, A = t3_B_matrix_and_area(xe, ye)
    B_tilde = Ls * B
    A_tilde = A / (Ls**2)
    Ke_tilde = thickness * A_tilde * (B_tilde.T @ D_tilde @ B_tilde)
    return Ke_tilde, B_tilde, A_tilde


def precompute_fem_quantities(nodes, elements, E, nu, Ls, thickness=1.0):
    D = constitutive_matrix_plane_stress(E, nu)
    D_tilde = constitutive_matrix_plane_stress_nondim(nu)

    elem_dofs = []
    elem_B = []
    elem_A = []
    elem_Ke = []

    elem_B_tilde = []
    elem_A_tilde = []
    elem_Ke_tilde = []

    for e in range(elements.shape[0]):
        conn = elements[e]
        xy = nodes[conn].to(device)
        xe = xy[:, 0]
        ye = xy[:, 1]

        Ke, B, A = t3_element_stiffness(xe, ye, D, thickness)
        Ke_tilde, B_tilde, A_tilde = t3_element_stiffness_nondim(
            xe, ye, D_tilde, Ls, thickness
        )

        dofs = []
        for nid in conn.tolist():
            dofs.extend([2 * nid, 2 * nid + 1])

        elem_dofs.append(torch.tensor(dofs, dtype=torch.long, device=device))
        elem_B.append(B)
        elem_A.append(A)
        elem_Ke.append(Ke)

        elem_B_tilde.append(B_tilde)
        elem_A_tilde.append(A_tilde)
        elem_Ke_tilde.append(Ke_tilde)

    return (
        torch.stack(elem_dofs, dim=0),
        torch.stack(elem_B, dim=0),
        torch.stack(elem_A, dim=0),
        torch.stack(elem_Ke, dim=0),
        torch.stack(elem_B_tilde, dim=0),
        torch.stack(elem_A_tilde, dim=0),
        torch.stack(elem_Ke_tilde, dim=0),
        D,
        D_tilde,
    )


# ============================================================
# 4. Sparse global stiffness
# ============================================================
def assemble_global_stiffness_sparse(num_nodes, elem_dofs, elem_Ke):
    """
    Return PyTorch sparse COO global stiffness matrix.
    No dense conversion here.
    """
    ndof = num_nodes * 2

    rows = elem_dofs[:, :, None].expand(-1, 6, 6).reshape(-1)
    cols = elem_dofs[:, None, :].expand(-1, 6, 6).reshape(-1)
    vals = elem_Ke.reshape(-1)

    indices = torch.stack([rows, cols], dim=0)

    K_sparse = torch.sparse_coo_tensor(
        indices,
        vals,
        size=(ndof, ndof),
        dtype=torch.float32,
        device=elem_Ke.device,
    ).coalesce()

    return K_sparse


def sparse_matvec(K_sparse, u):
    """
    Sparse COO matvec by index_add_.
    No dense matrix.
    CUDA-safe.
    """
    K_sparse = K_sparse.coalesce()

    indices = K_sparse.indices().to(u.device)
    values = K_sparse.values().to(u.device)

    rows = indices[0]
    cols = indices[1]

    Ku = torch.zeros(
        K_sparse.shape[0],
        dtype=u.dtype,
        device=u.device,
    )

    Ku.index_add_(0, rows, values.to(u.dtype) * u[cols])

    return Ku

def extract_sparse_coo_data(K_sparse):
    """
    Extract COO rows, cols, values once.
    No dense conversion.
    """
    K_sparse = K_sparse.coalesce()
    indices = K_sparse.indices()
    values = K_sparse.values()

    rows = indices[0].contiguous()
    cols = indices[1].contiguous()
    values = values.contiguous()

    return rows, cols, values, K_sparse.shape
def sparse_matvec_coo(rows, cols, values, shape, u):
    """
    CUDA-safe sparse matrix-vector product:
        Ku = K @ u

    Uses index_add_, not torch.sparse.mm.
    No dense matrix.
    """
    rows = rows.to(u.device)
    cols = cols.to(u.device)
    values = values.to(device=u.device, dtype=u.dtype)

    Ku = torch.zeros(
        shape[0],
        dtype=u.dtype,
        device=u.device,
    )

    Ku.index_add_(0, rows, values * u[cols])

    return Ku



def torch_sparse_to_scipy_csr(K_sparse):
    """
    Convert PyTorch sparse COO to SciPy CSR.
    This does NOT create a dense matrix.
    """
    K_cpu = K_sparse.coalesce().detach().cpu()

    indices = K_cpu.indices().numpy()
    values = K_cpu.values().numpy()
    shape = tuple(K_cpu.shape)

    K_csr = sp.coo_matrix(
        (values, (indices[0], indices[1])),
        shape=shape,
    ).tocsr()

    return K_csr


def solve_fem_reference_scipy_sparse(
    num_nodes,
    K_sparse,
    f_ext,
    fixed_dofs,
    fixed_vals,
):
    """
    FEM reference solve without dense conversion.

    Solves:
        K_ff u_f = f_f - K_fc u_c

    using SciPy sparse direct solver.
    """
    ndof = num_nodes * 2

    K_csr = torch_sparse_to_scipy_csr(K_sparse)

    f_cpu = f_ext.detach().cpu().numpy().astype(np.float64)
    fixed_dofs_cpu = fixed_dofs.detach().cpu().numpy().astype(np.int64)
    fixed_vals_cpu = fixed_vals.detach().cpu().numpy().astype(np.float64)

    all_dofs = np.arange(ndof, dtype=np.int64)
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[fixed_dofs_cpu] = False
    free_dofs_cpu = all_dofs[free_mask]

    K_ff = K_csr[free_dofs_cpu][:, free_dofs_cpu].tocsc()
    K_fc = K_csr[free_dofs_cpu][:, fixed_dofs_cpu].tocsr()

    rhs = f_cpu[free_dofs_cpu] - K_fc @ fixed_vals_cpu

    u_cpu = np.zeros(ndof, dtype=np.float64)
    u_cpu[fixed_dofs_cpu] = fixed_vals_cpu
    u_cpu[free_dofs_cpu] = spla.spsolve(K_ff, rhs)

    u = torch.tensor(u_cpu, dtype=torch.float32, device=device)
    return u.reshape(num_nodes, 2)


# ============================================================
# 5. Forces
# ============================================================
def build_external_force_from_edges(nodes, boundary_tractions, thickness=1.0):
    ndof = nodes.shape[0] * 2
    f = torch.zeros(ndof, dtype=torch.float32, device=device)

    for edges, traction_x, traction_y in boundary_tractions:
        tvec = torch.tensor([traction_x, traction_y], dtype=torch.float32, device=device)

        for edge in edges.tolist():
            n1, n2 = edge
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            L = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            fe = thickness * L / 2.0 * torch.cat([tvec, tvec])
            dofs = torch.tensor(
                [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1],
                dtype=torch.long,
                device=device,
            )
            f[dofs] += fe

    return f


def build_external_force_nondim_from_edges(
    nodes,
    boundary_tractions_tilde,
    Ls=1.0,
    thickness=1.0,
):
    ndof = nodes.shape[0] * 2
    f_tilde = torch.zeros(ndof, dtype=torch.float32, device=device)

    for edges, traction_x_tilde, traction_y_tilde in boundary_tractions_tilde:
        tvec_tilde = torch.tensor(
            [traction_x_tilde, traction_y_tilde],
            dtype=torch.float32,
            device=device,
        )

        for edge in edges.tolist():
            n1, n2 = edge
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            L = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            L_tilde = L / Ls

            fe_tilde = thickness * L_tilde / 2.0 * torch.cat([tvec_tilde, tvec_tilde])
            dofs = torch.tensor(
                [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1],
                dtype=torch.long,
                device=device,
            )
            f_tilde[dofs] += fe_tilde

    return f_tilde


# ============================================================
# 6. Boundary conditions
# ============================================================
def build_hard_bc_masks(num_nodes, left_nodes, bottom_nodes):
    mask_ux = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
    mask_uy = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

    mask_ux[left_nodes, 0] = 0.0
    mask_uy[bottom_nodes, 0] = 0.0

    return mask_ux, mask_uy


def apply_hard_bc(raw_u_tilde, mask_ux, mask_uy):
    ux = raw_u_tilde[:, 0:1] * mask_ux
    uy = raw_u_tilde[:, 1:2] * mask_uy
    return torch.cat([ux, uy], dim=1)


def build_free_dofs(num_nodes, fixed_dofs):
    ndof = num_nodes * 2
    all_dofs = torch.arange(ndof, dtype=torch.long, device=device)
    mask = torch.ones(ndof, dtype=torch.bool, device=device)
    mask[fixed_dofs] = False
    return all_dofs[mask]


# ============================================================
# 7. Fast sparse physics loss
# ============================================================
def fast_sparse_energy_residual_loss(
    u_pred_tilde,
    K_rows,
    K_cols,
    K_vals,
    K_shape,
    f_ext_tilde,
    free_dofs,
    weights,
):
    u_flat = u_pred_tilde.reshape(-1)

    Ku = sparse_matvec_coo(
        K_rows,
        K_cols,
        K_vals,
        K_shape,
        u_flat,
    )

    U_int = 0.5 * torch.dot(u_flat, Ku)
    W_ext = torch.dot(u_flat, f_ext_tilde)
    Pi = U_int - W_ext

    residual = Ku - f_ext_tilde
    r_free = residual[free_dofs]
    loss_weak = torch.mean(r_free**2)

    loss = weights["energy"] * Pi + weights["weak"] * loss_weak

    logs = {
        "loss": float(loss.detach().cpu()),
        "energy": float(Pi.detach().cpu()),
        "U_int": float(U_int.detach().cpu()),
        "W_ext": float(W_ext.detach().cpu()),
        "weak": float(loss_weak.detach().cpu()),
    }

    return loss, logs



def compute_strain_stress_vectorized(
    u_pred_tilde,
    elem_dofs,
    elem_B_tilde,
    D_tilde,
):
    u_flat = u_pred_tilde.reshape(-1)
    u_elem = u_flat[elem_dofs]

    strains_tilde = torch.einsum("eij,ej->ei", elem_B_tilde, u_elem)
    stresses_tilde = torch.einsum("ij,ej->ei", D_tilde, strains_tilde)

    return strains_tilde, stresses_tilde


def compute_element_average_stress_vm_from_tilde(stresses_tilde, scales):
    sigma_s = scales["sigma_s"]
    stresses = stresses_tilde * sigma_s

    sxx = stresses[:, 0]
    syy = stresses[:, 1]
    txy = stresses[:, 2]

    von_mises = torch.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * txy**2)
    return stresses, von_mises


def compute_energy_from_sparse_coo(
    u_phys,
    K_rows,
    K_cols,
    K_vals,
    K_shape,
    f_ext_phys,
):
    u_flat = u_phys.reshape(-1)

    Ku = sparse_matvec_coo(
        K_rows,
        K_cols,
        K_vals,
        K_shape,
        u_flat,
    )

    U_int = 0.5 * torch.dot(u_flat, Ku)
    W_ext = torch.dot(u_flat, f_ext_phys)
    Pi = U_int - W_ext

    return {
        "Pi": float(Pi.detach().cpu()),
        "U_int": float(U_int.detach().cpu()),
        "W_ext": float(W_ext.detach().cpu()),
    }



# ============================================================
# 8. GNN model
# ============================================================
class MPNNLayerScatter(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.0):
        super().__init__()

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h, edge_index, edge_attr):
        src = edge_index[0]
        dst = edge_index[1]

        m_in = torch.cat([h[src], h[dst], edge_attr], dim=1)
        m = self.message_mlp(m_in)

        if HAS_TORCH_SCATTER:
            agg = scatter_add(m, dst, dim=0, dim_size=h.shape[0])
        else:
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)

        dh = self.update_mlp(torch.cat([h, agg], dim=1))
        dh = self.dropout(dh)

        return self.norm(h + dh)


class PINN_MPNN_Local_Fast(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_dim,
        hidden_dim=64,
        mpnn_layers=4,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.mpnn_layers = nn.ModuleList([
            MPNNLayerScatter(hidden_dim, edge_dim, dropout=dropout)
            for _ in range(mpnn_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, node_feat, edge_index, edge_attr):
        h0 = self.encoder(node_feat)
        h = h0

        for layer in self.mpnn_layers:
            h = layer(h, edge_index, edge_attr)

        h = h + h0
        return self.decoder(h)


# ============================================================
# 9. Metrics and plotting
# ============================================================
def relative_l2_error(pred, ref, eps=1e-12):
    return (
        torch.sqrt(torch.sum((pred - ref) ** 2))
        / (torch.sqrt(torch.sum(ref**2)) + eps)
    ).item()


def evaluate_errors(u_pred, u_fem, elem_stress_pred, elem_stress_fem, vm_pred, vm_fem):
    ux_pred, uy_pred = u_pred[:, 0], u_pred[:, 1]
    ux_fem, uy_fem = u_fem[:, 0], u_fem[:, 1]

    umag_pred = torch.sqrt(ux_pred**2 + uy_pred**2)
    umag_fem = torch.sqrt(ux_fem**2 + uy_fem**2)

    return {
        "ux_rel_l2": relative_l2_error(ux_pred, ux_fem),
        "uy_rel_l2": relative_l2_error(uy_pred, uy_fem),
        "umag_rel_l2": relative_l2_error(umag_pred, umag_fem),
        "sxx_rel_l2": relative_l2_error(elem_stress_pred[:, 0], elem_stress_fem[:, 0]),
        "syy_rel_l2": relative_l2_error(elem_stress_pred[:, 1], elem_stress_fem[:, 1]),
        "txy_rel_l2": relative_l2_error(elem_stress_pred[:, 2], elem_stress_fem[:, 2]),
        "vm_rel_l2": relative_l2_error(vm_pred, vm_fem),
    }


def build_triangulation(nodes, elements):
    nodes_np = nodes.detach().cpu().numpy()
    elems_np = elements.detach().cpu().numpy()
    return mtri.Triangulation(nodes_np[:, 0], nodes_np[:, 1], elems_np)


def element_to_nodal_field(elements, elem_field, num_nodes):
    device_local = elem_field.device
    nodal_sum = torch.zeros(num_nodes, dtype=elem_field.dtype, device=device_local)
    nodal_count = torch.zeros(num_nodes, dtype=elem_field.dtype, device=device_local)

    for e in range(elements.shape[0]):
        conn = elements[e]
        val = elem_field[e]
        nodal_sum[conn] += val
        nodal_count[conn] += 1.0

    return nodal_sum / (nodal_count + 1e-12)


def plot_mesh(nodes, elements, save_path, title="Mesh", figsize=(7.2, 3.6)):
    nodes_np = nodes.detach().cpu().numpy()
    elems_np = elements.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    for conn in elems_np:
        xy = nodes_np[conn]
        xy_closed = np.vstack([xy, xy[0]])
        ax.plot(xy_closed[:, 0], xy_closed[:, 1], color="k", linewidth=0.25)

    style_axis_paper(ax, equal=True)
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_boundary_edges(nodes, boundary_edges_dict, save_path):
    nodes_np = nodes.detach().cpu().numpy()

    colors = {
        "left": "red",
        "right": "blue",
        "top": "green",
        "bottom": "orange",
        "hole": "purple",
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    for name, edges in boundary_edges_dict.items():
        if edges.numel() == 0:
            continue

        first = True
        for edge in edges.detach().cpu().numpy():
            p1 = nodes_np[edge[0]]
            p2 = nodes_np[edge[1]]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=colors.get(name, "black"),
                linewidth=1.6,
                label=name if first else None,
            )
            first = False

    style_axis_paper(ax, equal=True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Boundary classification")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_smooth_nodal_field(
    nodes,
    elements,
    field,
    save_path,
    title="Nodal Field",
    cmap=FIELD_CMAP,
    figsize=(7.2, 3.6),
    vmin=None,
    vmax=None,
    cbar_label=None,
):
    triang = build_triangulation(nodes, elements)
    field_np = field.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    tpc = ax.tripcolor(
        triang,
        field_np,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    add_paper_colorbar(fig, tpc, ax, cbar_label)
    style_axis_paper(ax, equal=True)
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_smooth_element_field(
    nodes,
    elements,
    elem_field,
    save_path,
    title="Element Field",
    cmap=FIELD_CMAP,
    figsize=(7.2, 3.6),
    vmin=None,
    vmax=None,
    cbar_label=None,
):
    nodal_field = element_to_nodal_field(elements, elem_field, nodes.shape[0])
    plot_smooth_nodal_field(
        nodes=nodes,
        elements=elements,
        field=nodal_field,
        save_path=save_path,
        title=title,
        cmap=cmap,
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label,
    )


def plot_deformed_mesh(
    nodes,
    elements,
    u_field,
    save_path,
    scale=1.0,
    title="Deformed mesh",
    figsize=(7.2, 3.6),
):
    nodes_np = nodes.detach().cpu().numpy()
    elems_np = elements.detach().cpu().numpy()
    u_np = u_field.detach().cpu().numpy()

    deformed = nodes_np + scale * u_np

    fig, ax = plt.subplots(figsize=figsize)

    for conn in elems_np:
        xy = nodes_np[conn]
        xy_closed = np.vstack([xy, xy[0]])
        ax.plot(xy_closed[:, 0], xy_closed[:, 1], color="0.75", linewidth=0.2)

    for conn in elems_np:
        xy = deformed[conn]
        xy_closed = np.vstack([xy, xy[0]])
        ax.plot(xy_closed[:, 0], xy_closed[:, 1], color="crimson", linewidth=0.35)

    style_axis_paper(ax, equal=True)
    ax.set_title(title + f" (scale = {scale:.2f})")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.tight_layout()
    savefig_paper(fig, save_path)


def save_main_figures(output_dir, nodes, elements, u_field, elem_stress, von_mises, prefix, Lx, Ly):
    ux = u_field[:, 0]
    uy = u_field[:, 1]
    umag = torch.sqrt(ux**2 + uy**2)

    sxx = elem_stress[:, 0]
    syy = elem_stress[:, 1]
    txy = elem_stress[:, 2]

    max_disp = umag.max().item()
    deform_scale = 0.1 * max(Lx, Ly) / max(max_disp, 1e-12)

    plot_smooth_nodal_field(
        nodes,
        elements,
        ux,
        os.path.join(output_dir, f"{prefix}_ux.png"),
        rf"{prefix}: $u_x$",
        cbar_label=r"$u_x$",
    )
    plot_smooth_nodal_field(
        nodes,
        elements,
        uy,
        os.path.join(output_dir, f"{prefix}_uy.png"),
        rf"{prefix}: $u_y$",
        cbar_label=r"$u_y$",
    )
    plot_smooth_nodal_field(
        nodes,
        elements,
        umag,
        os.path.join(output_dir, f"{prefix}_umag.png"),
        rf"{prefix}: $|\mathbf{{u}}|$",
        cbar_label=r"$|\mathbf{u}|$",
    )
    plot_deformed_mesh(
        nodes,
        elements,
        u_field,
        os.path.join(output_dir, f"{prefix}_deformed_mesh.png"),
        deform_scale,
        f"{prefix}: deformed mesh",
    )

    plot_smooth_element_field(
        nodes,
        elements,
        sxx,
        os.path.join(output_dir, f"{prefix}_sxx.png"),
        rf"{prefix}: $\sigma_{{xx}}$",
        cbar_label=r"$\sigma_{xx}$",
    )
    plot_smooth_element_field(
        nodes,
        elements,
        syy,
        os.path.join(output_dir, f"{prefix}_syy.png"),
        rf"{prefix}: $\sigma_{{yy}}$",
        cbar_label=r"$\sigma_{yy}$",
    )
    plot_smooth_element_field(
        nodes,
        elements,
        txy,
        os.path.join(output_dir, f"{prefix}_txy.png"),
        rf"{prefix}: $\tau_{{xy}}$",
        cbar_label=r"$\tau_{xy}$",
    )
    plot_smooth_element_field(
        nodes,
        elements,
        von_mises,
        os.path.join(output_dir, f"{prefix}_von_mises.png"),
        f"{prefix}: von Mises stress",
        cbar_label=r"$\sigma_{\mathrm{vM}}$",
    )


def save_error_figures(
    output_dir,
    nodes,
    elements,
    u_pred,
    u_fem,
    elem_stress_pred,
    elem_stress_fem,
    vm_pred,
    vm_fem,
):
    ux_error = torch.abs(u_pred[:, 0] - u_fem[:, 0])
    uy_error = torch.abs(u_pred[:, 1] - u_fem[:, 1])

    umag_pred = torch.sqrt(u_pred[:, 0] ** 2 + u_pred[:, 1] ** 2)
    umag_fem = torch.sqrt(u_fem[:, 0] ** 2 + u_fem[:, 1] ** 2)
    umag_error = torch.abs(umag_pred - umag_fem)

    sxx_error = torch.abs(elem_stress_pred[:, 0] - elem_stress_fem[:, 0])
    syy_error = torch.abs(elem_stress_pred[:, 1] - elem_stress_fem[:, 1])
    txy_error = torch.abs(elem_stress_pred[:, 2] - elem_stress_fem[:, 2])
    vm_error = torch.abs(vm_pred - vm_fem)

    plot_smooth_nodal_field(
        nodes,
        elements,
        ux_error,
        os.path.join(output_dir, "ux_error.png"),
        r"Absolute error of $u_x$",
        cbar_label=r"$|u_x-u_x^{\mathrm{FEM}}|$",
    )
    plot_smooth_nodal_field(
        nodes,
        elements,
        uy_error,
        os.path.join(output_dir, "uy_error.png"),
        r"Absolute error of $u_y$",
        cbar_label=r"$|u_y-u_y^{\mathrm{FEM}}|$",
    )
    plot_smooth_nodal_field(
        nodes,
        elements,
        umag_error,
        os.path.join(output_dir, "umag_error.png"),
        r"Absolute error of $|\mathbf{u}|$",
        cbar_label=r"$||\mathbf{u}|-|\mathbf{u}|^{\mathrm{FEM}}|$",
    )

    plot_smooth_element_field(
        nodes,
        elements,
        sxx_error,
        os.path.join(output_dir, "sxx_error.png"),
        r"Absolute error of $\sigma_{xx}$",
        cbar_label=r"$|\sigma_{xx}-\sigma_{xx}^{\mathrm{FEM}}|$",
    )
    plot_smooth_element_field(
        nodes,
        elements,
        syy_error,
        os.path.join(output_dir, "syy_error.png"),
        r"Absolute error of $\sigma_{yy}$",
        cbar_label=r"$|\sigma_{yy}-\sigma_{yy}^{\mathrm{FEM}}|$",
    )
    plot_smooth_element_field(
        nodes,
        elements,
        txy_error,
        os.path.join(output_dir, "txy_error.png"),
        r"Absolute error of $\tau_{xy}$",
        cbar_label=r"$|\tau_{xy}-\tau_{xy}^{\mathrm{FEM}}|$",
    )
    plot_smooth_element_field(
        nodes,
        elements,
        vm_error,
        os.path.join(output_dir, "von_mises_error.png"),
        r"Absolute error of von Mises stress",
        cbar_label=r"$|\sigma_{\mathrm{vM}}-\sigma_{\mathrm{vM}}^{\mathrm{FEM}}|$",
    )


def save_training_curves(output_dir, history):
    if len(history["loss"]) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.plot(history["loss"], label="loss", linewidth=1.4)
    ax.plot(history["U_int"], label=r"$U_{\mathrm{int}}$", linewidth=1.2)
    ax.plot(history["W_ext"], label=r"$W_{\mathrm{ext}}$", linewidth=1.2)
    ax.plot(history["weak"], label="weak residual", linewidth=1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training history")
    ax.legend(frameon=False)
    style_axis_paper(ax, equal=False)
    fig.tight_layout()
    savefig_paper(fig, os.path.join(output_dir, "training_history.png"))


def export_to_vtu(
    save_path,
    nodes,
    elements,
    u_pred_phys,
    u_fem,
    elem_stress_pred,
    von_mises_pred,
    elem_stress_fem,
    von_mises_fem,
):
    points = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    points[:, :2] = nodes.detach().cpu().numpy()

    cells = [("triangle", elements.detach().cpu().numpy())]

    ux_pred = u_pred_phys[:, 0].detach().cpu().numpy()
    uy_pred = u_pred_phys[:, 1].detach().cpu().numpy()
    ux_fem = u_fem[:, 0].detach().cpu().numpy()
    uy_fem = u_fem[:, 1].detach().cpu().numpy()

    umag_pred = np.sqrt(ux_pred**2 + uy_pred**2)
    umag_fem = np.sqrt(ux_fem**2 + uy_fem**2)

    u_pred_3 = np.stack([ux_pred, uy_pred, np.zeros_like(ux_pred)], axis=1)
    u_fem_3 = np.stack([ux_fem, uy_fem, np.zeros_like(ux_fem)], axis=1)
    u_error = u_pred_3 - u_fem_3

    cell_sxx_pred = elem_stress_pred[:, 0].detach().cpu().numpy()
    cell_syy_pred = elem_stress_pred[:, 1].detach().cpu().numpy()
    cell_txy_pred = elem_stress_pred[:, 2].detach().cpu().numpy()

    cell_sxx_fem = elem_stress_fem[:, 0].detach().cpu().numpy()
    cell_syy_fem = elem_stress_fem[:, 1].detach().cpu().numpy()
    cell_txy_fem = elem_stress_fem[:, 2].detach().cpu().numpy()

    vm_pred_np = von_mises_pred.detach().cpu().numpy()
    vm_fem_np = von_mises_fem.detach().cpu().numpy()

    point_data = {
        "u_pred": u_pred_3,
        "u_fem": u_fem_3,
        "u_error": u_error,
        "umag_pred": umag_pred,
        "umag_fem": umag_fem,
        "umag_error": np.abs(umag_pred - umag_fem),
    }

    cell_data = {
        "sxx_pred": [cell_sxx_pred],
        "syy_pred": [cell_syy_pred],
        "txy_pred": [cell_txy_pred],
        "von_mises_pred": [vm_pred_np],
        "sxx_fem": [cell_sxx_fem],
        "syy_fem": [cell_syy_fem],
        "txy_fem": [cell_txy_fem],
        "von_mises_fem": [vm_fem_np],
        "sxx_error_abs": [np.abs(cell_sxx_pred - cell_sxx_fem)],
        "syy_error_abs": [np.abs(cell_syy_pred - cell_syy_fem)],
        "txy_error_abs": [np.abs(cell_txy_pred - cell_txy_fem)],
        "von_mises_error_abs": [np.abs(vm_pred_np - vm_fem_np)],
    }

    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
    )
    meshio.write(save_path, mesh)
    print(f"VTU exported to: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    output_dir = "quarter_plate_with_hole_sparse_no_dense"
    ensure_dir(output_dir)

    msh_path = os.path.join(output_dir, "plate_with_hole.msh")
    vtu_path = os.path.join(output_dir, "results_quarter_plate_with_hole_sparse_no_dense.vtu")

    # -------------------------
    # Geometry / material / load
    # -------------------------
    Lx = 1.0
    Ly = 0.5
    R = 0.2

    E = 210e3
    nu = 0.3
    thickness = 1.0

    traction_x = 100.0
    traction_y = 0.0

    lc_min = 0.02
    lc_max = 0.08

    # -------------------------
    # Training parameters
    # -------------------------
    adamw_epochs = 800
    adamw_lr = 1e-3
    adamw_weight_decay = 1e-6
    adamw_min_lr = 1e-7

    eval_every = 20
    print_every = 100

    use_torch_compile = True

    loss_weights = {
        "energy": 1.0,
        "weak": 0.0,
    }

    # -------------------------
    # Nondimensional scales
    # -------------------------
    Ls = max(Lx, Ly)
    sigma_s = abs(traction_x) + 1e-12
    eps_s = sigma_s / E
    us = Ls * eps_s
    fs = sigma_s * Ls * thickness

    scales = {
        "Ls": torch.tensor(Ls, dtype=torch.float32, device=device),
        "sigma_s": torch.tensor(sigma_s, dtype=torch.float32, device=device),
        "eps_s": torch.tensor(eps_s, dtype=torch.float32, device=device),
        "us": torch.tensor(us, dtype=torch.float32, device=device),
        "fs": torch.tensor(fs, dtype=torch.float32, device=device),
    }

    print("Device:", device)
    print("Scales:")
    print("Ls =", Ls)
    print("sigma_s =", sigma_s)
    print("eps_s =", eps_s)
    print("us =", us)
    print("fs =", fs)

    # -------------------------
    # Mesh
    # -------------------------
    generate_gmsh_plate_with_hole(
        msh_path=msh_path,
        Lx=Lx,
        Ly=Ly,
        R=R,
        lc_min=lc_min,
        lc_max=lc_max,
    )

    nodes, elements, boundary_edges_dict = load_gmsh_mesh(msh_path)

    nodes = nodes.to(device)
    elements = elements.to(device)
    boundary_edges_dict = {k: v.to(device) for k, v in boundary_edges_dict.items()}

    boundaries = boundary_nodes_from_edges(boundary_edges_dict)
    boundaries = {k: v.to(device) for k, v in boundaries.items()}

    print("num_nodes =", nodes.shape[0])
    print("num_elements =", elements.shape[0])

    plot_boundary_edges(
        nodes,
        boundary_edges_dict,
        os.path.join(output_dir, "boundary_edges.png"),
    )
    plot_mesh(
        nodes,
        elements,
        os.path.join(output_dir, "mesh.png"),
        "Sparse FEM triangular mesh",
    )

    # -------------------------
    # Graph features
    # -------------------------
    edge_index = build_graph_from_triangles(elements).to(device)
    edge_attr = build_edge_features(nodes, edge_index, Ls)

    node_feat = build_node_features(
        nodes.detach().cpu(),
        {k: v.detach().cpu() for k, v in boundaries.items()},
        Ls,
    ).to(device)

    print("num_graph_edges =", edge_index.shape[1])

    # -------------------------
    # FEM precompute
    # -------------------------
    (
        elem_dofs,
        elem_B,
        elem_A,
        elem_Ke,
        elem_B_tilde,
        elem_A_tilde,
        elem_Ke_tilde,
        D_phys,
        D_tilde,
    ) = precompute_fem_quantities(nodes, elements, E, nu, Ls, thickness)

    # Sparse global stiffness matrices.
    # No dense conversion.
    K_phys_sparse = assemble_global_stiffness_sparse(
        nodes.shape[0],
        elem_dofs,
        elem_Ke,
    )

    K_tilde_sparse = assemble_global_stiffness_sparse(
        nodes.shape[0],
        elem_dofs,
        elem_Ke_tilde,
    )
    K_tilde_rows, K_tilde_cols, K_tilde_vals, K_tilde_shape = extract_sparse_coo_data(
        K_tilde_sparse
    )

    K_phys_rows, K_phys_cols, K_phys_vals, K_phys_shape = extract_sparse_coo_data(
        K_phys_sparse
    )

    print("K_phys_sparse nnz =", K_phys_sparse._nnz())
    print("K_tilde_sparse nnz =", K_tilde_sparse._nnz())

    # -------------------------
    # BC
    # -------------------------
    left_nodes = boundaries["left"]
    bottom_nodes = boundaries["bottom"]

    mask_ux, mask_uy = build_hard_bc_masks(nodes.shape[0], left_nodes, bottom_nodes)

    fixed_dofs = torch.unique(torch.cat([2 * left_nodes, 2 * bottom_nodes + 1]))
    fixed_vals_phys = torch.zeros(fixed_dofs.shape[0], dtype=torch.float32, device=device)
    free_dofs = build_free_dofs(nodes.shape[0], fixed_dofs)

    # -------------------------
    # External force
    # -------------------------
    f_ext_phys = build_external_force_from_edges(
        nodes,
        boundary_tractions=[
            (boundary_edges_dict["right"], traction_x, traction_y),
        ],
        thickness=thickness,
    )

    f_ext_tilde = build_external_force_nondim_from_edges(
        nodes,
        boundary_tractions_tilde=[
            (boundary_edges_dict["right"], traction_x / sigma_s, traction_y / sigma_s),
        ],
        Ls=Ls,
        thickness=thickness,
    )

    # -------------------------
    # FEM reference solve: sparse SciPy, no dense
    # -------------------------
    with torch.no_grad():
        u_fem = solve_fem_reference_scipy_sparse(
            num_nodes=nodes.shape[0],
            K_sparse=K_phys_sparse,
            f_ext=f_ext_phys,
            fixed_dofs=fixed_dofs,
            fixed_vals=fixed_vals_phys,
        )

        u_fem_tilde = u_fem / scales["us"]

        _, stresses_tilde_fem = compute_strain_stress_vectorized(
            u_fem_tilde,
            elem_dofs,
            elem_B_tilde,
            D_tilde,
        )

        elem_stress_fem, von_mises_fem = compute_element_average_stress_vm_from_tilde(
            stresses_tilde_fem,
            scales,
        )

    print("FEM reference solved with SciPy sparse solver. No dense K was created.")

    # -------------------------
    # Model
    # -------------------------
    model = PINN_MPNN_Local_Fast(
        node_in_dim=node_feat.shape[1],
        edge_dim=edge_attr.shape[1],
        hidden_dim=64,
        mpnn_layers=4,
        dropout=0.0,
    ).to(device)

    if use_torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled.")
        except Exception as e:
            print(f"torch.compile failed, continue without compile. Reason: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=adamw_lr,
        weight_decay=adamw_weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        threshold=1e-5,
        threshold_mode="rel",
        cooldown=10,
        min_lr=adamw_min_lr,
    )

    history = {
        "loss": [],
        "energy": [],
        "U_int": [],
        "W_ext": [],
        "weak": [],
        "ux_rel_l2": [],
        "uy_rel_l2": [],
        "umag_rel_l2": [],
        "lr": [],
    }

    def evaluate_current_model():
        model.eval()
        with torch.no_grad():
            raw_u_tilde = model(node_feat, edge_index, edge_attr)
            u_pred_tilde = apply_hard_bc(raw_u_tilde, mask_ux, mask_uy)

            loss_eval, logs_eval = fast_sparse_energy_residual_loss(
                u_pred_tilde=u_pred_tilde,
                K_rows=K_tilde_rows,
                K_cols=K_tilde_cols,
                K_vals=K_tilde_vals,
                K_shape=K_tilde_shape,
                f_ext_tilde=f_ext_tilde,
                free_dofs=free_dofs,
                weights=loss_weights,
            )

            u_pred_phys = u_pred_tilde * scales["us"]

            _, stresses_tilde_pred = compute_strain_stress_vectorized(
                u_pred_tilde,
                elem_dofs,
                elem_B_tilde,
                D_tilde,
            )

            elem_stress_pred, von_mises_pred = compute_element_average_stress_vm_from_tilde(
                stresses_tilde_pred,
                scales,
            )

            ux_rel = relative_l2_error(u_pred_phys[:, 0], u_fem[:, 0])
            uy_rel = relative_l2_error(u_pred_phys[:, 1], u_fem[:, 1])
            umag_rel = relative_l2_error(
                torch.sqrt(u_pred_phys[:, 0] ** 2 + u_pred_phys[:, 1] ** 2),
                torch.sqrt(u_fem[:, 0] ** 2 + u_fem[:, 1] ** 2),
            )

        return (
            u_pred_tilde,
            u_pred_phys,
            elem_stress_pred,
            von_mises_pred,
            loss_eval,
            logs_eval,
            ux_rel,
            uy_rel,
            umag_rel,
        )

    # -------------------------
    # Training
    # -------------------------
    print("\n================= Training: sparse K @ u, no dense =================")

    for epoch in range(1, adamw_epochs + 1):
        model.train()
        optimizer.zero_grad()

        raw_u_tilde = model(node_feat, edge_index, edge_attr)
        u_pred_tilde = apply_hard_bc(raw_u_tilde, mask_ux, mask_uy)

        loss, logs = fast_sparse_energy_residual_loss(
            u_pred_tilde=u_pred_tilde,
            K_rows=K_tilde_rows,
            K_cols=K_tilde_cols,
            K_vals=K_tilde_vals,
            K_shape=K_tilde_shape,
            f_ext_tilde=f_ext_tilde,
            free_dofs=free_dofs,
            weights=loss_weights,
        )

        if torch.isnan(loss):
            print(f"NaN encountered at epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(logs["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch == 1 or epoch % eval_every == 0 or epoch == adamw_epochs:
            (
                u_pred_tilde_eval,
                u_pred_phys,
                elem_stress_pred,
                von_mises_pred,
                loss_eval,
                logs_eval,
                ux_rel,
                uy_rel,
                umag_rel,
            ) = evaluate_current_model()

            history["loss"].append(logs_eval["loss"])
            history["energy"].append(logs_eval["energy"])
            history["U_int"].append(logs_eval["U_int"])
            history["W_ext"].append(logs_eval["W_ext"])
            history["weak"].append(logs_eval["weak"])
            history["ux_rel_l2"].append(ux_rel)
            history["uy_rel_l2"].append(uy_rel)
            history["umag_rel_l2"].append(umag_rel)
            history["lr"].append(current_lr)

            if epoch == 1 or epoch % print_every == 0 or epoch == adamw_epochs:
                print(
                    f"[AdamW] Epoch {epoch:5d}/{adamw_epochs} | "
                    f"lr={current_lr:.3e} | "
                    f"loss={logs_eval['loss']:.6e} | "
                    f"U_int={logs_eval['U_int']:.6e} | "
                    f"W_ext={logs_eval['W_ext']:.6e} | "
                    f"weak={logs_eval['weak']:.6e} | "
                    f"ux_rel={ux_rel:.6e} | "
                    f"uy_rel={uy_rel:.6e} | "
                    f"umag_rel={umag_rel:.6e}"
                )

    print("================= Training finished =================\n")

    # -------------------------
    # Final prediction
    # -------------------------
    model.eval()
    with torch.no_grad():
        raw_u_tilde = model(node_feat, edge_index, edge_attr)
        u_pred_tilde = apply_hard_bc(raw_u_tilde, mask_ux, mask_uy)
        u_pred_phys = u_pred_tilde * scales["us"]

        _, stresses_tilde_pred = compute_strain_stress_vectorized(
            u_pred_tilde,
            elem_dofs,
            elem_B_tilde,
            D_tilde,
        )

        elem_stress_pred, von_mises_pred = compute_element_average_stress_vm_from_tilde(
            stresses_tilde_pred,
            scales,
        )

    metrics = evaluate_errors(
        u_pred_phys,
        u_fem,
        elem_stress_pred,
        elem_stress_fem,
        von_mises_pred,
        von_mises_fem,
    )

    print("\n================= FINAL ERROR METRICS =================")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6e}")
    print("=======================================================\n")

    with open(os.path.join(output_dir, "error_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("FINAL ERROR METRICS\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.12e}\n")

    # -------------------------
    # Energy comparison
    # -------------------------
    with torch.no_grad():
        energy_torch_fem = compute_energy_from_sparse_coo(
            u_fem,
            K_phys_rows,
            K_phys_cols,
            K_phys_vals,
            K_phys_shape,
            f_ext_phys,
        )

        energy_model = compute_energy_from_sparse_coo(
            u_pred_phys,
            K_phys_rows,
            K_phys_cols,
            K_phys_vals,
            K_phys_shape,
            f_ext_phys,
        )

    print("\n================= ENERGY COMPARISON =================")
    print("[Sparse FEM]")
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

        f.write("[Sparse FEM]\n")
        f.write(f"Pi    = {energy_torch_fem['Pi']:.12e}\n")
        f.write(f"U_int = {energy_torch_fem['U_int']:.12e}\n")
        f.write(f"W_ext = {energy_torch_fem['W_ext']:.12e}\n\n")

        f.write("[Model]\n")
        f.write(f"Pi    = {energy_model['Pi']:.12e}\n")
        f.write(f"U_int = {energy_model['U_int']:.12e}\n")
        f.write(f"W_ext = {energy_model['W_ext']:.12e}\n\n")

        f.write("[Difference: Model - Sparse FEM]\n")
        f.write(f"dPi    = {energy_model['Pi'] - energy_torch_fem['Pi']:.12e}\n")
        f.write(f"dU_int = {energy_model['U_int'] - energy_torch_fem['U_int']:.12e}\n")
        f.write(f"dW_ext = {energy_model['W_ext'] - energy_torch_fem['W_ext']:.12e}\n")

    # -------------------------
    # Save figures and VTU
    # -------------------------
    export_to_vtu(
        save_path=vtu_path,
        nodes=nodes,
        elements=elements,
        u_pred_phys=u_pred_phys,
        u_fem=u_fem,
        elem_stress_pred=elem_stress_pred,
        von_mises_pred=von_mises_pred,
        elem_stress_fem=elem_stress_fem,
        von_mises_fem=von_mises_fem,
    )

    save_main_figures(
        output_dir,
        nodes,
        elements,
        u_pred_phys,
        elem_stress_pred,
        von_mises_pred,
        "pred",
        Lx,
        Ly,
    )
    save_main_figures(
        output_dir,
        nodes,
        elements,
        u_fem,
        elem_stress_fem,
        von_mises_fem,
        "fem",
        Lx,
        Ly,
    )
    save_error_figures(
        output_dir,
        nodes,
        elements,
        u_pred_phys,
        u_fem,
        elem_stress_pred,
        elem_stress_fem,
        von_mises_pred,
        von_mises_fem,
    )
    save_training_curves(output_dir, history)

    # -------------------------
    # Save checkpoint
    # -------------------------
    save_path = os.path.join(output_dir, "gpt_elastic_quarter_plate_sparse_no_dense.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "metrics": metrics,
            "loss_weights": loss_weights,
            "scales": {k: v.detach().cpu() for k, v in scales.items()},
            "u_pred_tilde": u_pred_tilde.detach().cpu(),
            "u_pred_phys": u_pred_phys.detach().cpu(),
            "u_fem": u_fem.detach().cpu(),
            "u_fem_tilde": u_fem_tilde.detach().cpu(),
            "nodes": nodes.detach().cpu(),
            "elements": elements.detach().cpu(),
            "energy_sparse_fem": energy_torch_fem,
            "energy_model": energy_model,
            "adamw_epochs": adamw_epochs,
            "lc_min": lc_min,
            "lc_max": lc_max,
            "note": "Training uses PyTorch sparse K@u. FEM reference uses SciPy sparse spsolve. No K.to_dense() is used.",
        },
        save_path,
    )

    print(f"Saved model/results to: {save_path}")
    print(f"All figures and VTU saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
