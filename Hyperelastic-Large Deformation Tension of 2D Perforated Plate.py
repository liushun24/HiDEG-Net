import os
import time
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import imageio.v2 as imageio

import gmsh
import meshio

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed
import multiprocessing

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
from torch_scatter import scatter


# ============================================================
# 全局设置
# ============================================================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

device_fem = torch.device("cpu")
device_gnn = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float32)

FIELD_CMAP = "jet"
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# ============================================================
# 论文风格统一绘图参数
# ============================================================
FIGSIZE_FIELD = (4.2, 6.8)
FIGSIZE_WIDE = (6.4, 4.2)
FIGSIZE_SNAPSHOT = (10.8, 8.6)
FIGSIZE_HISTORY = (7.2, 12.6)
FIGSIZE_TABLE = (13.5, 1.8)

COLOR_RED = "#C44E52"
COLOR_BLUE = "#4C72B0"
COLOR_GREEN = "#55A868"
COLOR_ORANGE = "#DD8452"
COLOR_PURPLE = "#8172B2"
COLOR_BLACK = "#222222"
COLOR_GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"

TITLE_KW = dict(fontsize=12, pad=6)
LABEL_KW = dict(fontsize=12)


# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def savefig_paper(fig, save_path_png):
    fig.savefig(save_path_png, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)


def savefig_video_frame(fig, save_path_png):
    fig.savefig(save_path_png, bbox_inches=None, pad_inches=0.0, dpi=200)
    plt.close(fig)


def style_axis_paper(ax, equal=True):
    if equal:
        ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

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
        length=2.0,
        top=False,
        right=False,
        bottom=True,
        left=True
    )
    ax.minorticks_on()


def add_paper_colorbar(fig, mappable, ax, label=None):
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.03)
    if label is not None:
        cbar.set_label(label, fontsize=11)
    cbar.outline.set_linewidth(0.7)
    cbar.ax.tick_params(direction="in", width=0.7, length=3, labelsize=9)
    return cbar


def relative_l2_error(pred, ref, eps=1e-12):
    pred = pred.reshape(-1)
    ref = ref.reshape(-1)
    return float(np.sqrt(np.sum((pred - ref) ** 2)) / (np.sqrt(np.sum(ref ** 2)) + eps))


def relative_energy_error(pred_energy, ref_energy, eps=1e-12):
    return float(abs(pred_energy - ref_energy) / (abs(ref_energy) + eps))


def max_abs_error(pred, ref):
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.max(np.abs(pred - ref)))


def format_seconds(sec):
    return f"{sec:.4f} s"


# ============================================================
# Gmsh 生成网格：三圆孔
# ============================================================
def generate_gmsh_full_plate_with_3_holes(
    msh_path,
    Lx=1.0,
    Ly=2.0,
    holes=None,
    lc_plate=0.08,
    lc_hole=0.04
):
    if holes is None:
        holes = [
            {"cx": 0.28, "cy": 0.45, "R": 0.10},
            {"cx": 0.73, "cy": 0.95, "R": 0.14},
            {"cx": 0.40, "cy": 1.58, "R": 0.11},
        ]

    if len(holes) != 3:
        raise ValueError("holes must contain exactly 3 circular holes.")

    gmsh.initialize()
    gmsh.model.add("full_plate_with_3_irregular_circular_holes_hyperelastic_fem_nn_dense")

    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_plate)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc_plate)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc_plate)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc_plate)

    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_left = gmsh.model.geo.addLine(p4, p1)
    outer_loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])

    hole_loops = []
    all_hole_arcs = []

    for hole in holes:
        cx = hole["cx"]
        cy = hole["cy"]
        R = hole["R"]

        pc = gmsh.model.geo.addPoint(cx, cy, 0.0, lc_hole)
        pr = gmsh.model.geo.addPoint(cx + R, cy, 0.0, lc_hole)
        pt = gmsh.model.geo.addPoint(cx, cy + R, 0.0, lc_hole)
        pl = gmsh.model.geo.addPoint(cx - R, cy, 0.0, lc_hole)
        pb = gmsh.model.geo.addPoint(cx, cy - R, 0.0, lc_hole)

        c1 = gmsh.model.geo.addCircleArc(pr, pc, pt)
        c2 = gmsh.model.geo.addCircleArc(pt, pc, pl)
        c3 = gmsh.model.geo.addCircleArc(pl, pc, pb)
        c4 = gmsh.model.geo.addCircleArc(pb, pc, pr)

        loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
        hole_loops.append(loop)
        all_hole_arcs.extend([c1, c2, c3, c4])

    surf = gmsh.model.geo.addPlaneSurface([outer_loop] + hole_loops)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l_left], 1)
    gmsh.model.setPhysicalName(1, 1, "left")

    gmsh.model.addPhysicalGroup(1, [l_right], 2)
    gmsh.model.setPhysicalName(1, 2, "right")

    gmsh.model.addPhysicalGroup(1, [l_top], 3)
    gmsh.model.setPhysicalName(1, 3, "top")

    gmsh.model.addPhysicalGroup(1, [l_bottom], 4)
    gmsh.model.setPhysicalName(1, 4, "bottom")

    gmsh.model.addPhysicalGroup(1, all_hole_arcs, 5)
    gmsh.model.setPhysicalName(1, 5, "hole")

    gmsh.model.addPhysicalGroup(2, [surf], 10)
    gmsh.model.setPhysicalName(2, 10, "domain")

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(lc_plate, lc_hole))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(lc_plate, lc_hole))

    gmsh.model.mesh.generate(2)
    gmsh.write(msh_path)
    gmsh.finalize()


# ============================================================
# 网格读取
# ============================================================
def load_gmsh_mesh(msh_path):
    mesh = meshio.read(msh_path)

    points = mesh.points[:, :2]
    nodes_np = points.astype(np.float64)
    nodes_torch = torch.tensor(points, dtype=torch.float32)

    if "gmsh:physical" not in mesh.cell_data:
        raise ValueError("No 'gmsh:physical' found in mesh.cell_data")

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

    tri_cells = np.vstack(triangle_blocks).astype(np.int64)
    line_cells = np.vstack(line_blocks).astype(np.int64)
    line_tags = np.concatenate(line_tags_blocks).astype(np.int64)

    elements_torch = torch.tensor(tri_cells, dtype=torch.long)

    boundary_edges_dict_np = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": [],
        "hole": [],
    }

    for edge, tag in zip(line_cells, line_tags):
        edge = edge.tolist()
        if tag == 1:
            boundary_edges_dict_np["left"].append(edge)
        elif tag == 2:
            boundary_edges_dict_np["right"].append(edge)
        elif tag == 3:
            boundary_edges_dict_np["top"].append(edge)
        elif tag == 4:
            boundary_edges_dict_np["bottom"].append(edge)
        elif tag == 5:
            boundary_edges_dict_np["hole"].append(edge)

    for k in boundary_edges_dict_np:
        if len(boundary_edges_dict_np[k]) == 0:
            boundary_edges_dict_np[k] = np.empty((0, 2), dtype=np.int64)
        else:
            boundary_edges_dict_np[k] = np.array(boundary_edges_dict_np[k], dtype=np.int64)

    boundary_edges_dict_torch = {
        k: torch.tensor(v, dtype=torch.long) if v.shape[0] > 0 else torch.empty((0, 2), dtype=torch.long)
        for k, v in boundary_edges_dict_np.items()
    }

    return nodes_np, nodes_torch, tri_cells, elements_torch, boundary_edges_dict_np, boundary_edges_dict_torch


def boundary_nodes_from_edges_np(boundary_edges_dict):
    boundaries = {}
    for name, edges in boundary_edges_dict.items():
        if edges.size == 0:
            boundaries[name] = np.empty(0, dtype=np.int64)
        else:
            boundaries[name] = np.unique(edges.reshape(-1))
    return boundaries


def boundary_nodes_from_edges_torch(boundary_edges_dict):
    boundaries = {}
    for name, edges in boundary_edges_dict.items():
        if edges.numel() == 0:
            boundaries[name] = torch.empty(0, dtype=torch.long)
        else:
            boundaries[name] = torch.unique(edges.flatten())
    return boundaries


# ============================================================
# 可视化
# ============================================================
def build_triangulation(nodes, elements):
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)


def element_to_nodal_field_numpy(elements, elem_field, num_nodes):
    nodal_sum = np.zeros(num_nodes, dtype=np.float64)
    nodal_count = np.zeros(num_nodes, dtype=np.float64)

    for e in range(elements.shape[0]):
        conn = elements[e]
        nodal_sum[conn] += elem_field[e]
        nodal_count[conn] += 1.0

    return nodal_sum / (nodal_count + 1e-12)


def plot_boundary_edges(nodes, boundary_edges_dict, save_path):
    colors = {
        "left": COLOR_RED,
        "right": COLOR_BLUE,
        "top": COLOR_GREEN,
        "bottom": COLOR_ORANGE,
        "hole": COLOR_PURPLE
    }

    labels_pretty = {
        "left": "Left",
        "right": "Right",
        "top": "Top",
        "bottom": "Bottom",
        "hole": "Hole"
    }

    plot_order = ["left", "right", "top", "bottom", "hole"]

    fig, ax = plt.subplots(figsize=FIGSIZE_FIELD)

    for name in plot_order:
        edges = boundary_edges_dict[name]
        if edges.size == 0:
            continue

        first = True
        for edge in edges:
            p1 = nodes[edge[0]]
            p2 = nodes[edge[1]]

            if first:
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    color=colors[name],
                    linewidth=2.0,
                    solid_capstyle="round",
                    label=labels_pretty[name]
                )
                first = False
            else:
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    color=colors[name],
                    linewidth=2.0,
                    solid_capstyle="round"
                )

    style_axis_paper(ax, equal=True)
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$y$", **LABEL_KW)
    ax.set_title("Boundary classification", **TITLE_KW)

    leg = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        facecolor="white",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.085),
        ncol=3,
        columnspacing=1.2,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.45
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_mesh(nodes, elements, save_path, title="Mesh", figsize=FIGSIZE_FIELD):
    fig, ax = plt.subplots(figsize=figsize)
    for conn in elements:
        xy = nodes[conn]
        xy_closed = np.vstack([xy, xy[0]])
        ax.plot(xy_closed[:, 0], xy_closed[:, 1], color=COLOR_BLACK, linewidth=0.25)

    style_axis_paper(ax, equal=True)
    ax.set_title(title, **TITLE_KW)
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$y$", **LABEL_KW)
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_smooth_nodal_field_numpy(nodes, elements, field, save_path, title="Nodal field",
                                  cmap=FIELD_CMAP, figsize=FIGSIZE_FIELD, vmin=None, vmax=None,
                                  cbar_label=None):
    triang = build_triangulation(nodes, elements)

    fig, ax = plt.subplots(figsize=figsize)
    tpc = ax.tripcolor(triang, field, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)

    add_paper_colorbar(fig, tpc, ax, cbar_label)
    style_axis_paper(ax, equal=True)
    ax.set_title(title, **TITLE_KW)
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$y$", **LABEL_KW)
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_smooth_element_field_numpy(nodes, elements, elem_field, save_path, title="Element field",
                                    cmap=FIELD_CMAP, figsize=FIGSIZE_FIELD, vmin=None, vmax=None,
                                    cbar_label=None):
    nodal_field = element_to_nodal_field_numpy(elements, elem_field, nodes.shape[0])
    plot_smooth_nodal_field_numpy(
        nodes=nodes,
        elements=elements,
        field=nodal_field,
        save_path=save_path,
        title=title,
        cmap=cmap,
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label
    )


def plot_deformed_mesh_numpy(nodes, elements, u_field, save_path, title="Deformed mesh", figsize=FIGSIZE_FIELD):
    deformed = nodes + u_field

    fig, ax = plt.subplots(figsize=figsize)

    for conn in elements:
        xy = nodes[conn]
        xy_closed = np.vstack([xy, xy[0]])
        ax.plot(xy_closed[:, 0], xy_closed[:, 1], color=LIGHT_GRAY, linewidth=0.25, alpha=0.95)

    for conn in elements:
        xy = deformed[conn]
        xy_closed = np.vstack([xy, xy[0]])
        ax.plot(xy_closed[:, 0], xy_closed[:, 1], color=COLOR_RED, linewidth=0.40)

    style_axis_paper(ax, equal=True)
    ax.set_title(title, **TITLE_KW)
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$y$", **LABEL_KW)
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_smooth_nodal_field_on_deformed_numpy(
    nodes,
    elements,
    field,
    u_field,
    save_path,
    title="Field on deformed configuration",
    cmap=FIELD_CMAP,
    figsize=FIGSIZE_FIELD,
    vmin=None,
    vmax=None,
    cbar_label=None
):
    deformed_nodes = nodes + u_field
    triang = build_triangulation(deformed_nodes, elements)

    fig, ax = plt.subplots(figsize=figsize)
    tpc = ax.tripcolor(triang, field, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)

    add_paper_colorbar(fig, tpc, ax, cbar_label)
    style_axis_paper(ax, equal=True)
    ax.set_title(title, **TITLE_KW)
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$y$", **LABEL_KW)
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_smooth_element_field_on_deformed_numpy(
    nodes,
    elements,
    elem_field,
    u_field,
    save_path,
    title="Element field on deformed configuration",
    cmap=FIELD_CMAP,
    figsize=FIGSIZE_FIELD,
    vmin=None,
    vmax=None,
    cbar_label=None
):
    nodal_field = element_to_nodal_field_numpy(elements, elem_field, nodes.shape[0])

    plot_smooth_nodal_field_on_deformed_numpy(
        nodes=nodes,
        elements=elements,
        field=nodal_field,
        u_field=u_field,
        save_path=save_path,
        title=title,
        cmap=cmap,
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label
    )


def plot_boundary_displacement_check(nodes, boundaries, u, save_path, title="Boundary displacement check"):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    top_nodes = boundaries["top"]
    bottom_nodes = boundaries["bottom"]

    if len(top_nodes) > 0:
        idx = np.argsort(nodes[top_nodes, 0])
        ax.plot(
            nodes[top_nodes, 0][idx],
            u[top_nodes, 1][idx],
            '-o',
            color=COLOR_RED,
            linewidth=1.5,
            markersize=3.5,
            label=r"Top boundary $u_y$"
        )

    if len(bottom_nodes) > 0:
        idx = np.argsort(nodes[bottom_nodes, 0])
        ax.plot(
            nodes[bottom_nodes, 0][idx],
            u[bottom_nodes, 1][idx],
            '-o',
            color=COLOR_BLUE,
            linewidth=1.5,
            markersize=3.5,
            label=r"Bottom boundary $u_y$"
        )

    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$u_y$", **LABEL_KW)
    ax.set_title(title, **TITLE_KW)

    leg = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        facecolor="white",
        loc="best",
        handlelength=2.3,
        borderpad=0.45
    )
    leg.get_frame().set_linewidth(0.8)

    style_axis_paper(ax, equal=False)
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_top_boundary_uy(nodes, boundaries, u, prescribed_uy_top, save_path, title="Top boundary displacement"):
    top_nodes = boundaries["top"]
    if len(top_nodes) == 0:
        return

    idx = np.argsort(nodes[top_nodes, 0])

    x_top = nodes[top_nodes, 0][idx]
    uy_top = u[top_nodes, 1][idx]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(
        x_top, uy_top,
        '-o',
        color=COLOR_RED,
        linewidth=1.5,
        markersize=3.5,
        label=r"Computed top boundary $u_y$"
    )
    ax.axhline(
        prescribed_uy_top,
        color=COLOR_BLACK,
        linestyle='--',
        linewidth=1.2,
        label=rf"Prescribed $u_y={prescribed_uy_top}$"
    )
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$u_y$", **LABEL_KW)
    ax.set_title(title, **TITLE_KW)

    leg = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        facecolor="white",
        loc="best",
        handlelength=2.4,
        borderpad=0.45
    )
    leg.get_frame().set_linewidth(0.8)

    style_axis_paper(ax, equal=False)
    fig.tight_layout()
    savefig_paper(fig, save_path)


def plot_training_snapshot(
    nodes,
    elements,
    u_pred,
    sigma_pred,
    u_fem,
    history,
    epoch,
    save_path,
    title_prefix="Training"
):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_SNAPSHOT)

    deformed_nodes = nodes + u_pred
    triang = build_triangulation(deformed_nodes, elements)

    ax = axes[0, 0]
    tpc1 = ax.tripcolor(triang, u_pred[:, 1], shading="gouraud", cmap=FIELD_CMAP)
    add_paper_colorbar(fig, tpc1, ax, r"$u_y$")
    style_axis_paper(ax, equal=True)
    ax.set_title(rf"{title_prefix}: displacement $u_y$ (epoch {epoch})", **TITLE_KW)
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$y$", **LABEL_KW)

    ax = axes[0, 1]
    sigma_yy_nodal = element_to_nodal_field_numpy(elements, sigma_pred[:, 1], nodes.shape[0])
    tpc2 = ax.tripcolor(triang, sigma_yy_nodal, shading="gouraud", cmap=FIELD_CMAP)
    add_paper_colorbar(fig, tpc2, ax, r"$\sigma_{yy}$")
    style_axis_paper(ax, equal=True)
    ax.set_title(rf"{title_prefix}: stress $\sigma_{{yy}}$ (epoch {epoch})", **TITLE_KW)
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$y$", **LABEL_KW)

    ax = axes[1, 0]
    uy_err = np.abs(u_pred[:, 1] - u_fem[:, 1])
    tpc3 = ax.tripcolor(triang, uy_err, shading="gouraud", cmap=FIELD_CMAP)
    add_paper_colorbar(fig, tpc3, ax, r"$|u_y-u_y^{\mathrm{FEM}}|$")
    style_axis_paper(ax, equal=True)
    ax.set_title(rf"{title_prefix}: absolute error in $u_y$ (epoch {epoch})", **TITLE_KW)
    ax.set_xlabel(r"$x$", **LABEL_KW)
    ax.set_ylabel(r"$y$", **LABEL_KW)

    ax = axes[1, 1]
    ax.plot(history["loss"], label="Total loss", linewidth=1.5, color=COLOR_BLACK)
    if len(history["ux_rel_l2"]) > 0:
        ax.plot(history["ux_rel_l2"], label=r"$u_x$ rel-L2", linewidth=1.2, color=COLOR_BLUE)
    if len(history["uy_rel_l2"]) > 0:
        ax.plot(history["uy_rel_l2"], label=r"$u_y$ rel-L2", linewidth=1.2, color=COLOR_RED)
    if len(history["umag_rel_l2"]) > 0:
        ax.plot(history["umag_rel_l2"], label=r"$|\mathbf{u}|$ rel-L2", linewidth=1.2, color=COLOR_GREEN)

    ax.set_xlabel("Epoch", **LABEL_KW)
    ax.set_ylabel("Value", **LABEL_KW)
    ax.set_title("Training history", **TITLE_KW)

    leg = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        facecolor="white",
        loc="best",
        handlelength=2.2,
        borderpad=0.45
    )
    leg.get_frame().set_linewidth(0.8)

    style_axis_paper(ax, equal=False)

    fig.tight_layout()
    savefig_video_frame(fig, save_path)


def make_training_video(frame_dir, video_path, fps=6, macro_block_size=16):
    frame_files = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".png")
    ])

    if len(frame_files) == 0:
        print("No training frames found, skip video generation.")
        return

    first_img = imageio.imread(frame_files[0])
    target_h, target_w = first_img.shape[0], first_img.shape[1]

    if macro_block_size is not None and macro_block_size > 1:
        target_h = int(np.ceil(target_h / macro_block_size) * macro_block_size)
        target_w = int(np.ceil(target_w / macro_block_size) * macro_block_size)

    def pad_or_crop_to_target(img, target_h, target_w):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        h, w = img.shape[:2]
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

        crop_h = min(h, target_h)
        crop_w = min(w, target_w)
        canvas[:crop_h, :crop_w] = img[:crop_h, :crop_w]
        return canvas

    with imageio.get_writer(video_path, fps=fps, macro_block_size=macro_block_size) as writer:
        for f in frame_files:
            img = imageio.imread(f)
            img_fixed = pad_or_crop_to_target(img, target_h, target_w)
            writer.append_data(img_fixed)

    print(f"Training video saved to: {video_path}")
    print(f"Video frame size = ({target_h}, {target_w})")


def save_summary_table(case_rows, save_csv, save_txt, save_png):
    headers = [
        "Case",
        "Rel. L2 err. (|u|)",
        "Rel. L2 err. (|σvm|)",
        "Rel(Π)",
        "Max err. (|u|)",
        "Max err. (|σvm|)",
        "Training time",
        "Inference time",
        "FEM solve time",
    ]

    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in case_rows:
            writer.writerow(row)

    with open(save_txt, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in case_rows:
            f.write("\t".join(map(str, row)) + "\n")

    fig, ax = plt.subplots(figsize=FIGSIZE_TABLE)
    ax.axis("off")
    table = ax.table(
        cellText=case_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    savefig_paper(fig, save_png)


# ============================================================
# PyG 图构建与特征
# ============================================================
def build_graph_from_triangles_pyg(elements_torch_cpu):
    tri = elements_torch_cpu
    e01 = tri[:, [0, 1]]
    e12 = tri[:, [1, 2]]
    e20 = tri[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)

    edges_sorted, _ = torch.sort(edges, dim=1)
    edges_unique = torch.unique(edges_sorted, dim=0)

    edge_index = edges_unique.t().contiguous()
    edge_index = to_undirected(edge_index)
    return edge_index


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

    return torch.cat([
        x_norm, y_norm,
        left_flag, right_flag, top_flag, bottom_flag, hole_flag
    ], dim=1)


def build_edge_features(nodes, edge_index, Ls):
    src = edge_index[0]
    dst = edge_index[1]
    dx = (nodes[dst, 0] - nodes[src, 0]).unsqueeze(1) / Ls
    dy = (nodes[dst, 1] - nodes[src, 1]).unsqueeze(1) / Ls
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-12)
    return torch.cat([dx, dy, dist], dim=1)


def build_pyg_data(nodes_torch_cpu, elements_torch_cpu, boundaries_torch_cpu, Ls):
    edge_index = build_graph_from_triangles_pyg(elements_torch_cpu)
    x = build_node_features(nodes_torch_cpu, boundaries_torch_cpu, Ls)
    edge_attr = build_edge_features(nodes_torch_cpu, edge_index, Ls)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data


# ============================================================
# PyG MPNN
# ============================================================
class PyGMPNNLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, dropout=0.0, aggr="add"):
        super().__init__(aggr=aggr, node_dim=0)

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
        agg = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        dh = self.update_mlp(torch.cat([x, agg], dim=-1))
        dh = self.dropout(dh)
        return self.norm(x + dh)

    def message(self, x_i, x_j, edge_attr):
        m_in = torch.cat([x_j, x_i, edge_attr], dim=-1)
        return self.message_mlp(m_in)


class PINN_MPNN_PyG(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_dim,
        hidden_dim=64,
        mpnn_layers=8,
        dropout=0.0,
        output_scale=1e-4
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.layers = nn.ModuleList([
            PyGMPNNLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(mpnn_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)
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
def build_displacement_bc_values(num_nodes, prescribed_node_values):
    bc_ux = torch.full((num_nodes, 1), float("nan"), dtype=torch.float32, device=device_gnn)
    bc_uy = torch.full((num_nodes, 1), float("nan"), dtype=torch.float32, device=device_gnn)

    for nodes_idx, ux_val, uy_val in prescribed_node_values:
        if nodes_idx.numel() == 0:
            continue
        if ux_val is not None:
            bc_ux[nodes_idx, 0] = float(ux_val)
        if uy_val is not None:
            bc_uy[nodes_idx, 0] = float(uy_val)

    return bc_ux, bc_uy


def build_hard_bc_masks_and_values(num_nodes, bc_ux, bc_uy):
    mask_ux = torch.ones((num_nodes, 1), dtype=torch.float32, device=device_gnn)
    mask_uy = torch.ones((num_nodes, 1), dtype=torch.float32, device=device_gnn)

    val_ux = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device_gnn)
    val_uy = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device_gnn)

    ux_fixed = ~torch.isnan(bc_ux)
    uy_fixed = ~torch.isnan(bc_uy)

    mask_ux[ux_fixed] = 0.0
    mask_uy[uy_fixed] = 0.0

    val_ux[ux_fixed] = bc_ux[ux_fixed]
    val_uy[uy_fixed] = bc_uy[uy_fixed]

    return mask_ux, mask_uy, val_ux, val_uy


def apply_hard_bc(raw_u_tilde, mask_ux, mask_uy, val_ux, val_uy):
    ux = raw_u_tilde[:, 0:1] * mask_ux + val_ux
    uy = raw_u_tilde[:, 1:2] * mask_uy + val_uy
    return torch.cat([ux, uy], dim=1)


def build_fixed_dofs_and_values_from_bc(bc_ux, bc_uy):
    fixed_dofs = []
    fixed_vals = []

    num_nodes = bc_ux.shape[0]
    for i in range(num_nodes):
        if not torch.isnan(bc_ux[i, 0]):
            fixed_dofs.append(2 * i)
            fixed_vals.append(bc_ux[i, 0].item())
        if not torch.isnan(bc_uy[i, 0]):
            fixed_dofs.append(2 * i + 1)
            fixed_vals.append(bc_uy[i, 0].item())

    fixed_dofs = torch.tensor(fixed_dofs, dtype=torch.long, device=device_gnn)
    fixed_vals = torch.tensor(fixed_vals, dtype=torch.float32, device=device_gnn)
    return fixed_dofs, fixed_vals


def build_free_dofs(num_nodes, fixed_dofs):
    ndof = num_nodes * 2
    all_dofs = torch.arange(ndof, dtype=torch.long, device=device_gnn)
    mask = torch.ones(ndof, dtype=torch.bool, device=device_gnn)
    mask[fixed_dofs] = False
    return all_dofs[mask]


# ============================================================
# FEM：超弹性非线性求解（CPU）
# ============================================================
def lame_parameters(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def neo_hookean_P(F, mu, lam):
    J = np.linalg.det(F)
    J = max(J, 1e-12)

    FinvT = np.linalg.inv(F).T
    logJ = np.log(J)

    P = mu * (F - FinvT) + lam * logJ * FinvT
    return P, J


def cauchy_from_P_F(P, F):
    J = np.linalg.det(F)
    J = max(J, 1e-12)
    sigma = (1.0 / J) * P @ F.T
    return sigma, J


def _precompute_element_reference_worker(e, elements, nodes):
    conn = elements[e]
    X = nodes[conn]

    x1, y1 = X[0]
    x2, y2 = X[1]
    x3, y3 = X[2]

    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A0 = 0.5 * abs(twoA)

    beta = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
    gamma = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)

    gradN = np.stack([beta, gamma], axis=1) / (2.0 * A0)

    dofs = []
    for nid in conn.tolist():
        dofs.extend([2 * nid, 2 * nid + 1])
    dofs = np.array(dofs, dtype=np.int64)

    return conn, A0, gradN, dofs


def precompute_reference_data_parallel(nodes, elements, n_jobs=N_JOBS):
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_precompute_element_reference_worker)(e, elements, nodes)
        for e in range(elements.shape[0])
    )

    conn_all = np.stack([r[0] for r in results], axis=0)
    A0_all = np.array([r[1] for r in results], dtype=np.float64)
    gradN_all = np.stack([r[2] for r in results], axis=0)
    dofs_all = np.stack([r[3] for r in results], axis=0)

    return conn_all, A0_all, gradN_all, dofs_all


def element_internal_force(ue, gradN, A0, thickness, mu, lam):
    u_nodes = ue.reshape(3, 2)
    grad_u = u_nodes.T @ gradN
    F = np.eye(2) + grad_u

    P, J = neo_hookean_P(F, mu, lam)

    fe = np.zeros(6, dtype=np.float64)
    for a in range(3):
        gradNa = gradN[a]
        fa = thickness * A0 * (P @ gradNa)
        fe[2 * a:2 * a + 2] = fa

    return fe, F, J


def element_tangent_stiffness_numeric(ue, gradN, A0, thickness, mu, lam, fd_eps=1e-7):
    ndofe = 6
    Ke = np.zeros((ndofe, ndofe), dtype=np.float64)

    f0, F0, J0 = element_internal_force(ue, gradN, A0, thickness, mu, lam)

    for j in range(ndofe):
        du = np.zeros(ndofe, dtype=np.float64)
        du[j] = fd_eps

        fp, _, _ = element_internal_force(ue + du, gradN, A0, thickness, mu, lam)
        fm, _, _ = element_internal_force(ue - du, gradN, A0, thickness, mu, lam)

        Ke[:, j] = (fp - fm) / (2.0 * fd_eps)

    Ke = 0.5 * (Ke + Ke.T)
    return Ke, f0, F0, J0


def _assemble_elements_chunk(elements_idx, dofs_all, gradN_all, A0_all,
                             u_global_flat, thickness, mu, lam, fd_eps):
    rows = []
    cols = []
    vals = []
    fint = []
    fint_idx = []

    sigma_chunk = []
    vm_chunk = []
    J_chunk = []

    for e in elements_idx:
        dofs = dofs_all[e]
        ue = u_global_flat[dofs]
        gradN = gradN_all[e]
        A0 = A0_all[e]

        Ke, fe, F, J = element_tangent_stiffness_numeric(
            ue, gradN, A0, thickness, mu, lam, fd_eps=fd_eps
        )

        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(rr.ravel())
        cols.append(cc.ravel())
        vals.append(Ke.ravel())

        fint_idx.append(dofs)
        fint.append(fe)

        P, _ = neo_hookean_P(F, mu, lam)
        sigma, J_sigma = cauchy_from_P_F(P, F)
        sxx = sigma[0, 0]
        syy = sigma[1, 1]
        txy = sigma[0, 1]
        vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * txy ** 2)

        sigma_chunk.append([sxx, syy, txy])
        vm_chunk.append(vm)
        J_chunk.append(J_sigma)

    rows = np.concatenate(rows) if len(rows) > 0 else np.empty(0, dtype=np.int64)
    cols = np.concatenate(cols) if len(cols) > 0 else np.empty(0, dtype=np.int64)
    vals = np.concatenate(vals) if len(vals) > 0 else np.empty(0, dtype=np.float64)

    fint_idx = np.concatenate(fint_idx) if len(fint_idx) > 0 else np.empty(0, dtype=np.int64)
    fint = np.concatenate(fint) if len(fint) > 0 else np.empty(0, dtype=np.float64)

    sigma_chunk = np.array(sigma_chunk, dtype=np.float64)
    vm_chunk = np.array(vm_chunk, dtype=np.float64)
    J_chunk = np.array(J_chunk, dtype=np.float64)

    return rows, cols, vals, fint_idx, fint, sigma_chunk, vm_chunk, J_chunk


def assemble_global_system_parallel(
    nodes,
    elements,
    dofs_all,
    gradN_all,
    A0_all,
    u,
    thickness,
    mu,
    lam,
    fd_eps=1e-7,
    n_jobs=N_JOBS
):
    ndof = nodes.shape[0] * 2
    ne = elements.shape[0]

    u_flat = u.reshape(-1)

    chunk_size = int(np.ceil(ne / n_jobs))
    element_chunks = [np.arange(i, min(i + chunk_size, ne), dtype=np.int64)
                      for i in range(0, ne, chunk_size)]

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_assemble_elements_chunk)(
            chunk, dofs_all, gradN_all, A0_all,
            u_flat, thickness, mu, lam, fd_eps
        )
        for chunk in element_chunks
    )

    rows = np.concatenate([r[0] for r in results])
    cols = np.concatenate([r[1] for r in results])
    vals = np.concatenate([r[2] for r in results])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()

    f_int = np.zeros(ndof, dtype=np.float64)
    for r in results:
        idx = r[3]
        val = r[4]
        np.add.at(f_int, idx, val)

    sigma_all = np.vstack([r[5] for r in results])
    vm_all = np.concatenate([r[6] for r in results])
    J_all = np.concatenate([r[7] for r in results])

    return K, f_int, sigma_all, vm_all, J_all


def build_dirichlet_bcs_numpy(nodes, boundaries, prescribed_uy_top):
    num_nodes = nodes.shape[0]

    bc_ux = np.full(num_nodes, np.nan, dtype=np.float64)
    bc_uy = np.full(num_nodes, np.nan, dtype=np.float64)

    bc_ux[boundaries["bottom"]] = 0.0
    bc_uy[boundaries["bottom"]] = 0.0
    bc_uy[boundaries["top"]] = prescribed_uy_top

    return bc_ux, bc_uy


def build_fixed_dofs_and_values_from_bc_numpy(bc_ux, bc_uy):
    fixed_dofs = []
    fixed_vals = []
    num_nodes = bc_ux.shape[0]

    for i in range(num_nodes):
        if not np.isnan(bc_ux[i]):
            fixed_dofs.append(2 * i)
            fixed_vals.append(bc_ux[i])
        if not np.isnan(bc_uy[i]):
            fixed_dofs.append(2 * i + 1)
            fixed_vals.append(bc_uy[i])

    fixed_dofs = np.array(fixed_dofs, dtype=np.int64)
    fixed_vals = np.array(fixed_vals, dtype=np.float64)
    return fixed_dofs, fixed_vals


def build_free_dofs_numpy(num_nodes, fixed_dofs):
    ndof = num_nodes * 2
    all_dofs = np.arange(ndof, dtype=np.int64)
    mask = np.ones(ndof, dtype=bool)
    mask[fixed_dofs] = False
    return all_dofs[mask]


def solve_hyperelastic_newton_parallel(
    nodes,
    elements,
    dofs_all,
    gradN_all,
    A0_all,
    thickness,
    mu,
    lam,
    fixed_dofs,
    fixed_vals,
    n_steps=20,
    newton_max_iter=25,
    newton_tol_res=1e-6,
    newton_tol_du=1e-8,
    fd_eps=1e-7,
    n_jobs=N_JOBS,
    verbose=True
):
    ndof = nodes.shape[0] * 2
    u = np.zeros(ndof, dtype=np.float64)

    free_dofs = build_free_dofs_numpy(nodes.shape[0], fixed_dofs)
    step_logs = []

    for step in range(1, n_steps + 1):
        load_factor = step / n_steps
        u[fixed_dofs] = load_factor * fixed_vals

        if verbose:
            print(f"\n================ Load step {step}/{n_steps} | factor = {load_factor:.6f} ================")

        converged = False

        for it in range(1, newton_max_iter + 1):
            K, f_int, sigma_all, vm_all, J_all = assemble_global_system_parallel(
                nodes=nodes,
                elements=elements,
                dofs_all=dofs_all,
                gradN_all=gradN_all,
                A0_all=A0_all,
                u=u.reshape(-1, 2),
                thickness=thickness,
                mu=mu,
                lam=lam,
                fd_eps=fd_eps,
                n_jobs=n_jobs
            )

            R = f_int.copy()
            Rf = R[free_dofs]
            res_norm = np.linalg.norm(Rf)

            Kff = K[free_dofs][:, free_dofs]
            du_f = spla.spsolve(Kff, -Rf)
            du_norm = np.linalg.norm(du_f)

            u[free_dofs] += du_f
            u[fixed_dofs] = load_factor * fixed_vals

            if verbose:
                print(
                    f"[Step {step:02d}] Iter {it:02d} | "
                    f"||R_f||={res_norm:.6e} | "
                    f"||du_f||={du_norm:.6e} | "
                    f"J_min={J_all.min():.6e} | "
                    f"J_max={J_all.max():.6e}"
                )

            if np.min(J_all) <= 0.0:
                raise RuntimeError(f"Element inversion detected at step={step}, iter={it}, min(J)={np.min(J_all):.6e}")

            if res_norm < newton_tol_res and du_norm < newton_tol_du:
                converged = True
                break

        if not converged:
            raise RuntimeError(f"Newton did not converge at load step {step}")

        step_logs.append({
            "step": step,
            "load_factor": load_factor,
            "res_norm": res_norm,
            "du_norm": du_norm,
            "J_min": float(np.min(J_all)),
            "J_max": float(np.max(J_all)),
        })

    K, f_int, sigma_all, vm_all, J_all = assemble_global_system_parallel(
        nodes=nodes,
        elements=elements,
        dofs_all=dofs_all,
        gradN_all=gradN_all,
        A0_all=A0_all,
        u=u.reshape(-1, 2),
        thickness=thickness,
        mu=mu,
        lam=lam,
        fd_eps=fd_eps,
        n_jobs=n_jobs
    )

    return {
        "u": u.reshape(nodes.shape[0], 2),
        "K": K,
        "f_int": f_int,
        "sigma": sigma_all,
        "von_mises": vm_all,
        "J": J_all,
        "step_logs": step_logs,
        "fixed_dofs": fixed_dofs,
        "fixed_vals": fixed_vals,
        "free_dofs": free_dofs,
    }


def summarize_boundary_reactions_numpy(f_int, boundary_nodes_dict):
    out = {}
    for name, nodes_idx in boundary_nodes_dict.items():
        if nodes_idx.size == 0:
            out[f"{name}_Rx"] = 0.0
            out[f"{name}_Ry"] = 0.0
            continue
        dofs_x = 2 * nodes_idx
        dofs_y = 2 * nodes_idx + 1
        out[f"{name}_Rx"] = np.sum(f_int[dofs_x])
        out[f"{name}_Ry"] = np.sum(f_int[dofs_y])
    return out


# ============================================================
# 神经网络超弹性物理（GPU，向量化）
# ============================================================
def hyperelastic_nondim_material_params(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    mu_tilde = mu / E
    lam_tilde = lam / E
    return mu, lam, mu_tilde, lam_tilde


def precompute_reference_triangle_data_torch(nodes, elements, Ls):
    X = nodes[elements]

    x1 = X[:, 0, 0]
    y1 = X[:, 0, 1]
    x2 = X[:, 1, 0]
    y2 = X[:, 1, 1]
    x3 = X[:, 2, 0]
    y3 = X[:, 2, 1]

    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * torch.abs(twoA)

    beta = torch.stack([y2 - y3, y3 - y1, y1 - y2], dim=1)
    gamma = torch.stack([x3 - x2, x1 - x3, x2 - x1], dim=1)

    gradN = torch.stack([beta, gamma], dim=2) / (2.0 * A[:, None, None])

    elem_area = A
    elem_area_tilde = A / (Ls ** 2)

    n1 = elements[:, 0]
    n2 = elements[:, 1]
    n3 = elements[:, 2]
    elem_dofs = torch.stack([
        2 * n1, 2 * n1 + 1,
        2 * n2, 2 * n2 + 1,
        2 * n3, 2 * n3 + 1
    ], dim=1)

    return elem_area, elem_area_tilde, gradN, elem_dofs


def det2x2_batch(F):
    return F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]


def invT2x2_batch(F, eps=1e-8):
    a = F[:, 0, 0]
    b = F[:, 0, 1]
    c = F[:, 1, 0]
    d = F[:, 1, 1]

    detF = a * d - b * c
    detF = torch.clamp(detF, min=eps)

    out = torch.zeros_like(F)
    out[:, 0, 0] = d / detF
    out[:, 0, 1] = -c / detF
    out[:, 1, 0] = -b / detF
    out[:, 1, 1] = a / detF
    return out, detF


def neo_hookean_energy_density_nondim_vectorized(F, mu_tilde, lam_tilde):
    J = det2x2_batch(F)
    J = torch.clamp(J, min=1e-8)

    C = F.transpose(-1, -2) @ F
    I1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)

    logJ = torch.log(J)
    W_tilde = (
        0.5 * mu_tilde * (I1 - 2.0 - 2.0 * logJ)
        + 0.5 * lam_tilde * (logJ ** 2)
    )
    return W_tilde, J


def compute_internal_energy_nondim_hyperelastic_vectorized(
    u_pred_tilde,
    elements,
    elem_area_tilde,
    elem_gradN,
    thickness,
    mu_tilde,
    lam_tilde,
    scales
):
    us = scales["us"]

    ue_tilde = u_pred_tilde[elements]
    grad_u_phys = us * torch.einsum("eai,eaj->eij", ue_tilde, elem_gradN)
    I = torch.eye(2, dtype=torch.float32, device=device_gnn).unsqueeze(0)
    F = I + grad_u_phys

    W_tilde, J = neo_hookean_energy_density_nondim_vectorized(F, mu_tilde, lam_tilde)
    U_int_tilde = torch.sum(W_tilde * elem_area_tilde * thickness)

    return U_int_tilde


def compute_hyperelastic_physics_nondim_vectorized(
    u_pred_tilde,
    elements,
    elem_area_tilde,
    elem_gradN,
    thickness,
    mu_tilde,
    lam_tilde,
    scales
):
    u_var = u_pred_tilde.clone().requires_grad_(True)

    U_int_tilde = compute_internal_energy_nondim_hyperelastic_vectorized(
        u_pred_tilde=u_var,
        elements=elements,
        elem_area_tilde=elem_area_tilde,
        elem_gradN=elem_gradN,
        thickness=thickness,
        mu_tilde=mu_tilde,
        lam_tilde=lam_tilde,
        scales=scales
    )

    residual_tilde_xy = torch.autograd.grad(U_int_tilde, u_var, create_graph=True)[0]
    return U_int_tilde, residual_tilde_xy


def compute_element_stress_neo_hookean_from_tilde_vectorized(
    u_pred_tilde,
    elements,
    elem_gradN,
    mu_tilde,
    lam_tilde,
    scales
):
    sigma_s = scales["sigma_s"]
    us = scales["us"]

    ue_tilde = u_pred_tilde[elements]
    grad_u_phys = us * torch.einsum("eai,eaj->eij", ue_tilde, elem_gradN)
    I = torch.eye(2, dtype=torch.float32, device=device_gnn).unsqueeze(0)
    F = I + grad_u_phys

    FinvT, J = invT2x2_batch(F, eps=1e-8)
    logJ = torch.log(J)

    P_tilde = mu_tilde * (F - FinvT) + lam_tilde * logJ[:, None, None] * FinvT
    sigma_tilde = (1.0 / J)[:, None, None] * (P_tilde @ F.transpose(-1, -2))
    sigma = sigma_s * sigma_tilde

    sxx = sigma[:, 0, 0]
    syy = sigma[:, 1, 1]
    txy = sigma[:, 0, 1]
    vm = torch.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * txy ** 2)

    sigma_out = torch.stack([sxx, syy, txy], dim=1)
    return sigma_out, vm, J


def hyperelastic_pinn_supervised_loss(
    u_pred_tilde,
    u_fem_tilde,
    elements,
    elem_area_tilde,
    elem_gradN,
    free_dofs,
    thickness,
    mu_tilde,
    lam_tilde,
    scales,
    weights
):
    U_int_tilde, residual_tilde_xy = compute_hyperelastic_physics_nondim_vectorized(
        u_pred_tilde=u_pred_tilde,
        elements=elements,
        elem_area_tilde=elem_area_tilde,
        elem_gradN=elem_gradN,
        thickness=thickness,
        mu_tilde=mu_tilde,
        lam_tilde=lam_tilde,
        scales=scales
    )

    residual_tilde = residual_tilde_xy.reshape(-1)
    residual_free = residual_tilde[free_dofs]

    rx = residual_free[0::2]
    ry = residual_free[1::2]

    loss_weak_x = torch.mean(rx ** 2) if rx.numel() > 0 else torch.tensor(0.0, dtype=torch.float32, device=device_gnn)
    loss_weak_y = torch.mean(ry ** 2) if ry.numel() > 0 else torch.tensor(0.0, dtype=torch.float32, device=device_gnn)
    loss_weak = loss_weak_x + loss_weak_y

    loss_data = torch.mean((u_pred_tilde - u_fem_tilde) ** 2)

    loss = (
        weights["energy"] * U_int_tilde
        + weights["weak"] * loss_weak
        + weights["data"] * loss_data
    )

    logs = {
        "loss": float(loss.item()),
        "U_int_tilde": float(U_int_tilde.item()),
        "weak": float(loss_weak.item()),
        "weak_x": float(loss_weak_x.item()),
        "weak_y": float(loss_weak_y.item()),
        "data": float(loss_data.item()),
    }

    return loss, logs, residual_tilde_xy


# ============================================================
# 误差评价
# ============================================================
def evaluate_errors(u_pred, u_fem, sigma_pred, sigma_fem, vm_pred, vm_fem):
    ux_pred, uy_pred = u_pred[:, 0], u_pred[:, 1]
    ux_fem, uy_fem = u_fem[:, 0], u_fem[:, 1]

    umag_pred = np.sqrt(ux_pred ** 2 + uy_pred ** 2)
    umag_fem = np.sqrt(ux_fem ** 2 + uy_fem ** 2)

    sxx_pred, syy_pred, txy_pred = sigma_pred[:, 0], sigma_pred[:, 1], sigma_pred[:, 2]
    sxx_fem, syy_fem, txy_fem = sigma_fem[:, 0], sigma_fem[:, 1], sigma_fem[:, 2]

    return {
        "ux_rel_l2": relative_l2_error(ux_pred, ux_fem),
        "uy_rel_l2": relative_l2_error(uy_pred, uy_fem),
        "umag_rel_l2": relative_l2_error(umag_pred, umag_fem),
        "sxx_rel_l2": relative_l2_error(sxx_pred, sxx_fem),
        "syy_rel_l2": relative_l2_error(syy_pred, syy_fem),
        "txy_rel_l2": relative_l2_error(txy_pred, txy_fem),
        "vm_rel_l2": relative_l2_error(vm_pred, vm_fem),
        "umag_max_abs": max_abs_error(umag_pred, umag_fem),
        "vm_max_abs": max_abs_error(vm_pred, vm_fem),
    }


# ============================================================
# VTU 导出
# ============================================================
def export_to_vtu(
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
    points = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    points[:, :2] = nodes

    cells = [("triangle", elements)]

    ux_pred = u_pred[:, 0]
    uy_pred = u_pred[:, 1]
    ux_fem = u_fem[:, 0]
    uy_fem = u_fem[:, 1]

    point_data = {
        "u_pred": np.stack([ux_pred, uy_pred, np.zeros_like(ux_pred)], axis=1),
        "u_fem": np.stack([ux_fem, uy_fem, np.zeros_like(ux_fem)], axis=1),
        "u_error": np.stack([ux_pred - ux_fem, uy_pred - uy_fem, np.zeros_like(ux_pred)], axis=1),
        "umag_pred": np.sqrt(ux_pred ** 2 + uy_pred ** 2),
        "umag_fem": np.sqrt(ux_fem ** 2 + uy_fem ** 2),
        "umag_error": np.abs(np.sqrt(ux_pred ** 2 + uy_pred ** 2) - np.sqrt(ux_fem ** 2 + uy_fem ** 2)),
    }

    cell_data = {
        "sxx_pred": [sigma_pred[:, 0]],
        "syy_pred": [sigma_pred[:, 1]],
        "txy_pred": [sigma_pred[:, 2]],
        "vm_pred": [vm_pred],
        "J_pred": [J_pred],
        "sxx_fem": [sigma_fem[:, 0]],
        "syy_fem": [sigma_fem[:, 1]],
        "txy_fem": [sigma_fem[:, 2]],
        "vm_fem": [vm_fem],
        "J_fem": [J_fem],
        "sxx_error": [np.abs(sigma_pred[:, 0] - sigma_fem[:, 0])],
        "syy_error": [np.abs(sigma_pred[:, 1] - sigma_fem[:, 1])],
        "txy_error": [np.abs(sigma_pred[:, 2] - sigma_fem[:, 2])],
        "vm_error": [np.abs(vm_pred - vm_fem)],
        "J_error": [np.abs(J_pred - J_fem)],
    }

    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data)
    meshio.write(save_path, mesh)
    print(f"VTU exported to: {save_path}")


# ============================================================
# 主程序
# ============================================================
def main():
    output_dir = "hyperelastic_fem_cpu_pyg_gpu_densemesh_3holes_large_disp_demo"
    ensure_dir(output_dir)

    frames_dir = os.path.join(output_dir, "training_frames")
    ensure_dir(frames_dir)

    msh_path = os.path.join(output_dir, "full_plate_with_3_holes.msh")
    vtu_path = os.path.join(output_dir, "results_hyperelastic_fem_plus_pyg_mpnn.vtu")
    training_video_path = os.path.join(output_dir, "training_process.mp4")

    # --------------------------------------------------------
    # 几何/材料/边界参数
    # --------------------------------------------------------
    Lx = 1.0
    Ly = 2.0

    holes = [
        {"cx": 0.24, "cy": 0.38, "R": 0.10},
        {"cx": 0.72, "cy": 0.92, "R": 0.13},
        {"cx": 0.43, "cy": 1.60, "R": 0.11},
    ]

    E = 1.0e3
    nu = 0.3
    thickness = 1.0
    prescribed_uy_top = 2.0

    lc_plate = 0.08
    lc_hole = 0.035

    # --------------------------------------------------------
    # FEM Newton 参数
    # --------------------------------------------------------
    n_steps = 20
    newton_max_iter = 25
    newton_tol_res = 1e-6
    newton_tol_du = 1e-8
    fd_eps = 1e-7

    # --------------------------------------------------------
    # 神经网络训练参数
    # --------------------------------------------------------
    adamw_epochs = 800
    adamw_lr = 1e-3
    adamw_weight_decay = 1e-6
    adamw_min_lr = 1e-7

    scheduler_factor = 0.5
    scheduler_patience = 50
    scheduler_threshold = 1e-5
    scheduler_cooldown = 10

    snapshot_interval = 20

    loss_weights = {
        "energy": 1.0,
        "weak": 0.0,
        "data": 0.0
    }

    # --------------------------------------------------------
    # 无量纲尺度
    # --------------------------------------------------------
    Ls = max(Lx, Ly)
    us = abs(prescribed_uy_top) + 1e-12
    eps_s = us / Ls
    sigma_s = E
    fs = sigma_s * Ls * thickness
    energy_s = sigma_s * (Ls ** 2) * thickness

    scales = {
        "Ls": torch.tensor(Ls, dtype=torch.float32, device=device_gnn),
        "us": torch.tensor(us, dtype=torch.float32, device=device_gnn),
        "eps_s": torch.tensor(eps_s, dtype=torch.float32, device=device_gnn),
        "sigma_s": torch.tensor(sigma_s, dtype=torch.float32, device=device_gnn),
        "fs": torch.tensor(fs, dtype=torch.float32, device=device_gnn),
        "energy_s": torch.tensor(energy_s, dtype=torch.float32, device=device_gnn),
    }

    print(f"FEM device = {device_fem}")
    print(f"GNN device = {device_gnn}")
    print(f"Using N_JOBS = {N_JOBS}")

    # --------------------------------------------------------
    # 网格
    # --------------------------------------------------------
    generate_gmsh_full_plate_with_3_holes(
        msh_path=msh_path,
        Lx=Lx,
        Ly=Ly,
        holes=holes,
        lc_plate=lc_plate,
        lc_hole=lc_hole
    )

    nodes_np, nodes_torch_cpu, elements_np, elements_torch_cpu, boundary_edges_np, boundary_edges_torch_cpu = load_gmsh_mesh(msh_path)
    boundaries_np = boundary_nodes_from_edges_np(boundary_edges_np)
    boundaries_torch_cpu = boundary_nodes_from_edges_torch(boundary_edges_torch_cpu)

    print("num_nodes =", nodes_np.shape[0])
    print("num_elements =", elements_np.shape[0])

    plot_boundary_edges(nodes_np, boundary_edges_np, os.path.join(output_dir, "boundary_edges.png"))
    plot_mesh(nodes_np, elements_np, os.path.join(output_dir, "mesh.png"), "Finite element mesh")

    # --------------------------------------------------------
    # FEM 参考解（CPU）
    # --------------------------------------------------------
    mu, lam = lame_parameters(E, nu)
    print("FEM material params:")
    print("E =", E)
    print("nu =", nu)
    print("mu =", mu)
    print("lambda =", lam)
    print("prescribed_uy_top =", prescribed_uy_top)
    print("holes =", holes)

    print("Precomputing FEM reference data in parallel...")
    conn_all, A0_all, gradN_all_np, dofs_all_np = precompute_reference_data_parallel(
        nodes_np, elements_np, n_jobs=N_JOBS
    )

    bc_ux_np, bc_uy_np = build_dirichlet_bcs_numpy(nodes_np, boundaries_np, prescribed_uy_top)
    fixed_dofs_np, fixed_vals_np = build_fixed_dofs_and_values_from_bc_numpy(bc_ux_np, bc_uy_np)

    top_nodes = boundaries_np["top"]
    bottom_nodes = boundaries_np["bottom"]
    left_nodes = boundaries_np["left"]
    right_nodes = boundaries_np["right"]
    hole_nodes = boundaries_np["hole"]

    print("\n========== BC DEBUG ==========")
    print("prescribed_uy_top =", prescribed_uy_top)
    print("num top nodes =", len(top_nodes))
    print("num bottom nodes =", len(bottom_nodes))
    print("num left nodes =", len(left_nodes))
    print("num right nodes =", len(right_nodes))
    print("num hole nodes =", len(hole_nodes))

    if len(top_nodes) > 0:
        print("Top node y min/max:", nodes_np[top_nodes, 1].min(), nodes_np[top_nodes, 1].max())
        print("bc_uy on top unique =", np.unique(np.round(bc_uy_np[top_nodes], 8)))

    if len(bottom_nodes) > 0:
        print("Bottom node y min/max:", nodes_np[bottom_nodes, 1].min(), nodes_np[bottom_nodes, 1].max())
        print("bc_ux on bottom unique =", np.unique(np.round(bc_ux_np[bottom_nodes], 8)))
        print("bc_uy on bottom unique =", np.unique(np.round(bc_uy_np[bottom_nodes], 8)))

    print("================================\n")

    print("Solving nonlinear hyperelastic FEM reference on CPU...")
    t_fem0 = time.perf_counter()
    fem_result = solve_hyperelastic_newton_parallel(
        nodes=nodes_np,
        elements=elements_np,
        dofs_all=dofs_all_np,
        gradN_all=gradN_all_np,
        A0_all=A0_all,
        thickness=thickness,
        mu=mu,
        lam=lam,
        fixed_dofs=fixed_dofs_np,
        fixed_vals=fixed_vals_np,
        n_steps=n_steps,
        newton_max_iter=newton_max_iter,
        newton_tol_res=newton_tol_res,
        newton_tol_du=newton_tol_du,
        fd_eps=fd_eps,
        n_jobs=N_JOBS,
        verbose=True
    )
    fem_solve_time = time.perf_counter() - t_fem0

    u_fem = fem_result["u"]
    sigma_fem = fem_result["sigma"]
    vm_fem = fem_result["von_mises"]
    J_fem = fem_result["J"]
    f_int_fem = fem_result["f_int"]

    reaction_stats_fem = summarize_boundary_reactions_numpy(
        f_int_fem,
        {
            "top": boundaries_np["top"],
            "bottom": boundaries_np["bottom"],
            "left": boundaries_np["left"],
            "right": boundaries_np["right"],
            "hole": boundaries_np["hole"],
        }
    )

    print("\nFEM reaction summary:")
    for k, v in reaction_stats_fem.items():
        print(f"{k:15s}: {v:.6e}")

    plot_boundary_displacement_check(
        nodes_np, boundaries_np, u_fem,
        os.path.join(output_dir, "fem_boundary_displacement_check.png"),
        title="Boundary displacement check for FEM"
    )

    plot_top_boundary_uy(
        nodes_np, boundaries_np, u_fem, prescribed_uy_top,
        os.path.join(output_dir, "fem_top_boundary_uy.png"),
        title=r"Top boundary displacement for FEM"
    )

    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, u_fem[:, 0], u_fem,
        os.path.join(output_dir, "fem_ux.png"),
        title=r"FEM displacement $u_x$",
        cbar_label=r"$u_x$"
    )
    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, u_fem[:, 1], u_fem,
        os.path.join(output_dir, "fem_uy.png"),
        title=r"FEM displacement $u_y$",
        cbar_label=r"$u_y$"
    )
    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, np.sqrt(u_fem[:, 0] ** 2 + u_fem[:, 1] ** 2), u_fem,
        os.path.join(output_dir, "fem_umag.png"),
        title=r"FEM displacement magnitude $|\mathbf{u}|$",
        cbar_label=r"$|\mathbf{u}|$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, sigma_fem[:, 0], u_fem,
        os.path.join(output_dir, "fem_sxx.png"),
        title=r"FEM stress $\sigma_{xx}$",
        cbar_label=r"$\sigma_{xx}$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, sigma_fem[:, 1], u_fem,
        os.path.join(output_dir, "fem_syy.png"),
        title=r"FEM stress $\sigma_{yy}$",
        cbar_label=r"$\sigma_{yy}$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, vm_fem, u_fem,
        os.path.join(output_dir, "fem_vm.png"),
        title=r"FEM von Mises stress",
        cbar_label=r"$\sigma_{\mathrm{vM}}$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, J_fem, u_fem,
        os.path.join(output_dir, "fem_J.png"),
        title=r"FEM Jacobian $J$",
        cbar_label=r"$J$"
    )
    plot_deformed_mesh_numpy(
        nodes_np, elements_np, u_fem,
        os.path.join(output_dir, "fem_deformed_mesh.png"),
        title="FEM deformed mesh"
    )

    # --------------------------------------------------------
    # PyG图网络部分（GPU）
    # --------------------------------------------------------
    nodes_torch = nodes_torch_cpu.to(device_gnn)
    elements_torch = elements_torch_cpu.to(device_gnn)
    boundaries_torch = {k: v.to(device_gnn) for k, v in boundaries_torch_cpu.items()}

    pyg_data = build_pyg_data(nodes_torch_cpu, elements_torch_cpu, boundaries_torch_cpu, Ls)
    pyg_data = pyg_data.to(device_gnn)

    _, _, mu_tilde, lam_tilde = hyperelastic_nondim_material_params(E, nu)

    elem_area, elem_area_tilde, elem_gradN_torch, elem_dofs_torch = precompute_reference_triangle_data_torch(
        nodes_torch, elements_torch, Ls
    )

    prescribed_uy_top_tilde = prescribed_uy_top / us

    bc_ux_tilde, bc_uy_tilde = build_displacement_bc_values(
        num_nodes=nodes_torch.shape[0],
        prescribed_node_values=[
            (boundaries_torch["bottom"], 0.0, 0.0),
            (boundaries_torch["top"], None, prescribed_uy_top_tilde),
        ]
    )

    mask_ux, mask_uy, val_ux, val_uy = build_hard_bc_masks_and_values(
        nodes_torch.shape[0], bc_ux_tilde, bc_uy_tilde
    )

    fixed_dofs_torch, fixed_vals_torch = build_fixed_dofs_and_values_from_bc(
        bc_ux_tilde, bc_uy_tilde
    )
    free_dofs_torch = build_free_dofs(nodes_torch.shape[0], fixed_dofs_torch)

    u_fem_torch = torch.tensor(u_fem, dtype=torch.float32, device=device_gnn)
    u_fem_tilde = u_fem_torch / scales["us"]

    with torch.no_grad():
        fem_energy_tilde = compute_internal_energy_nondim_hyperelastic_vectorized(
            u_pred_tilde=u_fem_tilde,
            elements=elements_torch,
            elem_area_tilde=elem_area_tilde,
            elem_gradN=elem_gradN_torch,
            thickness=thickness,
            mu_tilde=mu_tilde,
            lam_tilde=lam_tilde,
            scales=scales
        ).item()

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------
    model = PINN_MPNN_PyG(
        node_in_dim=pyg_data.x.shape[1],
        edge_dim=pyg_data.edge_attr.shape[1],
        hidden_dim=64,
        mpnn_layers=8,
        dropout=0.0,
        output_scale=1e-4
    ).to(device_gnn)

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

    history = {
        "loss": [],
        "U_int_tilde": [],
        "weak": [],
        "data": [],
        "ux_rel_l2": [],
        "uy_rel_l2": [],
        "umag_rel_l2": [],
        "lr": [],
    }

    print("\n================= Training PyG-MPNN/PINN on GNN device =================")
    t_train0 = time.perf_counter()
    for epoch in range(1, adamw_epochs + 1):
        model.train()
        optimizer.zero_grad()

        raw_u_tilde = model(pyg_data)
        u_pred_tilde = apply_hard_bc(raw_u_tilde, mask_ux, mask_uy, val_ux, val_uy)

        loss, logs, _ = hyperelastic_pinn_supervised_loss(
            u_pred_tilde=u_pred_tilde,
            u_fem_tilde=u_fem_tilde,
            elements=elements_torch,
            elem_area_tilde=elem_area_tilde,
            elem_gradN=elem_gradN_torch,
            free_dofs=free_dofs_torch,
            thickness=thickness,
            mu_tilde=mu_tilde,
            lam_tilde=lam_tilde,
            scales=scales,
            weights=loss_weights
        )

        if torch.isnan(loss):
            print(f"NaN encountered at epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(logs["loss"])

        with torch.no_grad():
            u_pred_phys = (u_pred_tilde * scales["us"]).detach().cpu().numpy()

            sigma_pred_t, vm_pred_t, J_pred_t = compute_element_stress_neo_hookean_from_tilde_vectorized(
                u_pred_tilde=u_pred_tilde,
                elements=elements_torch,
                elem_gradN=elem_gradN_torch,
                mu_tilde=mu_tilde,
                lam_tilde=lam_tilde,
                scales=scales
            )

            sigma_pred = sigma_pred_t.detach().cpu().numpy()
            vm_pred = vm_pred_t.detach().cpu().numpy()

            ux_rel = relative_l2_error(u_pred_phys[:, 0], u_fem[:, 0])
            uy_rel = relative_l2_error(u_pred_phys[:, 1], u_fem[:, 1])
            umag_rel = relative_l2_error(
                np.sqrt(u_pred_phys[:, 0] ** 2 + u_pred_phys[:, 1] ** 2),
                np.sqrt(u_fem[:, 0] ** 2 + u_fem[:, 1] ** 2)
            )

        current_lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(logs["loss"])
        history["U_int_tilde"].append(logs["U_int_tilde"])
        history["weak"].append(logs["weak"])
        history["data"].append(logs["data"])
        history["ux_rel_l2"].append(ux_rel)
        history["uy_rel_l2"].append(uy_rel)
        history["umag_rel_l2"].append(umag_rel)
        history["lr"].append(current_lr)

        if (epoch % snapshot_interval == 0) or (epoch == 1) or (epoch == adamw_epochs):
            frame_path = os.path.join(frames_dir, f"frame_{epoch:05d}.png")
            plot_training_snapshot(
                nodes=nodes_np,
                elements=elements_np,
                u_pred=u_pred_phys,
                sigma_pred=sigma_pred,
                u_fem=u_fem,
                history=history,
                epoch=epoch,
                save_path=frame_path,
                title_prefix="Training"
            )

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d}/{adamw_epochs} | "
                f"lr={current_lr:.3e} | "
                f"loss={logs['loss']:.6e} | "
                f"U_int_tilde={logs['U_int_tilde']:.6e} | "
                f"weak={logs['weak']:.6e} | "
                f"data={logs['data']:.6e} | "
                f"ux_rel={ux_rel:.6e} | "
                f"uy_rel={uy_rel:.6e} | "
                f"umag_rel={umag_rel:.6e}"
            )

    if device_gnn.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.perf_counter() - t_train0
    print("================= Training finished =================\n")

    make_training_video(
        frame_dir=frames_dir,
        video_path=training_video_path,
        fps=6,
        macro_block_size=16
    )

    # --------------------------------------------------------
    # 最终预测
    # --------------------------------------------------------
    model.eval()
    t_inf0 = time.perf_counter()

    with torch.no_grad():
        raw_u_tilde = model(pyg_data)
        u_pred_tilde = apply_hard_bc(raw_u_tilde, mask_ux, mask_uy, val_ux, val_uy)
        u_pred = (u_pred_tilde * scales["us"]).detach().cpu().numpy()

        sigma_pred_t, vm_pred_t, J_pred_t = compute_element_stress_neo_hookean_from_tilde_vectorized(
            u_pred_tilde=u_pred_tilde,
            elements=elements_torch,
            elem_gradN=elem_gradN_torch,
            mu_tilde=mu_tilde,
            lam_tilde=lam_tilde,
            scales=scales
        )

        pred_energy_tilde = compute_internal_energy_nondim_hyperelastic_vectorized(
            u_pred_tilde=u_pred_tilde,
            elements=elements_torch,
            elem_area_tilde=elem_area_tilde,
            elem_gradN=elem_gradN_torch,
            thickness=thickness,
            mu_tilde=mu_tilde,
            lam_tilde=lam_tilde,
            scales=scales
        ).item()

    if device_gnn.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - t_inf0

    sigma_pred = sigma_pred_t.detach().cpu().numpy()
    vm_pred = vm_pred_t.detach().cpu().numpy()
    J_pred = J_pred_t.detach().cpu().numpy()

    print("\nBoundary displacement check (Prediction):")
    if len(bottom_nodes) > 0:
        print(f"Pred bottom ux min/max = {u_pred[bottom_nodes, 0].min():.6f}, {u_pred[bottom_nodes, 0].max():.6f}")
        print(f"Pred bottom uy min/max = {u_pred[bottom_nodes, 1].min():.6f}, {u_pred[bottom_nodes, 1].max():.6f}")
    if len(top_nodes) > 0:
        print(f"Pred top uy min/max    = {u_pred[top_nodes, 1].min():.6f}, {u_pred[top_nodes, 1].max():.6f}")

    plot_boundary_displacement_check(
        nodes_np, boundaries_np, u_pred,
        os.path.join(output_dir, "pred_boundary_displacement_check.png"),
        title="Boundary displacement check for prediction"
    )

    plot_top_boundary_uy(
        nodes_np, boundaries_np, u_pred, prescribed_uy_top,
        os.path.join(output_dir, "pred_top_boundary_uy.png"),
        title=r"Top boundary displacement for prediction"
    )

    # --------------------------------------------------------
    # 误差评估
    # --------------------------------------------------------
    metrics = evaluate_errors(
        u_pred=u_pred,
        u_fem=u_fem,
        sigma_pred=sigma_pred,
        sigma_fem=sigma_fem,
        vm_pred=vm_pred,
        vm_fem=vm_fem
    )

    rel_Pi = relative_energy_error(pred_energy_tilde, fem_energy_tilde)

    print("\n================= FINAL ERROR METRICS =================")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6e}")
    print(f"{'rel_Pi':20s}: {rel_Pi:.6e}")
    print(f"{'training_time':20s}: {training_time:.6f} s")
    print(f"{'inference_time':20s}: {inference_time:.6f} s")
    print(f"{'fem_solve_time':20s}: {fem_solve_time:.6f} s")
    print("=======================================================\n")

    # --------------------------------------------------------
    # 绘图：Pred / Error
    # --------------------------------------------------------
    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, u_pred[:, 0], u_pred,
        os.path.join(output_dir, "pred_ux.png"),
        title=r"Predicted displacement $u_x$",
        cbar_label=r"$u_x$"
    )
    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, u_pred[:, 1], u_pred,
        os.path.join(output_dir, "pred_uy.png"),
        title=r"Predicted displacement $u_y$",
        cbar_label=r"$u_y$"
    )
    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, np.sqrt(u_pred[:, 0] ** 2 + u_pred[:, 1] ** 2), u_pred,
        os.path.join(output_dir, "pred_umag.png"),
        title=r"Predicted displacement magnitude $|\mathbf{u}|$",
        cbar_label=r"$|\mathbf{u}|$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, sigma_pred[:, 0], u_pred,
        os.path.join(output_dir, "pred_sxx.png"),
        title=r"Predicted stress $\sigma_{xx}$",
        cbar_label=r"$\sigma_{xx}$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, sigma_pred[:, 1], u_pred,
        os.path.join(output_dir, "pred_syy.png"),
        title=r"Predicted stress $\sigma_{yy}$",
        cbar_label=r"$\sigma_{yy}$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, vm_pred, u_pred,
        os.path.join(output_dir, "pred_vm.png"),
        title=r"Predicted von Mises stress",
        cbar_label=r"$\sigma_{\mathrm{vM}}$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, J_pred, u_pred,
        os.path.join(output_dir, "pred_J.png"),
        title=r"Predicted Jacobian $J$",
        cbar_label=r"$J$"
    )
    plot_deformed_mesh_numpy(
        nodes_np, elements_np, u_pred,
        os.path.join(output_dir, "pred_deformed_mesh.png"),
        title="Predicted deformed mesh"
    )

    ux_err = np.abs(u_pred[:, 0] - u_fem[:, 0])
    uy_err = np.abs(u_pred[:, 1] - u_fem[:, 1])
    umag_err = np.abs(np.sqrt(u_pred[:, 0] ** 2 + u_pred[:, 1] ** 2) - np.sqrt(u_fem[:, 0] ** 2 + u_fem[:, 1] ** 2))
    sxx_err = np.abs(sigma_pred[:, 0] - sigma_fem[:, 0])
    syy_err = np.abs(sigma_pred[:, 1] - sigma_fem[:, 1])
    vm_err = np.abs(vm_pred - vm_fem)
    J_err = np.abs(J_pred - J_fem)

    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, ux_err, u_fem,
        os.path.join(output_dir, "ux_error.png"),
        title=r"Absolute error in $u_x$",
        cbar_label=r"$|u_x-u_x^{\mathrm{FEM}}|$"
    )
    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, uy_err, u_fem,
        os.path.join(output_dir, "uy_error.png"),
        title=r"Absolute error in $u_y$",
        cbar_label=r"$|u_y-u_y^{\mathrm{FEM}}|$"
    )
    plot_smooth_nodal_field_on_deformed_numpy(
        nodes_np, elements_np, umag_err, u_fem,
        os.path.join(output_dir, "umag_error.png"),
        title=r"Absolute error in $|\mathbf{u}|$",
        cbar_label=r"$||\mathbf{u}|-|\mathbf{u}|^{\mathrm{FEM}}|$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, sxx_err, u_fem,
        os.path.join(output_dir, "sxx_error.png"),
        title=r"Absolute error in $\sigma_{xx}$",
        cbar_label=r"$|\sigma_{xx}-\sigma_{xx}^{\mathrm{FEM}}|$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, syy_err, u_fem,
        os.path.join(output_dir, "syy_error.png"),
        title=r"Absolute error in $\sigma_{yy}$",
        cbar_label=r"$|\sigma_{yy}-\sigma_{yy}^{\mathrm{FEM}}|$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, vm_err, u_fem,
        os.path.join(output_dir, "vm_error.png"),
        title=r"Absolute error in von Mises stress",
        cbar_label=r"$|\sigma_{\mathrm{vM}}-\sigma_{\mathrm{vM}}^{\mathrm{FEM}}|$"
    )
    plot_smooth_element_field_on_deformed_numpy(
        nodes_np, elements_np, J_err, u_fem,
        os.path.join(output_dir, "J_error.png"),
        title=r"Absolute error in $J$",
        cbar_label=r"$|J-J^{\mathrm{FEM}}|$"
    )

    # --------------------------------------------------------
    # 训练曲线
    # --------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=FIGSIZE_HISTORY)

    ax = axes[0]
    ax.plot(history["loss"], label="Total loss", linewidth=1.5, color=COLOR_BLACK)
    ax.plot(history["U_int_tilde"], label=r"Energy: $\widetilde{U}_{int}$", linewidth=1.2, color=COLOR_RED)
    if loss_weights["weak"] != 0.0:
        ax.plot(history["weak"], label="Weak residual", linewidth=1.2, color=COLOR_BLUE)
    if loss_weights["data"] != 0.0:
        ax.plot(history["data"], label="Supervised data", linewidth=1.2, color=COLOR_GREEN)
    ax.set_xlabel("Epoch", **LABEL_KW)
    ax.set_ylabel("Value", **LABEL_KW)
    ax.set_title("Training loss history", **TITLE_KW)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75", facecolor="white", loc="best")
    leg.get_frame().set_linewidth(0.8)
    style_axis_paper(ax, equal=False)

    ax = axes[1]
    ax.plot(history["ux_rel_l2"], label=r"$u_x$ rel-L2", linewidth=1.2, color=COLOR_BLUE)
    ax.plot(history["uy_rel_l2"], label=r"$u_y$ rel-L2", linewidth=1.2, color=COLOR_RED)
    ax.plot(history["umag_rel_l2"], label=r"$|\mathbf{u}|$ rel-L2", linewidth=1.2, color=COLOR_GREEN)
    ax.set_xlabel("Epoch", **LABEL_KW)
    ax.set_ylabel("Relative error", **LABEL_KW)
    ax.set_title("Relative displacement errors", **TITLE_KW)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75", facecolor="white", loc="best")
    leg.get_frame().set_linewidth(0.8)
    style_axis_paper(ax, equal=False)

    ax = axes[2]
    ax.plot(history["lr"], label="Learning rate", linewidth=1.2, color=COLOR_PURPLE)
    ax.set_xlabel("Epoch", **LABEL_KW)
    ax.set_ylabel("LR", **LABEL_KW)
    ax.set_title("Learning-rate schedule", **TITLE_KW)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75", facecolor="white", loc="best")
    leg.get_frame().set_linewidth(0.8)
    style_axis_paper(ax, equal=False)

    ax = axes[3]
    ax.plot([log["J_min"] for log in fem_result["step_logs"]], label=r"Step-wise $J_{\min}$", linewidth=1.2, color=COLOR_RED)
    ax.plot([log["J_max"] for log in fem_result["step_logs"]], label=r"Step-wise $J_{\max}$", linewidth=1.2, color=COLOR_BLUE)
    ax.set_xlabel("Load step", **LABEL_KW)
    ax.set_ylabel("J", **LABEL_KW)
    ax.set_title("FEM load-step kinematics", **TITLE_KW)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75", facecolor="white", loc="best")
    leg.get_frame().set_linewidth(0.8)
    style_axis_paper(ax, equal=False)

    fig.tight_layout()
    savefig_paper(fig, os.path.join(output_dir, "training_history.png"))

    # --------------------------------------------------------
    # 汇总表格
    # --------------------------------------------------------
    case_name = "3-hole plate"

    summary_rows = [[
        case_name,
        f"{metrics['umag_rel_l2']:.6e}",
        f"{metrics['vm_rel_l2']:.6e}",
        f"{rel_Pi:.6e}",
        f"{metrics['umag_max_abs']:.6e}",
        f"{metrics['vm_max_abs']:.6e}",
        format_seconds(training_time),
        format_seconds(inference_time),
        format_seconds(fem_solve_time),
    ]]

    save_summary_table(
        case_rows=summary_rows,
        save_csv=os.path.join(output_dir, "summary_table.csv"),
        save_txt=os.path.join(output_dir, "summary_table.txt"),
        save_png=os.path.join(output_dir, "summary_table.png"),
    )

    print("Summary table:")
    headers = [
        "Case",
        "Rel. L2 err. (|u|)",
        "Rel. L2 err. (|σvm|)",
        "Rel(Π)",
        "Max err. (|u|)",
        "Max err. (|σvm|)",
        "Training time",
        "Inference time",
        "FEM solve time",
    ]
    print(" | ".join(headers))
    print(" | ".join(summary_rows[0]))

    # --------------------------------------------------------
    # 导出 VTU
    # --------------------------------------------------------
    export_to_vtu(
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
    # 保存文本与模型
    # --------------------------------------------------------
    with open(os.path.join(output_dir, "error_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("FINAL ERROR METRICS\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.12e}\n")
        f.write(f"rel_Pi: {rel_Pi:.12e}\n")
        f.write(f"training_time: {training_time:.12e}\n")
        f.write(f"inference_time: {inference_time:.12e}\n")
        f.write(f"fem_solve_time: {fem_solve_time:.12e}\n")

    with open(os.path.join(output_dir, "fem_reaction_summary.txt"), "w", encoding="utf-8") as f:
        f.write("FEM REACTION SUMMARY\n")
        for k, v in reaction_stats_fem.items():
            f.write(f"{k}: {v:.12e}\n")

    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": metrics,
        "rel_Pi": rel_Pi,
        "loss_weights": loss_weights,
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
        "prescribed_uy_top": prescribed_uy_top,
        "Lx": Lx,
        "Ly": Ly,
        "holes": holes,
        "E": E,
        "nu": nu,
        "thickness": thickness,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "initial_lr": adamw_lr,
        "min_lr": adamw_min_lr,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "scheduler_threshold": scheduler_threshold,
        "snapshot_interval": snapshot_interval,
        "training_time": training_time,
        "inference_time": inference_time,
        "fem_solve_time": fem_solve_time,
        "fem_energy_tilde": fem_energy_tilde,
        "pred_energy_tilde": pred_energy_tilde,
        "summary_table": summary_rows,
        "device_fem": str(device_fem),
        "device_gnn": str(device_gnn),
    }, os.path.join(output_dir, "hyperelastic_fem_plus_pyg_mpnn_model.pt"))

    print(f"All results saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
