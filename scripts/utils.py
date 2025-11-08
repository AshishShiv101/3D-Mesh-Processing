import os
import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt

# ---------- I/O ----------
def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()  # handle scenes
    return mesh

def save_trimesh(vertices, faces, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tmesh.export(path)

# ---------- Stats ----------
def vertex_stats(vertices):
    return {
        "count": len(vertices),
        "min": vertices.min(axis=0),
        "max": vertices.max(axis=0),
        "mean": vertices.mean(axis=0),
        "std": vertices.std(axis=0),
    }

# ---------- Normalization ----------
def normalize_minmax(vertices):
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    span = np.clip(vmax - vmin, 1e-12, None)
    norm = (vertices - vmin) / span
    ctx = {"type": "minmax", "vmin": vmin, "vmax": vmax}
    return norm, ctx

def denormalize_minmax(norm_vertices, ctx):
    vmin, vmax = ctx["vmin"], ctx["vmax"]
    return norm_vertices * (vmax - vmin) + vmin

def normalize_unit_sphere(vertices):
    mean = vertices.mean(axis=0)
    centered = vertices - mean
    scale = np.max(np.linalg.norm(centered, axis=1))
    scale = max(scale, 1e-12)
    norm = centered / scale * 0.5 + 0.5  # map to [0,1] for shared quantizer
    ctx = {"type": "unit_sphere", "mean": mean, "scale": scale}
    return norm, ctx

def denormalize_unit_sphere(norm_vertices, ctx):
    # reverse the 0.5 shift/scale
    centered = (norm_vertices - 0.5) * ctx["scale"] * 2.0
    return centered + ctx["mean"]

# ---------- Quantization ----------
def quantize01(norm_vertices, bins=1024):
    q = np.clip(np.floor(norm_vertices * (bins - 1) + 0.5), 0, bins - 1).astype(np.int32)
    return q

def dequantize01(q, bins=1024):
    return q.astype(np.float64) / (bins - 1)

# ---------- Error ----------
def mse(a, b, axis=None):
    return np.mean((a - b) ** 2, axis=axis)

def mae(a, b, axis=None):
    return np.mean(np.abs(a - b), axis=axis)

# ---------- Visualization (Open3D offscreen) ----------
def trimesh_to_o3d(mesh: trimesh.Trimesh):
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def save_mesh_screenshot(vertices, faces, out_png, width=1200, height=900, bg=(1,1,1), show_back_face=True):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    o3d_mesh = trimesh_to_o3d(tmesh)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg)
    opt.mesh_show_back_face = show_back_face

    vis.add_geometry(o3d_mesh)

    # Fit the view nicely
    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    ctr = vis.get_view_control()
    ctr.set_lookat(bbox.get_center())
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    ctr.set_front([1, 1, 1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.7 if diag > 0 else 1.0)

    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(out_png, do_render=True)
    vis.destroy_window()

# ---------- Plot helpers ----------
def plot_error_bars(mse_xyz, title, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    labels = ['X','Y','Z']
    plt.figure(figsize=(5,4), dpi=140)
    plt.bar(labels, mse_xyz)
    plt.ylabel('MSE')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def scatter_slice(vertices, out_png, title="XY scatter (Z≈median)", n=5000):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    if len(vertices) > n:
        idx = np.random.choice(len(vertices), n, replace=False)
        verts = vertices[idx]
    else:
        verts = vertices
    z_med = np.median(verts[:,2])
    band = np.abs(verts[:,2] - z_med) < 0.01 * (verts[:,2].ptp() + 1e-9)
    pts = verts[band]
    plt.figure(figsize=(5,5), dpi=140)
    plt.scatter(pts[:,0], pts[:,1], s=1)
    plt.xlabel('X'); plt.ylabel('Y'); plt.title(title)
    plt.axis('equal'); plt.tight_layout()
    plt.savefig(out_png); plt.close()
