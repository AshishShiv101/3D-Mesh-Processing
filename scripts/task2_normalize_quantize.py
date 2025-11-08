import os, glob, numpy as np
from utils import (load_mesh, save_trimesh, save_mesh_screenshot,
                   normalize_minmax, normalize_unit_sphere,
                   quantize01)

DATA_DIR = "data"
OUT_NORM = "output/normalized"
OUT_QUANT = "output/quantized"
SHOT_DIR = "report/screenshots"
BINS = 1024

def process_one(path):
    name = os.path.splitext(os.path.basename(path))[0]
    mesh = load_mesh(path)
    V, F = mesh.vertices, mesh.faces

    # --- Min-Max ---
    Vn_mm, ctx_mm = normalize_minmax(V)
    save_trimesh(Vn_mm, F, os.path.join(OUT_NORM, f"{name}_minmax.obj"))
    save_mesh_screenshot(Vn_mm, F, os.path.join(SHOT_DIR, f"{name}_minmax.png"))

    Q_mm = quantize01(Vn_mm, BINS)
    save_trimesh(Q_mm.astype(np.float32), F, os.path.join(OUT_QUANT, f"{name}_minmax_q.obj"))
    save_mesh_screenshot(Q_mm, F, os.path.join(SHOT_DIR, f"{name}_minmax_q.png"))

    # --- Unit Sphere (mapped to [0,1]) ---
    Vn_us, ctx_us = normalize_unit_sphere(V)
    save_trimesh(Vn_us, F, os.path.join(OUT_NORM, f"{name}_unitsphere.obj"))
    save_mesh_screenshot(Vn_us, F, os.path.join(SHOT_DIR, f"{name}_unitsphere.png"))

    Q_us = quantize01(Vn_us, BINS)
    save_trimesh(Q_us.astype(np.float32), F, os.path.join(OUT_QUANT, f"{name}_unitsphere_q.obj"))
    save_mesh_screenshot(Q_us, F, os.path.join(SHOT_DIR, f"{name}_unitsphere_q.png"))

def main():
    os.makedirs(OUT_NORM, exist_ok=True)
    os.makedirs(OUT_QUANT, exist_ok=True)
    os.makedirs(SHOT_DIR, exist_ok=True)
    for p in glob.glob(os.path.join(DATA_DIR, "*.obj")):
        process_one(p)

if __name__ == "__main__":
    main()
