import os, glob, numpy as np
from utils import (load_mesh, save_trimesh, save_mesh_screenshot,
                   denormalize_minmax, denormalize_unit_sphere,
                   normalize_minmax, normalize_unit_sphere,
                   quantize01, dequantize01, mse, plot_error_bars)

DATA_DIR = "data"
OUT_RECON = "output/reconstructed"
PLOT_DIR = "output/plots"
SHOT_DIR = "report/screenshots"
BINS = 1024

def pipeline(vertices, faces, norm_method: str):
    if norm_method == "minmax":
        Vn, ctx = normalize_minmax(vertices)
        de_norm = denormalize_minmax
    else:
        Vn, ctx = normalize_unit_sphere(vertices)
        de_norm = denormalize_unit_sphere

    Q = quantize01(Vn, BINS)
    Vn_rec = dequantize01(Q, BINS)
    V_rec = de_norm(Vn_rec, ctx)
    return Vn, Q, V_rec

def main():
    os.makedirs(OUT_RECON, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(SHOT_DIR, exist_ok=True)

    for path in glob.glob(os.path.join(DATA_DIR, "*.obj")):
        name = os.path.splitext(os.path.basename(path))[0]
        mesh = load_mesh(path)
        V, F = mesh.vertices, mesh.faces

        for method in ["minmax", "unitsphere"]:
            Vn, Q, Vrec = pipeline(V, F, method)

            # Save and visualize reconstructed mesh
            out_obj = os.path.join(OUT_RECON, f"{name}_{method}_recon.obj")
            save_trimesh(Vrec, F, out_obj)
            save_mesh_screenshot(Vrec, F, os.path.join(SHOT_DIR, f"{name}_{method}_recon.png"))

            # Errors
            mse_xyz = mse(V, Vrec, axis=0)
            plot_error_bars(mse_xyz, f"{name} – {method} MSE per axis",
                            os.path.join(PLOT_DIR, f"{name}_{method}_mse_xyz.png"))

            # Quick overall print
            print(f"{name} [{method}] MSE xyz = {mse_xyz}, overall={mse(V,Vrec):.6e}")

if __name__ == "__main__":
    main()
