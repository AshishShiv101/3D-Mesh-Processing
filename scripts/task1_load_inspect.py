import os, glob
import numpy as np
from utils import load_mesh, vertex_stats, save_mesh_screenshot, scatter_slice

DATA_DIR = "data"
SHOT_DIR = "report/screenshots"

def main():
    for path in glob.glob(os.path.join(DATA_DIR, "*.obj")):
        name = os.path.splitext(os.path.basename(path))[0]
        mesh = load_mesh(path)
        V, F = mesh.vertices, mesh.faces

        stats = vertex_stats(V)
        print(f"[{name}] count={stats['count']}")
        print(f"  min={stats['min']}, max={stats['max']}")
        print(f"  mean={stats['mean']}, std={stats['std']}")

        # Save pretty renders
        save_mesh_screenshot(V, F, os.path.join(SHOT_DIR, f"{name}_original.png"))
        scatter_slice(V, os.path.join(SHOT_DIR, f"{name}_original_xy.png"),
                      title=f"{name}: XY slice")

if __name__ == "__main__":
    main()
