import trimesh
import numpy as np

import trimesh
import numpy as np
from PIL import Image
import io

def plot_smplx_with_com(vertices, faces, com, mode='transparent', title='SMPL-X with CoM', save_path=None):
    """
    Visualizes SMPL-X mesh with CoM and optionally saves a cropped image.

    Parameters:
        vertices:  (V, 3) numpy array
        faces:     (F, 3) numpy array
        com:       (3,) numpy array
        mode:      'transparent' or 'wireframe'
        title:     window title
        save_path: optional path to save rendered image (e.g. "com.png")
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    sphere = trimesh.creation.icosphere(radius=0.02)
    sphere.apply_translation(com)
    sphere.visual.vertex_colors = [255, 0, 0, 255]

    if mode == 'transparent':
        mesh.visual.face_colors = [150, 150, 150, 80]
    elif mode == 'wireframe':
        mesh.visual.face_colors = [0, 0, 0, 0]
        mesh.visual.edge_colors = [0, 0, 255, 255]
    else:
        raise ValueError("mode must be 'transparent' or 'wireframe'")

    scene = trimesh.Scene([mesh, sphere])
    scene.set_camera(angles=[0, -np.pi/6, 0], distance=2.5, center=com)

    if save_path is not None:
        # Render as image and crop around non-background
        png = scene.save_image(resolution=(800, 800), visible=True)
        image = Image.open(io.BytesIO(png)).convert("RGBA")

        # Auto-crop to bounding box of non-transparent content
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

        image.save(save_path)
        print(f"[INFO] Saved image to: {save_path}")
    else:
        # Interactive viewer
        scene.show(title=title)
