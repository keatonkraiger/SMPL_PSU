import os
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation
import trimesh
import numpy as np
from PIL import Image
import io

def compare_com_trajectory(subject, take, npz_dir, mat_path, save_dir, ds_rate=5, create_com_movie=False):
    os.makedirs(save_dir, exist_ok=True)
    npz_files = sorted(glob(f"{npz_dir}/*.npz"))
    if not npz_files:
        print(f"[WARN] No NPZ files found in {npz_dir}")
        return None

    # Load CoM from NPZ
    print(f"[INFO] Loading CoM from NPZ files in {npz_dir} for {subject} {take}")
    npz_coms = []
    for npz_file in npz_files:
        data = np.load(npz_file)
        com = data['com']
        com = np.array((com[0], -com[2], com[1])) * 10e2  # reorder & convert to mm
        npz_coms.append(com)

    # Load ground-truth CoM from .mat
    mat_coms = loadmat(mat_path)["CoM"]
    mat_coms = mat_coms[::ds_rate]  # downsample to match ds_rate
    npz_arr, mat_arr, diffs = [], [], []
    print(f"[INFO] Comparing CoM trajectories for {subject} {take}")
    invalid = 0
    for i in range(min(len(npz_coms), len(mat_coms))):
        if mat_coms[i, -1] == 0:  # invalid frame
            invalid += 1
            continue
        npz_arr.append(npz_coms[i])
        mat_arr.append(mat_coms[i, :3])
        diffs.append(npz_coms[i] - mat_coms[i, :3])

    print(f"[INFO] Found {len(npz_arr)} valid frames, {invalid} invalid mat CoM frames in {take}")
    npz_arr = np.array(npz_arr)
    mat_arr = np.array(mat_arr)
    diffs = np.array(diffs)

    if len(diffs) == 0:
        print(f"[WARN] No valid CoM frames to compare in {take}")
        return None

    # Save npz, mat, diff, and num of invalid mat frames
    npy_path = os.path.join(save_dir, f"com_comparison.npz")
    data = {
        'npz_coms': npz_arr,
        'mat_coms': mat_arr,
        'diffs': diffs,
        'invalid_frames': invalid
    }
    np.savez(npy_path, **data)
    print(f"[INFO] Saved CoM data to {npy_path}")

    # Plot stats
    fig, ax = plt.subplots()
    mean_error = np.mean(np.abs(diffs), axis=0)
    std_error = np.std(np.abs(diffs), axis=0)
    ax.bar(np.arange(3), mean_error, yerr=std_error, capsize=5, edgecolor='black', colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.set_ylabel("Mean Absolute Error (mm)")
    ax.set_title(f"Mean CoM Error: {subject} {take}")
    plt_path = os.path.join(save_dir, f"{subject}_{take}_com_error.png")
    fig.savefig(plt_path, dpi=200)
    plt.close(fig)

    # Optional animation
    fig = plt.figure(figsize=(14, 14))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, height_ratios=[4, 1])
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax2d = fig.add_subplot(gs[1])

    frames = np.arange(len(diffs))
    ax2d.plot(frames, diffs[:, 0], label='X diff')
    ax2d.plot(frames, diffs[:, 1], label='Y diff')
    ax2d.plot(frames, diffs[:, 2], label='Z diff')
    timeline = ax2d.axvline(0, color='k', linestyle='--')
    ax2d.set_xlabel("Frame")
    ax2d.set_ylabel("Difference (mm)")
    ax2d.legend(loc='upper right')
    ax2d.grid()

    xlims = (npz_arr[:, 0].min() - 100, npz_arr[:, 0].max() + 100)
    ylims = (npz_arr[:, 1].min() - 100, npz_arr[:, 1].max() + 100)
    zlims = (npz_arr[:, 2].min() - 100, npz_arr[:, 2].max() + 100)

    def update(frame):
        ax3d.cla()
        ax3d.scatter(npz_arr[frame, 0], npz_arr[frame, 1], npz_arr[frame, 2], c='r', label='NPZ CoM')
        ax3d.scatter(mat_arr[frame, 0], mat_arr[frame, 1], mat_arr[frame, 2], c='g', label='MAT CoM')
        ax3d.plot(npz_arr[:frame+1, 0], npz_arr[:frame+1, 1], npz_arr[:frame+1, 2], 'r--', alpha=0.5)
        ax3d.plot(mat_arr[:frame+1, 0], mat_arr[:frame+1, 1], mat_arr[:frame+1, 2], 'g--', alpha=0.5)

        ax3d.set_xlim(xlims)
        ax3d.set_ylim(ylims)
        ax3d.set_zlim(zlims)
        ax3d.set_title(f'3D CoM Trajectory (Frame {frame})')
        ax3d.set_xlabel("X (mm)")
        ax3d.set_ylabel("Y (mm)")
        ax3d.set_zlabel("Z (mm)")
        ax3d.legend()

        # Update timeline marker
        window = 100
        start = max(0, frame - window)
        end = min(len(diffs), frame + window)
        ax2d.set_xlim(start, end)
        timeline.set_xdata([frame])

        return ax3d, timeline

    if create_com_movie:
        print(f"[INFO] Creating animation for {subject} {take} CoM trajectory")
        start = time.time()
        ani = FuncAnimation(fig, update, frames=len(diffs), interval=50, blit=False, repeat=False)
        anim_path = os.path.join(save_dir, f"{subject}_{take}_com_animation.mp4")
        fps = 50 // ds_rate
        ani.save(anim_path, fps=fps, dpi=150)
        end = time.time()
        print(f"[INFO] Animation saved to {anim_path} (took {end - start:.2f}s)")

    return diffs  # You can accumulate these later

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
