import argparse
import os
from pathlib import Path
import json
import subprocess
import bpy
from functools import partial
import numpy as np
import torch
from smplx import SMPLX
from scipy.spatial.transform import Rotation as R
import contextlib
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from smpl_psu.smplx_support.body_segments import SMPLX_PART_BOUNDS, PART_VID_FID, SimpleCoM
from smpl_psu.blender_support.smplx_support import *
from smpl_psu.blender_support.blend_util import *
from smpl_psu.utils.viz import compare_com_trajectory

# ------------------------------------------------------------------------------
#                                     CONFIG
# ------------------------------------------------------------------------------
MAX_FRAMES=np.inf
OVERWRITE=False
DEBUG=False

cfg_global = {
    'mesh': {
        'npz_path':           '/mnt/e/Research/SMPL/SMPL_TMM/PSU_AMASS/Subject1_MOCAP_MRK_1_gt_stageii.npz',
        'model_folder':       'body_models/smplx',          # contains e.g. SMPLX_FEMALE.npz
        'output_obj_dir':     'output_objs',
        'gender':             'female',
        'ds_rate':            1,     # down-sample every N frames
        'num_betas':          10,
        'scale_cm_to_m':      False,
        'n_jobs':             16,     # number of parallel jobs for obj conversion
    },
    'motion': {
        'coordinate_system': 'global',  # 'local' or 'global'
    },
    'env': {
        'floor_size': 10 # suggested 10 or 3
    },
    'output': {
        'save_npz': True,  # save the .npz file after conveting everything
        'save_blend': True,  # save the .blend file after rendering
        'create_animation': False,  # create a single animation from all frames
        'render_images': False,  # whether to render images
    },
    'dirs': {
        'save_dir':         'output',  # where to save all results
        'render_out_dir':     'render',
        'cleanup_objs':     True,  # remove old output dirs if they exist
        'cleanup_imgs':   False,  # remove old output dirs if they exist
    },
    'render': {
        'engine':             'CYCLES',  # or 'CYCLES'
        'cycles_samples':     5,
        'device':           'GPU',  # 'CPU' or 'GPU'
        'cycles_denoise':     True,
        'eevee_gtao':         True,
        'eevee_shadow_size':  512,
        'resolution':         (1920, 1080),
        'stop_early':  False, # render only the first frame
        'suppress_output': True,
        'camera_view': 1, # view 1 or 2 if global
        "render_com": 'none', # options include ['xyz', 'xy', 'none']
    },
}
# ------------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_output(enabled=True):
    """Suppress stdout and stderr even for native code (e.g., Blender)."""
    if not enabled:
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    # Save original file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        # Redirect stdout and stderr to devnull
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull)

# ------------------------------------------------------------------------------
#                                  MESH CONVERSION
# ------------------------------------------------------------------------------
import pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def rotate_points_xyz(pts, angles_deg):
    """Rotate points by Euler angles (in degrees) around x, y, z axes."""
    Rmat = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
    return np.einsum('ij,bnj->bni', Rmat, pts)


def process_frame(i, model, trans, root_orient, pose_body, pose_hand, jaw_pose, eye_pose,
                  betas, scale, coordinate_system, theta_z_deg, part_vid_fid, part_bounds,
                  obj_dir, npz_dir, save_npz, ds_rate):
    name = os.path.join(obj_dir, f"frame_{i:05d}.obj")
    if not OVERWRITE and os.path.exists(name):
        if DEBUG:
            print(f"[DEBUG] Skipping existing file: {name}")
        return None

    inputs = {
        'global_orient':     torch.tensor(root_orient[i:i+1], dtype=torch.float32),
        'body_pose':         torch.tensor(pose_body[i:i+1],   dtype=torch.float32),
        'betas':             torch.tensor(betas[None],         dtype=torch.float32),
        'jaw_pose':          torch.tensor(jaw_pose[i:i+1],    dtype=torch.float32),
        'leye_pose':         torch.tensor(eye_pose[i:i+1,:3], dtype=torch.float32),
        'reye_pose':         torch.tensor(eye_pose[i:i+1,3:], dtype=torch.float32),
        'left_hand_pose':    torch.tensor(pose_hand[i,0].reshape(1,-1), dtype=torch.float32),
        'right_hand_pose':   torch.tensor(pose_hand[i,1].reshape(1,-1), dtype=torch.float32),
    }
    if coordinate_system == 'global':
        inputs['transl'] = torch.tensor(trans[i:i+1], dtype=torch.float32) * scale

    out = model(**inputs, return_verts=True)
    verts = out.vertices[0].detach().cpu().numpy()[None]

    if coordinate_system == 'local':
        verts = rotate_points_xyz(verts, [0, 0, -theta_z_deg])
    verts = rotate_points_xyz(verts, [-90, 0, 0])
    verts = verts[0]

    # COM
    faces = model.faces
    t_faces = torch.from_numpy(faces.astype(np.int32))
    com_calc = SimpleCoM(part_vid_fid, part_bounds, t_faces)
    t_verts = torch.from_numpy(verts).unsqueeze(0)
    com = com_calc.compute_com(t_verts)[0].detach().cpu().numpy()

    # Write OBJ
    with open(name, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    # Save NPZ
    if save_npz:
        save_path = os.path.join(npz_dir, f"frame_{i:05d}.npz")
        np.savez_compressed(save_path,
                            verts=verts.astype(np.float32),
                            faces=faces.astype(np.int32),
                            com=com.astype(np.float32),
                            frame=i,
                            ds_rate=ds_rate)

    return {'frame': i, 'path': name, 'com': com}

def convert_npz_to_objs(cfg):
    """Load NPZ, run SMPL-X, write per-frame .obj files, return list of paths + initial trans/orient for camera."""
    data = np.load(cfg['mesh']['npz_path'])
    model = SMPLX(
        model_path=os.path.join(cfg['mesh']['model_folder'],
                                f"SMPLX_{cfg['mesh']['gender'].upper()}.npz"),
        gender=cfg['mesh']['gender'],
        batch_size=1
    )

    trans       = data['trans']
    root_orient = data['root_orient']
    pose_body   = data['pose_body']
    pose_hand   = data['pose_hand']
    jaw_pose    = data['pose_jaw']
    eye_pose    = data['pose_eye']
    betas       = data['betas'][:cfg['mesh']['num_betas']]
    
    scale = 0.01 if cfg['mesh']['scale_cm_to_m'] else 1.0  # cm → m
    coordinate_system = cfg['motion']['coordinate_system']

    # Pre‑compute heading if in local mode
    if coordinate_system == 'local':
        R0, _ = Rodrigues(root_orient[0])
        theta_z_deg = np.rad2deg(np.arctan2(R0[1, 0], R0[0, 0]))
        print(f"[INFO] Local mode: rotating subject -{theta_z_deg:.2f}° to face camera")
    else:
        theta_z_deg = 0.0

    # Ensure output dirs
    os.makedirs(cfg['dirs']['obj_dir'], exist_ok=True)
    if cfg['output'].get('save_npz'):
        os.makedirs(cfg['dirs']['npz_dir'], exist_ok=True)

    common_args = dict(
        model=model,
        trans=trans,
        root_orient=root_orient,
        pose_body=pose_body,
        pose_hand=pose_hand,
        jaw_pose=jaw_pose,
        eye_pose=eye_pose,
        betas=betas,
        scale=scale,
        coordinate_system=coordinate_system,
        theta_z_deg=theta_z_deg,
        part_vid_fid=load_pickle(PART_VID_FID),
        part_bounds=load_pickle(SMPLX_PART_BOUNDS),
        obj_dir=cfg['dirs']['obj_dir'],
        npz_dir=cfg['dirs']['npz_dir'],
        save_npz=cfg['output'].get('save_npz', False),
        ds_rate=cfg['mesh']['ds_rate'],
    )

    # Frame indices to process
    idxs = list(range(0, trans.shape[0], cfg['mesh']['ds_rate']))
    wrapped = partial(process_frame, **common_args)
    results = Parallel(n_jobs=cfg['mesh']['n_jobs'])(
        delayed(wrapped)(i) for i in tqdm(idxs, desc="Converting frames", total=len(idxs))
    ) 
     
    # Filter out skipped frames
    out_paths = [r for r in results if r is not None]

    # Honor stop_early / max_frames
    if cfg['render']['stop_early'] and len(out_paths) > MAX_FRAMES:
        out_paths = out_paths[:MAX_FRAMES]

    return out_paths

# ------------------------------------------------------------------------------
#                                 SCENE SETUP
# ------------------------------------------------------------------------------
def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

def set_world_background_color(color=(1.0, 1.0, 1.0, 1.0), strength=1.0):
    """Set the world background to a specific color (white by default)."""
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = color  # RGBA
    bg.inputs[1].default_value = strength  # Strength

def set_world_transparent():
    """Replace the World background with a truly transparent shader."""
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # remove any existing background nodes
    for n in list(nodes):
        nodes.remove(n)

    # create a transparent node + output
    transp = nodes.new(type="ShaderNodeBsdfTransparent")
    out   = nodes.new(type="ShaderNodeOutputWorld")

    links.new(transp.outputs["BSDF"], out.inputs["Surface"])

def enable_transparent_background():
    scene = bpy.context.scene
    scene.render.film_transparent = True  # Only for Cycles and Eevee Next
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

def force_background_to_white_compositor():
    """Force the render background to pure white using Blender's compositor (post-render)."""
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # 1. Render Layers node (input from renderer)
    render_layers = nodes.new(type='CompositorNodeRLayers')

    # 2. Alpha Over node (composite over white)
    alpha_over = nodes.new(type='CompositorNodeAlphaOver')
    alpha_over.inputs[1].default_value = (1, 1, 1, 1)  # solid white RGBA

    # 3. Composite output node
    composite = nodes.new(type='CompositorNodeComposite')

    # Connect the nodes
    links.new(render_layers.outputs['Image'], alpha_over.inputs[2])  # Image → bottom
    links.new(alpha_over.outputs['Image'], composite.inputs['Image'])
    # links.new(alpha_over.outputs['Image'], viewer.inputs['Image'])  # for preview



# ------------------------------------------------------------------------------
#                               IMPORT & RENDER LOOP
# ------------------------------------------------------------------------------


def import_and_render(objs, cfg):
    suppress = cfg['render'].get('suppress_output', False)
    for i, data in tqdm(enumerate(objs), desc="Rendering frames", total=len(objs)):
        frame_num = data['frame']
        path = data['path']
        outpath = os.path.join(cfg['dirs']['render_out_dir'], f"frame_{frame_num:05d}.png")
        if not OVERWRITE and os.path.exists(outpath):
            continue 
        
        with suppress_output(enabled=suppress):
            bpy.ops.wm.obj_import(filepath=path)
            obj = bpy.context.selected_objects[0]
            obj.name = f"frame_{frame_num:05d}"
            alpha = 1.0 if cfg['render']['render_com'] == 'none' else 0.5
            mat = create_smplx_material(alpha=alpha)
            obj.data.materials.append(mat)

            if  'com' in data and cfg['render']['render_com'] != 'none':
                com_coord  = data['com']  # numpy (x, y, z)
                # Convert to Blender's coordinate system (Z-up, Y-forward)
                if cfg['motion']['coordinate_system'] == 'local':
                    com_coord = rotate_points_xyz(com_coord.reshape(1, 1, 3), [0, 0, -90])[0, 0]
                elif cfg['motion']['coordinate_system'] == 'global':
                    com_coord = rotate_points_xyz(com_coord.reshape(1, 1, 3), [-90, 0, 0])[0, 0]
                    com_coord = np.array((com_coord[0], -com_coord[1], com_coord[2]))
                    print(f"[DEBUG] CoM in global coords: {com_coord}")
                else:
                    raise ValueError(f"Unsupported coordinate system: {cfg['motion']['coordinate_system']}")
                
                com_coord[2] *= -1  # Flip Z
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=0.05,                      # adjust size as needed
                    location=tuple(com_coord),
                    segments=16,
                    ring_count=8
                )
                marker = bpy.context.active_object
                marker.name = f"CoM_marker_{frame_num:05d}"
                # red, fully opaque
                com_mat = bpy.data.materials.get("CoM_Mat") or bpy.data.materials.new("CoM_Mat")
                com_mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)
                marker.data.materials.append(com_mat)

            if cfg['output']['save_blend']:
                blend_path = os.path.join(cfg['dirs']['blend_dir'], f"frame_{frame_num:05d}.blend")
                if os.path.exists(blend_path):
                    os.remove(blend_path)
                bpy.ops.wm.save_as_mainfile(filepath=blend_path)
               
            if cfg['output']['render_images']:
                bpy.context.scene.render.filepath = outpath
                bpy.ops.render.render(write_still=True)
                
            bpy.data.objects.remove(obj, do_unlink=True)

        if cfg['render']['stop_early'] and i >= MAX_FRAMES:
            break
# ------------------------------------------------------------------------------
#                                     MAIN
# ------------------------------------------------------------------------------
def frames_to_video(frame_dir, output_path, fps=50):
    """
    Use ffmpeg to stitch a directory of PNG frames named frame_00000.png, frame_00001.png, … into a video.
    """
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(".png")
    ])

    # Create temporary frames.txt file
    list_path = os.path.join(frame_dir, "frames.txt")
    with open(list_path, "w") as f:
        for path in frame_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    # Build ffmpeg command using concat mode
    cmd = [
        "ffmpeg",
        "-y",                        # Overwrite output file if exists
        "-r", str(fps),              # Input framerate
        "-f", "concat",
        "-safe", "0",                # Allow absolute paths
        "-i", list_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[INFO] Video saved to: {output_path}")
 
def cleanup_dirs(cfg):
    """
    Remove old output directories if they exist.
    """
    import shutil 
    # Force removal of old directories
    if cfg['dirs']['cleanup_objs']:
        shutil.rmtree(cfg['dirs']['obj_dir'], ignore_errors=True)
    if cfg['dirs']['cleanup_imgs']:
        shutil.rmtree(cfg['dirs']['render_out_dir'], ignore_errors=True)
    
def setup_dirs(cfg, subject_name, take_name): 
    """
    Update save_dir, obj_dir, render_dir to follow structured output:
    output_dir/Subject/Take_N/<coordinate_system>/{objs,images,config}
    """
    coord_system = cfg['motion']['coordinate_system']
    cam_view = f"V{cfg['render']['camera_view']}" if coord_system == 'global' else ''
    base_dir = os.path.join(cfg['dirs']['save_dir'], subject_name, take_name, coord_system)
    if cam_view:
        base_dir = os.path.join(base_dir, cam_view)
        
    obj_dir = os.path.join(base_dir, 'objs')
    render_dir = os.path.join(base_dir, 'images')
    blend_dir = os.path.join(base_dir, 'blends')
    npz_dir = os.path.join(base_dir, 'npz')
    com_dir = os.path.join(base_dir, 'com')

    cfg['dirs']['blend_dir'] = blend_dir
    cfg['dirs']['obj_dir'] = obj_dir
    cfg['dirs']['render_out_dir'] = render_dir
    cfg['dirs']['save_dir'] = base_dir  # unify root
    cfg['dirs']['npz_dir'] = npz_dir
    cfg['dirs']['com_dir'] = com_dir

    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(blend_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(com_dir, exist_ok=True)
    if cfg['output'].get('save_npz'):
        os.makedirs(npz_dir, exist_ok=True)

    # Save config as JSON in the same folder
    config_out = os.path.join(base_dir, 'config.json')
    with open(config_out, 'w') as f:
        json.dump(cfg, f, indent=2)

    return cfg
 
def main(cfg):
    obj_paths = convert_npz_to_objs(cfg)

    clear_scene()
    z_offset = 0.0 if cfg['motion']['coordinate_system'] == 'global' else -0.85
    setup_floor(floor_size=cfg['env']['floor_size'],z_offset=z_offset)
    setup_light(cfg)
    setup_camera(cfg)

 
    apply_render_settings(cfg)
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
    enable_transparent_background()
    force_background_to_white_compositor()

    # 3) Import each OBJ & render
    import_and_render(obj_paths, cfg)
    print("All done generating frames!")

    if cfg['output']['create_animation']:
        create_animation(cfg)

    video_path = os.path.join(cfg['dirs']['save_dir'], 'render.mp4')
    if cfg['output']['render_images']:
        with suppress_output(enabled=cfg['render']['suppress_output']):
            frames_to_video(
                os.path.join(cfg['dirs']['render_out_dir']),
                video_path,
                fps=cfg['render'].get('fps', 50)  # default to 50 FPS if not specified
            )
    
    cleanup_dirs(cfg)

import re
def extract_take_number(filename):
    """Extract the take number (e.g. 1, 10) from a filename like 'Subject1_MOCAP_MRK_10_gt_stageii.npz'."""
    match = re.search(r'_MRK_(\d+)_gt', filename)
    if not match:
        raise ValueError(f"Could not extract take number from {filename}")
    return int(match.group(1))


from copy import deepcopy
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Generate SMPLX meshes and render them.")
    args.add_argument('--ds_rate', type=int, default=1, help="Downsample")
    args.add_argument('--smplx_data_dir', type=str, default='/mnt/d/Data/PSU100/SMPLX', help="Path to SMPLX Subject_wise data directory")
    args.add_argument('--stop_early', action='store_true', help="Stop after processing the first MAX_FRAMES frames")
    args.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 11)), help="List of subject numbers to process (default: 1-10)")
    args = args.parse_args()
    
    cfg_global['mesh']['ds_rate'] = args.ds_rate
    cfg_global['render']['stop_early'] = args.stop_early
    

    for i in args.subjects:
        print(f"[INFO] Processing Subject {i}")
        subject = f"Subject{i}"
        sub_dir = os.path.join(args.smplx_data_dir, subject)

        # Detect gender
        if os.path.exists(os.path.join(sub_dir, "male_stagei.npz")):
            gender = "male"
        else:
            gender = "female"
            
        cfg = deepcopy(cfg_global) 
        cfg['mesh']['gender'] = gender
        
        # All take files for this subject
        take_files = glob(os.path.join(sub_dir, "*_gt_stageii.npz"))

        all_diffs = []
        # After processing all takes
        for take_path in take_files:
            take_cfg = deepcopy(cfg)
            take_number = extract_take_number(Path(take_path).stem)
            take_name = f"Take_{take_number:02d}"
            take_cfg = setup_dirs(take_cfg, subject, take_name)
            take_cfg['mesh']['npz_path'] = take_path
           
            print(f"[INFO] Processing {take_name} for {subject}.") 
            
            try:
                main(take_cfg)
            except Exception as e:
                print(f"[ERROR] Failed to process {take_name} for {subject}: {e}")
                continue
                
            npz_dir = take_cfg['dirs']['npz_dir']
            mat_com_path = f"/mnt/d/Data/PSU100/Subject_wise/{subject}/CoM_{take_number}.mat"
            com_vis_dir = take_cfg['dirs']['com_dir']

            # Get set fps w.r.t ds_rate
            diffs = compare_com_trajectory(subject, take_name, npz_dir, mat_com_path, com_vis_dir, ds_rate=cfg_global['mesh']['ds_rate'], create_com_movie=False)
            if diffs is not None:
                all_diffs.append(diffs)
            
        # Save overall stats per subject
        if all_diffs:
            all_diffs = np.concatenate(all_diffs, axis=0)
            mean_err = np.abs(all_diffs).mean(axis=0)
            std_err  = np.abs(all_diffs).std(axis=0)


            fig, ax = plt.subplots()
            ax.bar(['X', 'Y', 'Z'], mean_err, yerr=std_err, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
            ax.set_ylabel("Mean Absolute CoM Error (mm)")
            ax.set_title(f"{subject} — Overall CoM Error")

            summary_path = os.path.join(cfg['dirs']['save_dir'], 'CoM_diffs', f"{subject}.png")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            plt.tight_layout()
            fig.savefig(summary_path, dpi=200)
            plt.close(fig)

            print(f"[INFO] Saved overall error summary to: {summary_path}")