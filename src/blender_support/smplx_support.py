import os
import numpy as np
import torch
from smplx import SMPLX

def convert_to_mesh_once(
    npz_path,
    model_folder,
    output_folder,
    gender='neutral',
    ds_rate=1,
    num_betas=10,
    scale_cm_to_m=True,
):
    """
    Convert a SOMA/AMASS .npz file into a sequence of .obj meshes using SMPL-X.

    Args:
        npz_path (str): Path to the *_stageii.npz file.
        model_folder (str): Path to SMPL-X model directory (should contain e.g. smplx/neutral/model.npz).
        output_folder (str): Where to save the .obj files.
        gender (str): 'male', 'female', or 'neutral'.
        ds_rate (int): Downsample rate.
        num_betas (int): Number of shape coefficients.
        scale_cm_to_m (bool): Whether to scale translation from cm to meters.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Loading: {npz_path}")
    data = np.load(npz_path)

    # Load SMPL-X model
    model_path = os.path.join(model_folder, gender, 'model.npz')
    model = SMPLX(model_path=model_path, gender=gender, batch_size=1)

    # Extract inputs
    trans = data['trans']                       # [N, 3]
    root_orient = data['root_orient']           # [N, 3]
    pose_body = data['pose_body']               # [N, 63]
    pose_hand = data['pose_hand']               # [N, 2, 15, 3]
    jaw_pose = data['pose_jaw']                 # [N, 3]
    eye_pose = data['pose_eye']                 # [N, 6]
    betas = data['betas'][:num_betas]           # [num_betas]

    n_frames = trans.shape[0]
    print(f"[INFO] Found {n_frames} frames. Exporting every {ds_rate} frame(s).")

    for i in range(0, n_frames, ds_rate):
        body_pose = torch.tensor(pose_body[i:i+1], dtype=torch.float32)
        global_orient = torch.tensor(root_orient[i:i+1], dtype=torch.float32)

        transl = torch.tensor(trans[i:i+1], dtype=torch.float32)
        if scale_cm_to_m:
            transl /= 100.0  # Convert cm to meters

        betas_tensor = torch.tensor(betas[None], dtype=torch.float32)

        jaw = torch.tensor(jaw_pose[i:i+1], dtype=torch.float32)
        leye = torch.tensor(eye_pose[i:i+1, :3], dtype=torch.float32)
        reye = torch.tensor(eye_pose[i:i+1, 3:], dtype=torch.float32)

        lhand = torch.tensor(pose_hand[i, 0].reshape(1, -1), dtype=torch.float32)
        rhand = torch.tensor(pose_hand[i, 1].reshape(1, -1), dtype=torch.float32)

        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas_tensor,
            transl=transl,
            jaw_pose=jaw,
            leye_pose=leye,
            reye_pose=reye,
            left_hand_pose=lhand,
            right_hand_pose=rhand,
            return_verts=True,
        )

        verts = output.vertices[0].cpu().numpy()
        faces = model.faces

        # Write to .obj
        fname = os.path.join(output_folder, f'frame_{i:05d}.obj')
        with open(fname, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"[INFO] Saved: {fname}")

    print("[✓] Done.")