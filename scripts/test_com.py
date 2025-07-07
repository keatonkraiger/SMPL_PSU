import numpy as np
from scipy.io import loadmat
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# -------- Load Data --------
npzs_dir = '/mnt/e/Research/SMPL/SMPL_TMM/output/Subject1/Take_1/global/V1/npz'
npz_files = sorted(glob(f"{npzs_dir}/*.npz"))

npz_coms = []
for i, npz_file in enumerate(npz_files):
    data = np.load(npz_file)
    com = data['com']
    com = np.array((com[0], -com[2], com[1])) * 10e2  # Convert to mm
    npz_coms.append(com)

mat_coms = loadmat('/mnt/d/Data/PSU100/Subject_wise/Subject1/CoM_1.mat')["CoM"]
mat_coms = mat_coms[::5]

# -------- Align and Compute Differences --------
npz_arr = []
mat_arr = []
diffs = []

for i in range(len(npz_coms)):
    if mat_coms[i, -1] == 0:
        continue
    npz_arr.append(npz_coms[i])
    mat_arr.append(mat_coms[i, :3])
    diffs.append(npz_coms[i] - mat_coms[i, :3])

npz_arr = np.array(npz_arr)
mat_arr = np.array(mat_arr)
diffs = np.array(diffs)

# -------- Setup Animation --------
xlims = (npz_arr[:, 0].min() - 100, npz_arr[:, 0].max() + 100)
ylims = (npz_arr[:, 1].min() - 100, npz_arr[:, 1].max() + 100)
zlims = (npz_arr[:, 2].min() - 100, npz_arr[:, 2].max() + 100)
          
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])  # More space to 3D

ax3d = fig.add_subplot(gs[0], projection='3d')
ax2d = fig.add_subplot(gs[1])

frames = np.arange(len(diffs))
ax2d.plot(frames, diffs[:, 0], label='X diff')
ax2d.plot(frames, diffs[:, 1], label='Y diff')
ax2d.plot(frames, diffs[:, 2], label='Z diff')
timeline = ax2d.axvline(0, color='k', linestyle='--')

ax2d.set_xlabel("Frame")
ax2d.set_ylabel("Difference (mm)")
ax2d.legend(loc='upper right')  # 🔧 Fix the legend position here
ax2d.grid()

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

    # 🔁 Scroll bottom plot to follow
    window = 100
    start = max(0, frame - window)
    end = min(len(diffs), frame + window)
    ax2d.set_xlim(start, end)
    timeline.set_xdata([frame])

    return ax3d, timeline

max_frames = len(diffs)
ani = FuncAnimation(fig, update, frames=max_frames, interval=50)

# -------- Save to File --------
out_path = "com_difference_animation.mp4"
ani.save(out_path, fps=20, dpi=150)
print(f"Animation saved to: {out_path}")
