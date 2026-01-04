import os
import glob
import subprocess

# Paths
INPUT_DIR = "/mnt/f/BrainTumors/IXI/IXI-T1"
OUTPUT_DIR = "/mnt/f/BrainTumors/IXI/IXI-T1-segmented"
FS_LICENSE = "/mnt/c/Users/gunte/OneDrive/Desktop/Projects/LaserAblation/FastSurfer_run/license.txt"

# Docker image
DOCKER_IMAGE = "deepmi/fastsurfer:gpu-latest"

# Glob all patient NIfTI files
nii_files = glob.glob(os.path.join(INPUT_DIR, "IXI*.nii.gz"))

print(f"Found {len(nii_files)} files")

for nii_file in nii_files:
    base = os.path.basename(nii_file)              # "patient_001_image.nii.gz"
    pid_str, _ = os.path.splitext(base)

    out_folder = os.path.join(OUTPUT_DIR, f"{pid_str}")
    file_name = os.path.join(out_folder, f"patient_{pid_str}", "mri", "aparc.DKTatlas+aseg.deep.mgz")
    print(f"Looking for: {file_name}.")
    if os.path.isfile(file_name):
        print(f"{file_name} exists. Skipping folder: {out_folder}.")
        continue

    os.makedirs(out_folder, exist_ok=True)

    # Docker mount paths (WSL)
    nii_mount = f"{nii_file}:/data/t1.nii.gz"
    out_mount = f"{out_folder}:/data/out"

    print("---------------------------------------------------")
    print(f"Doing: {nii_mount}, {out_mount}")
    print("---------------------------------------------------")

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-u", "$(id -u):$(id -g)",
        "-v", nii_mount,
        "-v", out_mount,
        "-v", f"{FS_LICENSE}:/fs_license.txt",
        DOCKER_IMAGE,
        "--t1", "/data/t1.nii.gz",
        "--sid", f"patient_{pid_str}",
        "--sd", "/data/out",
        "--fs_license", "/fs_license.txt"
    ]

    print(f"Running FastSurfer for patient {pid_str} ...")
    subprocess.run(" ".join(cmd), shell=True)
