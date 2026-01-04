docker run --rm --gpus all -u $(id -u):$(id -g) \
  -v /mnt/f/BrainTumorSegment/nifti/subj001/t1.nii.gz:/data/t1.nii.gz \
  -v /mnt/f/BrainTumorSegment/nifti/subj001_out:/data/out \
  -v /mnt/f/BrainTumorSegment/license.txt:/fs_license.txt \
  deepmi/fastsurfer:gpu-latest \
  --t1 /data/t1.nii.gz \
  --sid subj001 \
  --sd /data/out \
  --fs_license /fs_license.txt


"F:\Brats2020\IXI\IXI-T1\IXI002-Guys-0828-T1.nii.gz"

docker run --rm --gpus all -u $(id -u):$(id -g) \
  -v /mnt/f/BrainTumorSegment/IXI/IXI-T1/IXI002-Guys-0828-T1.nii.gz:/data/t1.nii.gz \
  -v /mnt/f/BrainTumorSegment/IXI/IXI-T1/outputs/IXI002-Guys-0828-T1:/data/out \
  -v /mnt/f/BrainTumorSegment/license.txt:/fs_license.txt \
  deepmi/fastsurfer:gpu-latest \
  --t1 /data/t1.nii.gz \
  --sid Guys-0828 \
  --sd /data/out \
  --fs_license /fs_license.txt


docker run --rm --gpus all -u $(id -u):$(id -g) \
  -v /mnt/f/BrainTumorSegment/preprocessed_nifti/patient_001_image.nii.gz:/data/t1.nii.gz \
  -v /mnt/f/BrainTumorSegment/structure/patient_001:/data/out \
  -v /mnt/f/BrainTumorSegment/license.txt:/fs_license.txt \
  deepmi/fastsurfer:gpu-latest \
  --t1 /data/t1.nii.gz \
  --sid patient_001 \
  --sd /data/out \
  --fs_license /fs_license.txt