# extract_voxelwise_timeseries_denoise
# -------------------------------
# Script 1: Manual confound regression & z-scoring
# -------------------------------

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

# === Paths ===
func_path = Path('/BICNAS2/group-northoff/rsfMRI-FEOBV/sub-002/ses-002/func/sub-002_ses-002_task-rest_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
confound_path = Path('/BICNAS2/group-northoff/rsfMRI-FEOBV/sub-002/ses-002/func/sub-002_ses-002_task-rest_run-02_desc-confounds_regressors.tsv')
mask_dir = Path('./ROIs_NII')
output_dir = Path('./voxelwise_timeseries_manual')
output_dir.mkdir(parents=True, exist_ok=True)

# === Load functional data ===
print("Loading functional data...")
func_img = nib.load(str(func_path))
func_data = func_img.get_fdata()  # shape: (X, Y, Z, T)
n_timepoints = func_data.shape[3]
print(f"Functional data shape: {func_data.shape}")

# === Load confounds ===
print("Loading confounds...")
confounds_df = pd.read_csv(confound_path, sep='\t')

# Choose confounds: modify based on rationale or recommendations
confound_columns = [
    'trans_x', 'trans_y', 'trans_z',  # motion (translation)
    'rot_x', 'rot_y', 'rot_z',        # motion (rotation)
    'csf', 'white_matter', 'global_signal'  # signal components
]

# Only keep columns that exist
confound_columns = [col for col in confound_columns if col in confounds_df.columns]
confounds = confounds_df[confound_columns].fillna(0).values

# Center confounds (optional but helps regression stability)
confounds -= confounds.mean(axis=0)

print(f"Using confounds: {confound_columns}")

# === Load masks by layer ===
layers = {
    'Exteroception': list(mask_dir.glob('Exteroception_*.nii.gz')),
    'Interoception': list(mask_dir.glob('Interoception_*.nii.gz')),
    'Cognition': list(mask_dir.glob('Cognition_*.nii.gz'))
}

for layer, masks in layers.items():
    print(f"{layer}: {len(masks)} masks found")

# === Denoising function ===
def regress_out_confounds(timeseries, confounds):
    """
    Regress out confounds from voxelwise time series.
    timeseries: shape (T, n_voxels)
    confounds: shape (T, n_confounds)
    """
    model = LinearRegression()
    model.fit(confounds, timeseries)
    predicted = model.predict(confounds)
    residuals = timeseries - predicted
    return residuals

# === Process each mask ===
for layer, mask_paths in layers.items():
    print(f"\nProcessing layer: {layer}")
    (layer_output := output_dir / layer).mkdir(parents=True, exist_ok=True)

    for mask_path in mask_paths:
        mask_name = mask_path.stem
        print(f"  Mask: {mask_name}")

        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata().astype(bool)

        # Check shape
        if mask_data.shape != func_data.shape[:3]:
            raise ValueError(f"Shape mismatch: mask {mask_name} and functional data")

        # Extract voxel time series: shape (n_voxels, T) → transpose to (T, n_voxels)
        masked_voxels = func_data[mask_data].T

        # Regress out confounds
        denoised_ts = regress_out_confounds(masked_voxels, confounds)

        # Z-score across time (axis 0)
        zscored_ts = zscore(denoised_ts, axis=0)

        # Save
        save_path = layer_output / f"{func_path.stem}_{mask_name}_voxelwise_timeseries.npy"
        np.save(save_path, zscored_ts)
        print(f"    Saved: {save_path.name}")

print("\n[Done] All voxelwise time series saved.")
