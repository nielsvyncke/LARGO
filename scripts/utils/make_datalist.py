"""
This script processes the BraTS 2018 dataset to generate a datalist JSON file in a Decathlon-style format.

Usage:
    $ python -m scripts.utils.make_datalist --data_dir /path/to/brats2018 --save_path configs/datalist.json

The data directory must contain a ``training`` folder with one sub-folder per case
(``Brats18_<id>/Brats18_<id>_{flair,t1,t1ce,t2,seg}.nii[.gz]``).

Arguments:
    - data_dir: Path to the dataset directory (required).
    - save_path: Path to save the datalist JSON file (default: '../configs/datalist.json').
    - num_bins: Number of bins for stratified K-fold (default: 5).
    - num_folds: Number of folds for stratified K-fold (default: 5).

"""

import argparse
import glob
import os
import json

import numpy as np
import torch
from monai.data import Dataset
from monai import transforms
from sklearn.model_selection import StratifiedKFold


def make_datalist():
    """Makes the train datalist for BraTS 2018."""
    train_folders = sorted(glob.glob("Brats18_*"))

    # Make training set
    train_datalist = []
    for folder in train_folders:
        id_ = os.path.basename(folder)
        # BraTS 2018 naming: flair, t1, t1ce, t2
        flair = glob.glob(os.path.join(folder, "*_flair.nii*"))[0]
        t1 = glob.glob(os.path.join(folder, "*_t1.nii*"))[0]
        t1ce = glob.glob(os.path.join(folder, "*_t1ce.nii*"))[0]
        t2 = glob.glob(os.path.join(folder, "*_t2.nii*"))[0]
        seg = glob.glob(os.path.join(folder, "*_seg.nii*"))[0]
        # Order: FLAIR, T1, T1ce, T2 (matching BraTS convention)
        sample = {"id": id_, "image": [flair, t1, t1ce, t2], "label": seg}
        train_datalist.append(sample)

    return train_datalist


def lesion_volume(mask):
    """Computes lesion volume from a segmentation mask."""
    # For BraTS 2018, enhancing tumor is class 4
    lesion_num_voxels = torch.sum(mask == 4)
    voxel_size = np.prod(mask.meta["pixdim"][1:4])
    lesion_vol = lesion_num_voxels * voxel_size
    return lesion_vol.item()


def add_volume(datalist):
    """Adds lesion volumes to the given datalist."""
    transform = transforms.Compose(
        [
            transforms.CopyItemsd(keys="label", names="volume"),
            transforms.LoadImaged(keys="volume"),
            transforms.Lambdad(keys="volume", func=lesion_volume),
        ]
    )

    dataset = Dataset(datalist, transform)
    updated_datalist = list(dataset)

    return updated_datalist


def stratified_kfold_with_mod(datalist, num_bins=5, num_folds=5):
    """
    Makes a stratified k-fold cross-validation based on lesion volume
    and assigns 'mod' field (0-14) uniformly distributed across all folds.
    """

    # Quantize lesion volumes
    lesion_volumes = [x["volume"] for x in datalist]
    _, bins = np.histogram(lesion_volumes, bins=num_bins)
    y = np.digitize(lesion_volumes, bins[:-1])

    # Stratified K-fold based on quantized lesion volumes
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    skf.get_n_splits(None, y)

    updated_datalist = []
    for fold, (_, val_idx) in enumerate(skf.split(datalist, y)):
        for j in val_idx:
            sample = datalist[j]
            del sample["volume"]
            updated_datalist.extend([{**sample, "fold": fold}])

    updated_datalist = sorted(updated_datalist, key=lambda x: x["id"])

    # Assign 'mod' field (0-14) uniformly across the dataset
    # 15 modality combinations: 0-14
    num_samples = len(updated_datalist)
    num_mods = 15
    
    # Create a list with uniform distribution of mod values
    mods = []
    samples_per_mod = num_samples // num_mods
    remainder = num_samples % num_mods
    
    for mod in range(num_mods):
        count = samples_per_mod + (1 if mod < remainder else 0)
        mods.extend([mod] * count)
    
    # Shuffle to distribute uniformly across folds
    np.random.seed(42)
    np.random.shuffle(mods)
    
    # Assign mod values
    for i, sample in enumerate(updated_datalist):
        sample["mod"] = mods[i]
    
    # Verify distribution
    print("\n=== Mod Distribution ===")
    print(f"Total samples: {num_samples}")
    for mod in range(num_mods):
        count = sum(1 for s in updated_datalist if s["mod"] == mod)
        print(f"Mod {mod}: {count} samples")
    
    print("\n=== Mod Distribution per Fold ===")
    for fold in range(num_folds):
        fold_samples = [s for s in updated_datalist if s["fold"] == fold]
        print(f"\nFold {fold} ({len(fold_samples)} samples):")
        for mod in range(num_mods):
            count = sum(1 for s in fold_samples if s["mod"] == mod)
            print(f"  Mod {mod}: {count} samples")

    return updated_datalist


def export_datalist(train_list, file_path):
    """Export the dataset splits to a JSON file."""
    data_list = {"training": train_list}
    with open(file_path, "w") as f:
        json.dump(data_list, f, indent=2)
    print(f"\nDatalist saved to {file_path}.")


def main(args):
    # Set working directory to data directory/training
    os.chdir(os.path.join(args.data_dir, 'training'))

    # Create datalists for BraTS 2018 dataset
    train_list = make_datalist()
    print(f"Found {len(train_list)} training samples")

    # Add lesion volumes to training list
    print("Computing lesion volumes...")
    train_list = add_volume(train_list)

    # Apply Stratified K-Fold to training list with mod assignment
    print("Creating stratified K-fold splits with uniform mod distribution...")
    train_list = stratified_kfold_with_mod(
        train_list, num_bins=args.num_bins, num_folds=args.num_folds
    )

    # Define path to save datalist
    if args.save_path is None:
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "datalist.json"
        )
    else:
        file_path = args.save_path
    file_path = os.path.abspath(file_path)

    # Export datalist to JSON
    export_datalist(train_list, file_path)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare datalist for BraTS 2018 dataset.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save the datalist JSON file"
    )
    parser.add_argument(
        "--num_bins", type=int, default=5, help="Number of bins for stratified K-fold"
    )
    parser.add_argument(
        "--num_folds", type=int, default=5, help="Number of folds for stratified K-fold"
    )
    args = parser.parse_args()
    main(args)
