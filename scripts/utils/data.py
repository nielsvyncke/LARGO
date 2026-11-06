from collections.abc import Hashable, Mapping

import numpy as np
import torch
from monai.data import load_decathlon_datalist
from monai.transforms import Transform, MapTransform
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.enums import TransformBackends


def load_datalist(datalist_path, data_dir=None, key="training", fold=None, section=None):
    data_list = load_decathlon_datalist(
        datalist_path, data_list_key=key, base_dir=data_dir
    )
    if fold is None:
        return data_list
    elif section in ("training", "train"):
        return [x for x in data_list if x["fold"] != fold]
    elif section in ("validation", "val"):
        return [x for x in data_list if x["fold"] == fold]
    else:
        raise ValueError(
            "When `fold` is provided, `section` must be one of ['training', 'validation']."
        )


class BraTSOneHotEncoder(Transform):
    """
    Convert a BraTS label map to the three nested tumor regions used for evaluation:

    input classes:  1 = necrotic/non-enhancing tumor core (NCR/NET), 2 = edema (ED), 4 = enhancing tumor (ET)
    output channels: 0 = ET, 1 = tumor core (ET + NCR/NET), 2 = whole tumor (ET + NCR/NET + ED)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        ncr, ed, et = 1, 2, 4
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = [
            (img == et),
            (img == et) | (img == ncr),
            (img == et) | (img == ncr) | (img == ed),
        ]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class BraTSOneHotEncoderd(MapTransform):
    """Dictionary-based wrapper of `BraTSOneHotEncoder`."""

    backend = BraTSOneHotEncoder.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = BraTSOneHotEncoder()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def make_inference_datalist(input_dir, modalities, suffixes=("flair", "t1", "t1ce", "t2")):
    """List the cases of ``input_dir`` for inference with the modality subset ``modalities``.

    ``input_dir`` contains one folder per case with files ``<case>_<suffix>.nii[.gz]``. Only the
    files of the selected modalities (indices into ``suffixes``) must exist. Cases that miss one of
    them are skipped with a warning.
    """
    import glob
    import os
    import warnings

    datalist = []
    for case_dir in sorted(d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)):
        case = os.path.basename(case_dir)
        images = []
        for i in modalities:
            matches = sorted(glob.glob(os.path.join(case_dir, f"*_{suffixes[i]}.nii*")))
            if not matches:
                warnings.warn(f"{case}: no *_{suffixes[i]}.nii file found, skipping this case.")
                images = None
                break
            images.append(matches[0])
        if images:
            datalist.append({"id": case, "image": images})
    if not datalist:
        raise FileNotFoundError(f"No cases with the requested modalities found in {input_dir}.")
    return datalist
