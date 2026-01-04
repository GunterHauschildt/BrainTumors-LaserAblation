import torch
import monai
import json
from monai.transforms import MapTransform


class RelabelFastSurferByName(monai.transforms.Transform):
    """
    MONAI transform to remap FastSurfer / FreeSurfer labels using
    a name-based mapping to importance classes.
    """

    def __init__(self, keys, fs_lut_path, mapping_by_name_path):
        """
        Args:
            keys: list of data keys to apply mapping to, e.g. ["label"]
            fs_lut_path: path to FreeSurferColorLUT.txt
            mapping_by_name: dict {region_name: new_class}
                             region_name should be canonical (no lh-/rh-/Left-/Right- prefix)
                             e.g., "superiorfrontal": 1
        """
        self.keys = keys
        with open(mapping_by_name_path, "r") as f:
            self.mapping_by_name = json.load(f)

        # self.mapping_by_name = mapping_by_name
        self.name_by_id, self.id_by_name = self._load_fs_lut(fs_lut_path)
        self.lut = self._build_id_lut()


    def _load_fs_lut(self, path):
        """Load FS LUT, return name_by_id and id_by_name dicts"""
        name_by_id = {}
        id_by_name = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                    raw_name = parts[1]
                except:
                    continue

                # canonicalize: remove hemisphere prefixes, lowercase
                name = raw_name
                for prefix in ("ctx-lh-", "ctx-rh-", "wm-lh-", "wm-rh-", "Left-", "Right-"):
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                name = name.lower()

                name_by_id[idx] = name
                id_by_name.setdefault(name, []).append(idx)

        return name_by_id, id_by_name

    def _build_id_lut(self):
        """Convert name-based mapping to integer LUT"""
        max_id = max(self.name_by_id.keys())
        lut = torch.zeros(max_id + 1, dtype=torch.int16)

        for name, cls in self.mapping_by_name.items():
            if name not in self.id_by_name:
                print(f"Warning: {name} not found in FS LUT")
                continue
            for idx in self.id_by_name[name]:
                # print(f"Info: Found {name} FS LUT")
                lut[idx] = cls
        return lut

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = torch.as_tensor(d[key], dtype=torch.int64)
            # handle rare case: segment IDs larger than LUT
            max_id_in_seg = int(seg.max().item())
            if max_id_in_seg >= len(self.lut):
                pad = torch.zeros(max_id_in_seg + 1 - len(self.lut), dtype=self.lut.dtype)
                self.lut = torch.cat([self.lut, pad], dim=0)
            d[key] = self.lut[seg]
        return d
