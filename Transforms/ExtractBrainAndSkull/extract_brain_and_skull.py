import numpy as np
from monai.transforms import Transform
from scipy.ndimage import binary_fill_holes
import re
from scipy.ndimage import binary_closing, generate_binary_structure


def generate_brain_label_ids(fs_lut_path):
    """
    Parse FreeSurferColorLUT.txt and return a set of label IDs that belong to brain tissue.
    """

    brain_ids = set()

    # Anything matching these patterns is considered brain
    include_patterns = [
        r"^ctx-",  # cortical parcellations
        r"^wm-", r"^WM-",  # white matter parcels
        r"cerebellum",
        r"thalamus",
        r"caudate",
        r"putamen",
        r"pallidum",
        r"hippocampus",
        r"amygdala",
        r"accumbens",
        r"ventraldc",
        r"brain-stem",
        r"cerebral-white-matter",
        r"cerebral-cortex",
        r"ventricle",  # automatically include all ventricles
        r"csf",  # include CSF as part of brain
        r"superior",
        r"inferior"


    ]

    # Anything matching these are explicitly NOT brain
    exclude_patterns = [
        r"skull",
        r"scalp",
        r"soft-tissue",
        r"background",
        r"air",
        r"unknown",
    ]

    include_re = re.compile("|".join(include_patterns), re.IGNORECASE)
    exclude_re = re.compile("|".join(exclude_patterns), re.IGNORECASE)

    with open(fs_lut_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or line.startswith("{{{") or line.startswith("}}}"):
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue

            try:
                label_id = int(parts[0])
            except ValueError:
                continue

            name = parts[1]

            lname = name.lower()

            if include_re.search(lname) and not exclude_re.search(lname):
                brain_ids.add(label_id)

    return brain_ids


class ExtractBrainAndSkull(Transform):

    def __init__(self, fs_lut_path):
        super().__init__()
        self._brain_ids = generate_brain_label_ids(fs_lut_path)

    def __call__(self, data):
        d = dict(data)

        # Brain: label voxel is in brain_label_ids
        brain = np.isin(d["labels"], list(self._brain_ids))
        brain = binary_closing(brain, structure=np.ones((5, 5, 5)))
        brain = binary_fill_holes(brain)
        d["brain"] = brain.astype(np.uint8)

        # to do: skull

        return d
