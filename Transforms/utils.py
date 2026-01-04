from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    z: int
    def as_tuple(self): return (self.x, self.y, self.z)


@dataclass(frozen=True)
class BoundingBox3D:
    p0: Point
    p1: Point

    def as_slices(self):
        return (slice(self.p0.z, self.p1.z),
                slice(self.p0.y, self.p1.y),
                slice(self.p0.x, self.p1.x))

    def shape(self):
        return (self.p1.z - self.p0.z,
                self.p1.y - self.p0.y,
                self.p1.x - self.p0.x)


@dataclass
class TumorSamplePaths:
    json_file: Path
    mask_np: Path
    image_np: Path


@dataclass(frozen=True)
class TumorInfo:
    patient_name: str
    zooms: tuple[float, float, float]
    affine: np.ndarray
    bbox: BoundingBox3D

    def save(self, out_dir: Path):
        out_dir.mkdir(exist_ok=True)

        # JSON-safe metadata
        meta = {
            "name": self.patient_name,
            "zooms": [float(z) for z in self.zooms],
            "bbox": {
                "p0": self.bbox.p0.as_tuple(),
                "p1": self.bbox.p1.as_tuple(),
            },
            "affine": self.affine.astype(float).tolist()
        }

        with open(out_dir / "tumor_info.json", "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(path: Path) -> "TumorInfo":
        with open(path) as f:
            meta = json.load(f)

        bbox = BoundingBox3D(
            p0=Point(*meta["bbox"]["p0"]),
            p1=Point(*meta["bbox"]["p1"]),
        )

        affine = np.array(meta["affine"], dtype=np.float32)

        return TumorInfo(
            meta["name"],
            tuple(meta["zooms"]),
            affine,
            bbox
        )

