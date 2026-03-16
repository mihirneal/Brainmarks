import os
import json
from pathlib import Path

import fsspec

from fmri_fm_eval.datasets.base import HFDataset, load_arrow_dataset
from fmri_fm_eval.datasets.registry import register_dataset

AABC_REMOTE_ROOT = "s3://medarc/fmri-datasets/eval"
AABC_ROOT = os.getenv("AABC_ROOT", AABC_REMOTE_ROOT)
AABC_LOCAL_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "AABC"

AABC_TARGET_MAP_DICT = {
    # Demographics
    "sex": "aabc_target_map_sex.json",
    "age": "aabc_target_map_age_open.json",
    # NEO-FFI Personality
    "neo_n": "aabc_target_map_neo_n.json",
    "neo_e": "aabc_target_map_neo_e.json",
    "neo_o": "aabc_target_map_neo_o.json",
    "neo_a": "aabc_target_map_neo_a.json",
    "neo_c": "aabc_target_map_neo_c.json",
    # Cognitive Composites
    "fluid_iq": "aabc_target_map_FluidIQ_Tr35_60y.json",
    "cryst_iq": "aabc_target_map_CrystIQ_Tr35_60y.json",
    "memory": "aabc_target_map_Memory_Tr35_60y.json",
}

AABC_TASK5_TRIAL_TYPES = {
    "go_hit": 0,
    "nogo_correct": 1,
    "encoding": 2,
    "recall": 3,
    "vismotor": 4,
}


def _create_aabc(space: str, target: str, **kwargs):
    target_key = "sub"
    target_map_path = _resolve_aabc_target_path(AABC_TARGET_MAP_DICT[target])

    with fsspec.open(target_map_path, "r") as f:
        target_map = json.load(f)

    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{_resolve_aabc_dataset_dir('aabc', space)}/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset = HFDataset(dataset, target_key=target_key, target_map=target_map)
        dataset_dict[split] = dataset

    return dataset_dict


def _create_aabc_task(name: str, space: str, **kwargs):
    dataset_dict = {}
    splits = ["train", "validation", "test"]
    dataset_dir = _resolve_aabc_dataset_dir(f"aabc-{name}", space)
    for split in splits:
        url = f"{dataset_dir}/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset = HFDataset(dataset, target_key="cond_id")
        dataset_dict[split] = dataset

    return dataset_dict


def _resolve_aabc_target_path(filename: str) -> str:
    env_root = os.getenv("AABC_ROOT")
    if env_root:
        if "://" not in env_root:
            root = Path(env_root)
            for candidate in (
                root / "targets" / filename,
                root / "metadata" / "targets" / filename,
            ):
                if candidate.exists():
                    return str(candidate)
        return f"{env_root}/targets/{filename}"

    local_target_path = AABC_LOCAL_ROOT / "metadata" / "targets" / filename
    if local_target_path.exists():
        return str(local_target_path)
    return f"{AABC_REMOTE_ROOT}/targets/{filename}"


def _resolve_aabc_dataset_dir(name: str, space: str) -> str:
    dataset_dir = f"{name}.{space}.arrow"

    env_root = os.getenv("AABC_ROOT")
    if env_root:
        if "://" not in env_root:
            root = Path(env_root)
            for candidate in (root / dataset_dir, root / "data" / "processed" / dataset_dir):
                if candidate.exists():
                    return str(candidate)
        return f"{env_root}/{dataset_dir}"

    local_dataset_dir = AABC_LOCAL_ROOT / "data" / "processed" / dataset_dir
    if local_dataset_dir.exists():
        return str(local_dataset_dir)
    return f"{AABC_REMOTE_ROOT}/{dataset_dir}"


# Demographics
@register_dataset
def aabc_sex(space: str, **kwargs):
    return _create_aabc(space, target="sex", **kwargs)


@register_dataset
def aabc_age(space: str, **kwargs):
    return _create_aabc(space, target="age", **kwargs)


# NEO-FFI Personality
@register_dataset
def aabc_neo_n(space: str, **kwargs):
    return _create_aabc(space, target="neo_n", **kwargs)


@register_dataset
def aabc_neo_e(space: str, **kwargs):
    return _create_aabc(space, target="neo_e", **kwargs)


@register_dataset
def aabc_neo_o(space: str, **kwargs):
    return _create_aabc(space, target="neo_o", **kwargs)


@register_dataset
def aabc_neo_a(space: str, **kwargs):
    return _create_aabc(space, target="neo_a", **kwargs)


@register_dataset
def aabc_neo_c(space: str, **kwargs):
    return _create_aabc(space, target="neo_c", **kwargs)


# Cognitive Composites
@register_dataset
def aabc_fluid_iq(space: str, **kwargs):
    return _create_aabc(space, target="fluid_iq", **kwargs)


@register_dataset
def aabc_cryst_iq(space: str, **kwargs):
    return _create_aabc(space, target="cryst_iq", **kwargs)


@register_dataset
def aabc_memory(space: str, **kwargs):
    return _create_aabc(space, target="memory", **kwargs)


@register_dataset
def aabc_task5(space: str, **kwargs):
    return _create_aabc_task("task5", space, **kwargs)
