"""Create an HCPYA-sized balanced AABC task dataset in Arrow format.

The dataset mirrors the HCP-YA task Arrow schema but uses a coarse 5-class
taxonomy for AABC task fMRI:

- go_hit
- nogo_correct
- encoding
- recall
- vismotor

This builder is intentionally closer to `hcpya-task21` than to the earlier
subject-unique AABC task dataset:

- V1 only
- subject-disjoint train/validation/test splits defined by subject batches
- multiple clips per subject are allowed within a split
- one clip per CARIT event
- multiple clips per long FACENAME / VISMOTOR block

The default split counts are the closest 5-way balanced match to the saved
row counts of `hcpya-task21.flat.arrow`:

- train: 19,000 total = 3,800 per class
- validation: 4,030 total = 806 per class
- test: 5,040 total = 1,008 per class
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets as hfds
import numpy as np
from nibabel.nifti2 import Nifti2Header

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers

hfds.config.DEFAULT_MAX_BATCH_SIZE = 256

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
DEFAULT_RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"

AABC_TR = 0.72
TARGET_TR = 1.0
INTERPOLATION = "pchip"
NUM_FRAMES = 16

DEFAULT_SPLIT_BATCHES = {
    "train": [0, 1, 2, 3, 7, 10, 11, 12, 13, 14],
    "validation": [15, 16],
    "test": [17, 18, 19],
}
DEFAULT_SPLIT_COUNTS = {
    "train": 3800,
    "validation": 806,
    "test": 1008,
}
WINDOWS_PER_EVENT = {
    "go_hit": 1,
    "nogo_correct": 1,
    "encoding": 2,
    "recall": 3,
    "vismotor": 3,
}

AABC_TASK5_CONDITIONS = {
    "go_hit": {
        "task": "CARIT",
        "task_dir": "tfMRI_CARIT_PA",
        "event_file": "goHit.txt",
        "cond_id": 0,
    },
    "nogo_correct": {
        "task": "CARIT",
        "task_dir": "tfMRI_CARIT_PA",
        "event_file": "nogoCR.txt",
        "cond_id": 1,
    },
    "encoding": {
        "task": "FACENAME",
        "task_dir": "tfMRI_FACENAME_PA",
        "event_file": "encoding.txt",
        "cond_id": 2,
    },
    "recall": {
        "task": "FACENAME",
        "task_dir": "tfMRI_FACENAME_PA",
        "event_file": "recall.txt",
        "cond_id": 3,
    },
    "vismotor": {
        "task": "VISMOTOR",
        "task_dir": "tfMRI_VISMOTOR_PA",
        "event_file": "vismotor.txt",
        "cond_id": 4,
    },
}


def main(args):
    raw_root = resolve_raw_root(args.root)
    out_root = Path(args.out_root or PROCESSED_ROOT)
    outdir = out_root / f"aabc-task5.{args.space}.arrow"

    _logger.info("Generating dataset: %s", outdir)
    _logger.info("Using raw root: %s", raw_root)
    if outdir.exists():
        if not args.overwrite:
            _logger.warning("Output %s exists; exiting.", outdir)
            return 1
        _logger.info("Removing existing output: %s", outdir)
        shutil.rmtree(outdir)

    with (ROOT / "metadata/aabc_subject_batch_splits.json").open() as f:
        sub_batch_splits = json.load(f)

    split_batch_ids = resolve_split_batch_ids(args)
    split_subjects = {
        split: sorted(
            {
                sub
                for batch_id in batch_ids
                for sub in sub_batch_splits[f"batch-{batch_id:02d}"]
            }
        )
        for split, batch_ids in split_batch_ids.items()
    }

    for split, subjects in split_subjects.items():
        _logger.info(
            "Subject pool (%s) from batches %s: %d",
            split,
            split_batch_ids[split],
            len(subjects),
        )

    split_condition_pools = {}
    for split, subjects in split_subjects.items():
        pools, stats = build_split_condition_pools(
            raw_root,
            subjects,
            args.space,
            candidate_workers=args.candidate_workers,
        )
        split_condition_pools[split] = pools
        _logger.info(
            "%s V1 subjects: present=%d complete=%d incomplete=%d",
            split,
            stats["present"],
            stats["complete"],
            stats["incomplete"],
        )
        for cond, pool in pools.items():
            _logger.info("  %s candidates (%s): %d", split, cond, len(pool))

    split_counts = resolve_split_counts(args)
    _logger.info("Target per-class split counts: %s", split_counts)

    rng = np.random.default_rng(args.seed)
    sample_splits = select_balanced_samples(
        split_condition_pools=split_condition_pools,
        split_counts=split_counts,
        rng=rng,
    )

    for split, samples in sample_splits.items():
        cond_counts = {}
        for sample in samples:
            cond = sample["cond"]
            cond_counts[cond] = cond_counts.get(cond, 0) + 1
        _logger.info("Num samples (%s): %d", split, len(samples))
        _logger.info("  %s condition breakdown: %s", split, cond_counts)
        _logger.info(
            "  %s unique subjects: %d", split, len({sample['sub'] for sample in samples})
        )

    grouped_sample_splits = {
        split: group_samples_by_path(samples) for split, samples in sample_splits.items()
    }
    for split, sample_groups in grouped_sample_splits.items():
        _logger.info("Grouped sample paths (%s): %d", split, len(sample_groups))

    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]
    _logger.info("Using reader for space '%s' with dimension: %d", args.space, dim)

    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "task": hfds.Value("string"),
            "cond": hfds.Value("string"),
            "cond_id": hfds.Value("int32"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "n_frames": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    writer_batch_size = args.writer_batch_size
    if writer_batch_size is None:
        if args.space == "flat":
            writer_batch_size = 16
        elif args.space in {"mni", "mni_cortex"}:
            writer_batch_size = 8

    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, sample_groups in grouped_sample_splits.items():
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={
                    "sample_groups": sample_groups,
                    "root": raw_root,
                    "reader": reader,
                    "dim": dim,
                    "sample_workers": args.sample_workers,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                writer_batch_size=writer_batch_size,
                fingerprint=(
                    "aabc-task5-"
                    f"{args.space}-"
                    f"{','.join(str(b) for b in split_batch_ids['train'])}-"
                    f"{','.join(str(b) for b in split_batch_ids['validation'])}-"
                    f"{','.join(str(b) for b in split_batch_ids['test'])}-"
                    f"{split_counts['train']}-{split_counts['validation']}-{split_counts['test']}-"
                    f"{args.seed}-{split}"
                ),
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")

    _logger.info("Dataset saved to: %s", outdir)
    return 0


def resolve_raw_root(root_arg: str | None) -> Path:
    root = root_arg or os.getenv("AABC_RAW_ROOT")
    root = Path(root) if root is not None else DEFAULT_RAW_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"AABC raw root {root} does not exist. Pass --root or set AABC_RAW_ROOT."
        )
    return root


def resolve_split_batch_ids(args) -> dict[str, list[int]]:
    return {
        "train": args.train_batch_ids or DEFAULT_SPLIT_BATCHES["train"],
        "validation": args.validation_batch_ids or DEFAULT_SPLIT_BATCHES["validation"],
        "test": args.test_batch_ids or DEFAULT_SPLIT_BATCHES["test"],
    }


def build_split_condition_pools(
    root: Path,
    subjects: list[str],
    space: str,
    *,
    candidate_workers: int,
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    pools = {cond: [] for cond in AABC_TASK5_CONDITIONS}
    stats = {"present": 0, "complete": 0, "incomplete": 0}

    def scan_subject(sub: str) -> tuple[str, dict[str, list[dict]]]:
        subject_dir = root / f"{sub}_V1_MR"
        if not subject_dir.is_dir():
            return "missing", {}

        candidates = collect_v1_candidates(root, sub, space)
        if len(candidates) != len(AABC_TASK5_CONDITIONS):
            return "incomplete", {}

        return "complete", candidates

    if candidate_workers <= 1:
        results = map(scan_subject, subjects)
    else:
        with ThreadPoolExecutor(max_workers=candidate_workers) as executor:
            results = executor.map(scan_subject, subjects)

    for status, candidates in results:
        if status == "missing":
            continue
        stats["present"] += 1
        if status != "complete":
            stats["incomplete"] += 1
            continue
        stats["complete"] += 1
        for cond, cond_samples in candidates.items():
            pools[cond].extend(cond_samples)

    return pools, stats


def collect_v1_candidates(root: Path, sub: str, space: str) -> dict[str, list[dict]]:
    visit = "V1"
    candidates = {}

    for cond, spec in AABC_TASK5_CONDITIONS.items():
        task_dir = spec["task_dir"]
        suffix = get_task_suffix(space, task_dir)
        relpath = f"{sub}_{visit}_MR/MNINonLinear/Results/{task_dir}/{suffix}"
        fullpath = root / relpath
        if not fullpath.exists():
            continue

        ev_path = (
            root
            / f"{sub}_{visit}_MR/MNINonLinear/Results/{task_dir}/EVs/{spec['event_file']}"
        )
        if not ev_path.exists():
            continue

        events = load_events(ev_path)
        if not events:
            continue

        resampled_length = get_resampled_length(fullpath, space)
        cond_samples = build_event_samples(
            sub=sub,
            task=spec["task"],
            cond=cond,
            cond_id=spec["cond_id"],
            relpath=relpath,
            events=events,
            resampled_length=resampled_length,
        )
        if not cond_samples:
            continue

        candidates[cond] = cond_samples

    return candidates


def get_task_suffix(space: str, task_dir: str) -> str:
    if space in readers.VOLUME_SPACES:
        return f"{task_dir}_hp0_clean_rclean_tclean.nii.gz"
    return f"{task_dir}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii"


def load_events(path: Path) -> list[dict[str, float]]:
    events = []
    seen = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        onset = float(parts[0])
        duration = float(parts[1])
        if onset < 0 or duration <= 0:
            continue

        weight = float(parts[2]) if len(parts) >= 3 else 1.0
        key = (round(onset, 3), round(duration, 3), round(weight, 3))
        if key in seen:
            continue
        seen.add(key)

        events.append({"onset": onset, "duration": duration, "weight": weight})

    return events


def get_resampled_length(path: Path, space: str) -> int:
    if space in readers.VOLUME_SPACES:
        import nibabel as nib

        img = nib.load(path)
        n_vols = img.shape[-1]
    else:
        with path.open("rb") as f:
            hdr = Nifti2Header.from_fileobj(f)
        n_vols = hdr.get_data_shape()[-2]
    return compute_resampled_length(n_vols, tr=AABC_TR, new_tr=TARGET_TR)


def compute_resampled_length(n_vols: int, *, tr: float, new_tr: float) -> int:
    if tr == new_tr:
        return n_vols

    new_length = int(tr * n_vols / new_tr)
    last_time = tr * (n_vols - 1)
    while new_length > 0 and (new_tr * (new_length - 1)) > last_time:
        new_length -= 1
    return new_length


def build_event_samples(
    *,
    sub: str,
    task: str,
    cond: str,
    cond_id: int,
    relpath: str,
    events: list[dict[str, float]],
    resampled_length: int,
) -> list[dict]:
    max_start = resampled_length - NUM_FRAMES
    if max_start < 0:
        return []

    windows_per_event = WINDOWS_PER_EVENT[cond]
    seen_starts = set()
    samples = []

    for event in events:
        starts = compute_clip_starts(
            onset=event["onset"],
            duration=event["duration"],
            windows_per_event=windows_per_event,
            max_start=max_start,
        )
        for start in starts:
            if start in seen_starts:
                continue
            seen_starts.add(start)
            samples.append(
                {
                    "sub": sub,
                    "task": task,
                    "cond": cond,
                    "cond_id": cond_id,
                    "path": relpath,
                    "start": start,
                    "end": start + NUM_FRAMES,
                }
            )

    return samples


def compute_clip_starts(
    *,
    onset: float,
    duration: float,
    windows_per_event: int,
    max_start: int,
) -> list[int]:
    max_offset = max(duration - (NUM_FRAMES * TARGET_TR), 0.0)
    offsets = np.linspace(0.0, max_offset, num=windows_per_event)
    starts = []
    for offset in offsets:
        start = int((onset + float(offset)) / TARGET_TR)
        if 0 <= start <= max_start:
            starts.append(start)
    return sorted(set(starts))


def resolve_split_counts(args) -> dict[str, int]:
    split_counts = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }
    if any(count <= 0 for count in split_counts.values()):
        raise ValueError("Per-class split counts must be positive")
    return split_counts


def select_balanced_samples(
    *,
    split_condition_pools: dict[str, dict[str, list[dict]]],
    split_counts: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, list[dict]]:
    sample_splits = {}
    for split, cond_pools in split_condition_pools.items():
        split_samples = []
        need = split_counts[split]
        for cond in AABC_TASK5_CONDITIONS:
            pool = list(cond_pools[cond])
            if len(pool) < need:
                raise ValueError(
                    f"{split}:{cond} only has {len(pool)} candidate clips; need {need}"
                )
            rng.shuffle(pool)
            split_samples.extend(pool[:need])
        rng.shuffle(split_samples)
        sample_splits[split] = split_samples
    return sample_splits


def group_samples_by_path(samples: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["path"]].append(sample)

    sample_groups = []
    for path, group_samples in grouped.items():
        group_samples = sorted(group_samples, key=lambda sample: (sample["start"], sample["cond"]))
        sample_groups.append({"path": path, "samples": group_samples})
    sample_groups.sort(key=lambda group: group["path"])
    return sample_groups


def generate_samples(
    sample_groups: list[dict],
    *,
    root: Path,
    reader,
    dim: int,
    sample_workers: int,
):
    if sample_workers <= 1:
        for sample_group in sample_groups:
            for record in process_sample_group(
                sample_group=sample_group,
                root=root,
                reader=reader,
                dim=dim,
            ):
                yield record
        return

    with ThreadPoolExecutor(max_workers=sample_workers) as executor:
        for records in executor.map(
            lambda sample_group: process_sample_group(
                sample_group=sample_group,
                root=root,
                reader=reader,
                dim=dim,
            ),
            sample_groups,
        ):
            for record in records:
                yield record


def process_sample_group(
    *,
    sample_group: dict,
    root: Path,
    reader,
    dim: int,
) -> list[dict]:
    fullpath = root / sample_group["path"]
    series = reader(fullpath)
    _, data_dim = series.shape
    assert data_dim == dim, f"Expected dim {dim}, got {data_dim} for {sample_group['path']}"

    series, mean, std = nisc.scale(series)
    series = nisc.resample_timeseries(
        series,
        tr=AABC_TR,
        new_tr=TARGET_TR,
        kind=INTERPOLATION,
    )

    records = []
    for sample in sample_group["samples"]:
        start = sample["start"]
        end = sample["end"]
        if end > len(series):
            _logger.warning(
                "Selected clip exceeds series length for %s (%d > %d); skipping.",
                sample["path"],
                end,
                len(series),
            )
            continue

        clip = series[start:end]
        records.append(
            {
                "sub": sample["sub"],
                "task": sample["task"],
                "cond": sample["cond"],
                "cond_id": sample["cond_id"],
                "path": sample["path"],
                "start": start,
                "end": end,
                "n_frames": len(clip),
                "tr": TARGET_TR,
                "bold": clip.astype(np.float16),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create AABC task5 Arrow dataset")
    parser.add_argument(
        "--space",
        type=str,
        default="flat",
        choices=list(readers.READER_DICT),
        help="Target anatomical space for processing (default: flat)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to extracted AABC raw data root (default: AABC_RAW_ROOT or datasets/AABC/data/raw)",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Output root for Arrow dataset (default: datasets/AABC/data/processed)",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=DEFAULT_SPLIT_COUNTS["train"],
        help="Training samples per class (default: 3800)",
    )
    parser.add_argument(
        "--validation-per-class",
        type=int,
        default=DEFAULT_SPLIT_COUNTS["validation"],
        help="Validation samples per class (default: 806)",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=DEFAULT_SPLIT_COUNTS["test"],
        help="Test samples per class (default: 1008)",
    )
    parser.add_argument(
        "--train-batch-ids",
        nargs="+",
        type=int,
        default=None,
        help="Training subject batch ids (default: 0 1 2 3 7 10 11 12 13 14)",
    )
    parser.add_argument(
        "--validation-batch-ids",
        nargs="+",
        type=int,
        default=None,
        help="Validation subject batch ids (default: 15 16)",
    )
    parser.add_argument(
        "--test-batch-ids",
        nargs="+",
        type=int,
        default=None,
        help="Test subject batch ids (default: 17 18 19)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2912,
        help="Random seed for balanced clip selection",
    )
    parser.add_argument(
        "--num_proc",
        "-j",
        type=int,
        default=32,
        help="Number of parallel generator processes",
    )
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=1,
        help="Number of path-loading threads per generator process",
    )
    parser.add_argument(
        "--candidate-workers",
        type=int,
        default=8,
        help="Number of subject-scan threads for building candidate clip pools",
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=None,
        help="Arrow writer batch size (default: 16 for flat, 8 for mni spaces)",
    )
    sys.exit(main(parser.parse_args()))
