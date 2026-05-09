from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler


def one_hot_encode(sequence: str, dtype: np.dtype = np.float32) -> np.ndarray:
    sequence = sequence.upper()
    encoded = np.zeros((len(sequence), 4), dtype=dtype)
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, base in enumerate(sequence):
        j = lookup.get(base)
        if j is not None:
            encoded[i, j] = 1
    return encoded


def load_chrom_names(chrom_sizes: str | Path, filter_out: Iterable[str] = ("_", "M", "Un", "EBV")) -> list[str]:
    chroms = []
    with Path(chrom_sizes).open() as handle:
        for line in handle:
            chrom = line.strip().split()[0]
            if not chrom.startswith("chr"):
                continue
            if any(token in chrom for token in filter_out):
                continue
            chroms.append(chrom)
    return chroms


def load_bed(path: str | Path) -> dict[str, np.ndarray]:
    raw = np.loadtxt(path, dtype=str, usecols=(0, 1, 2))
    if raw.ndim == 1:
        raw = raw[None, :]
    return {
        "chrom": raw[:, 0],
        "start": raw[:, 1].astype(int),
        "end": raw[:, 2].astype(int),
    }


def _open_bigwig(path: str | Path):
    import pyBigWig

    bw = pyBigWig.open(str(path), "r")
    if bw is None:
        raise OSError(f"Could not open BigWig: {path}")
    return bw


def _get_signal(bw, chrom: str, start: int, end: int) -> np.ndarray:
    values = bw.values(chrom, start, end, numpy=True)
    return np.nan_to_num(values).astype(np.float32, copy=False)


def _extract_one_locus(
    *,
    fasta,
    bigwigs: list,
    mask_bw,
    chrom: str,
    start: int,
    end: int,
    input_length: int,
    output_length: int,
    max_jitter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mid = start + (end - start) // 2

    half_in = input_length // 2
    seq_start = mid - half_in - max_jitter
    seq_end = mid + half_in + max_jitter + (input_length % 2)
    if seq_start < 0:
        raise ValueError(f"Sequence start is negative for {chrom}:{start}-{end}")

    half_out = output_length // 2
    sig_start = mid - half_out - max_jitter
    sig_end = mid + half_out + max_jitter + (output_length % 2)
    if sig_start < 0:
        raise ValueError(f"Signal start is negative for {chrom}:{start}-{end}")

    seq = one_hot_encode(str(fasta[chrom][seq_start:seq_end])).T
    signals = np.stack([_get_signal(bw, chrom, sig_start, sig_end) for bw in bigwigs])

    mask = None
    if mask_bw is not None:
        mask_values = _get_signal(mask_bw, chrom, sig_start, sig_end)
        mask_one_strand = (mask_values > 0).astype(np.bool_)
        mask = np.stack([mask_one_strand] * len(bigwigs))

    expected_seq_len = input_length + 2 * max_jitter
    expected_sig_len = output_length + 2 * max_jitter
    if seq.shape != (4, expected_seq_len):
        raise ValueError(f"Unexpected sequence shape {seq.shape}; expected (4, {expected_seq_len})")
    if signals.shape != (len(bigwigs), expected_sig_len):
        raise ValueError(f"Unexpected signal shape {signals.shape}; expected ({len(bigwigs)}, {expected_sig_len})")
    return seq, signals, mask


def extract_loci(
    *,
    genome_path: str | Path,
    chroms: list[str],
    bw_paths: list[str | Path],
    bed_path: str | Path,
    mask_bw_path: str | Path | None = None,
    input_length: int,
    output_length: int,
    max_jitter: int,
    verbose: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    from pyfaidx import Fasta
    from tqdm import tqdm

    fasta = Fasta(str(genome_path), sequence_always_upper=True)
    bigwigs = [_open_bigwig(path) for path in bw_paths]
    mask_bw = _open_bigwig(mask_bw_path) if mask_bw_path else None

    loci = load_bed(bed_path)
    keep = np.isin(loci["chrom"], chroms)
    kept_chroms = loci["chrom"][keep]
    starts = loci["start"][keep]
    ends = loci["end"][keep]
    if verbose:
        print(f"Loaded {len(kept_chroms)} loci from {bed_path}")

    seqs = []
    signals = []
    masks = []
    try:
        iterator = zip(kept_chroms, starts, ends)
        for chrom, start, end in tqdm(iterator, total=len(kept_chroms), disable=not verbose, desc="Extracting loci"):
            try:
                seq, signal, mask = _extract_one_locus(
                    fasta=fasta,
                    bigwigs=bigwigs,
                    mask_bw=mask_bw,
                    chrom=str(chrom),
                    start=int(start),
                    end=int(end),
                    input_length=input_length,
                    output_length=output_length,
                    max_jitter=max_jitter,
                )
            except ValueError as exc:
                if verbose:
                    print(f"Skipping {chrom}:{start}-{end}: {exc}")
                continue
            seqs.append(seq)
            signals.append(signal)
            if mask_bw is not None:
                masks.append(mask)
    finally:
        fasta.close()
        for bw in bigwigs:
            bw.close()
        if mask_bw is not None:
            mask_bw.close()

    if not seqs:
        raise RuntimeError(f"No valid loci extracted from {bed_path}")

    seq_tensor = torch.from_numpy(np.stack(seqs)).to(torch.float32)
    signal_tensor = torch.from_numpy(np.stack(signals)).to(torch.float32)
    mask_tensor = torch.from_numpy(np.stack(masks)).to(torch.bool) if mask_bw_path else None
    if verbose:
        print(
            f"Extracted {seq_tensor.shape[0]} examples: "
            f"seqs={tuple(seq_tensor.shape)}, signals={tuple(signal_tensor.shape)}, "
            f"masks={tuple(mask_tensor.shape) if mask_tensor is not None else None}"
        )
    return seq_tensor, signal_tensor, mask_tensor


class ProfileDataset(Dataset):
    def __init__(
        self,
        *,
        sequences: Tensor,
        signals: Tensor,
        masks: Tensor | None,
        input_length: int,
        output_length: int,
        max_jitter: int,
        reverse_complement: bool,
        random_seed: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.signals = signals
        self.masks = masks
        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.max_jitter = int(max_jitter)
        self.reverse_complement = bool(reverse_complement)
        self.rng = np.random.RandomState(random_seed)
        self._check_shapes()

    def _check_shapes(self) -> None:
        expected_seq_len = self.input_length + 2 * self.max_jitter
        expected_signal_len = self.output_length + 2 * self.max_jitter
        if self.sequences.ndim != 3 or self.sequences.shape[1:] != (4, expected_seq_len):
            raise ValueError(f"Unexpected sequence shape {tuple(self.sequences.shape)}")
        if self.signals.ndim != 3 or self.signals.shape[1:] != (2, expected_signal_len):
            raise ValueError(f"Unexpected signal shape {tuple(self.signals.shape)}")
        if self.masks is not None and tuple(self.masks.shape) != tuple(self.signals.shape):
            raise ValueError(f"Mask shape {tuple(self.masks.shape)} does not match signals {tuple(self.signals.shape)}")

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self.max_jitter == 0:
            jitter = 0
        else:
            jitter = int(self.rng.randint(0, 2 * self.max_jitter))

        x = self.sequences[idx, :, jitter : jitter + self.input_length]
        y = self.signals[idx, :, jitter : jitter + self.output_length]
        mask = None
        if self.masks is not None:
            mask = self.masks[idx, :, jitter : jitter + self.output_length]

        if self.reverse_complement and np.random.rand() < 0.5:
            x = torch.flip(x, dims=(0, 1))
            y = torch.flip(y, dims=(0, 1))
            if mask is not None:
                mask = torch.flip(mask, dims=(0, 1))

        item = {"x": x.to(torch.float32), "y": y.to(torch.float32)}
        if mask is not None:
            item["mask"] = mask.to(torch.bool)
        return item


class MultiSourceBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset_lengths: list[int], source_fracs: list[float], batch_size: int, seed: int | None = None) -> None:
        if len(dataset_lengths) != len(source_fracs):
            raise ValueError("dataset_lengths and source_fracs must have the same length.")
        if abs(sum(source_fracs) - 1.0) > 1e-6:
            raise ValueError(f"source_fracs must sum to 1, got {sum(source_fracs)}.")
        self.dataset_lengths = [int(length) for length in dataset_lengths]
        self.source_fracs = [float(frac) for frac in source_fracs]
        self.batch_size = int(batch_size)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))

        self.batch_sizes = [int(torch.ceil(torch.tensor(self.batch_size * frac)).item()) for frac in self.source_fracs]
        overflow = sum(self.batch_sizes) - self.batch_size
        self.batch_sizes[0] -= overflow
        if self.batch_sizes[0] < 1:
            raise ValueError(f"Primary source batch size must be at least 1, got {self.batch_sizes[0]}.")

        self.offsets = [0]
        for length in self.dataset_lengths[:-1]:
            self.offsets.append(self.offsets[-1] + length)
        self.num_batches = self.dataset_lengths[0] // self.batch_sizes[0]

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        primary = torch.randperm(self.dataset_lengths[0], generator=self.generator)
        primary = primary[: self.num_batches * self.batch_sizes[0]].view(self.num_batches, self.batch_sizes[0])

        for batch_idx in range(self.num_batches):
            parts = [primary[batch_idx]]
            for source_idx in range(1, len(self.dataset_lengths)):
                start = self.offsets[source_idx]
                end = start + self.dataset_lengths[source_idx]
                count = self.batch_sizes[source_idx]
                parts.append(torch.randint(start, end, (count,), generator=self.generator))
            batch = torch.cat(parts)
            yield batch[torch.randperm(batch.numel(), generator=self.generator)].tolist()


class ProCapDataModule:
    def __init__(
        self,
        *,
        config: dict,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int | None = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = prefetch_factor
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.verbose = bool(verbose)
        self.chroms = load_chrom_names(config["chrom_size_path"])
        self.train_dataset = None
        self.valid_dataset = None

    def setup(self) -> None:
        self.train_dataset = self._make_train_dataset()
        self.valid_dataset = self._make_dataset(
            self.config["val_peak_path"],
            mask=False,
            jitter=False,
            reverse_complement=False,
        )

    def _make_dataset(self, bed_path: str, *, mask: bool, jitter: bool, reverse_complement: bool) -> ProfileDataset:
        max_jitter = int(self.config["max_jitter"]) if jitter else 0
        seqs, signals, masks = extract_loci(
            genome_path=self.config["genome_path"],
            chroms=self.chroms,
            bw_paths=[self.config["plus_bw_path"], self.config["minus_bw_path"]],
            bed_path=bed_path,
            mask_bw_path=self.config["mask_bw_path"] if mask else None,
            input_length=int(self.config["input_length"]),
            output_length=int(self.config["output_length"]),
            max_jitter=max_jitter,
            verbose=self.verbose,
        )
        return ProfileDataset(
            sequences=seqs,
            signals=signals,
            masks=masks,
            input_length=int(self.config["input_length"]),
            output_length=int(self.config["output_length"]),
            max_jitter=max_jitter,
            reverse_complement=reverse_complement,
            random_seed=self.config.get("random_seed"),
        )

    def _make_train_dataset(self):
        peak_dataset = self._make_dataset(
            self.config["train_peak_path"],
            mask=bool(self.config["mask_bw_path"]),
            jitter=True,
            reverse_complement=bool(self.config["reverse_complement"]),
        )
        if not self.config["use_dnase"]:
            return peak_dataset
        dnase_dataset = self._make_dataset(
            self.config["dnase_train_path"],
            mask=bool(self.config["mask_bw_path"]),
            jitter=True,
            reverse_complement=bool(self.config["reverse_complement"]),
        )
        return [peak_dataset, dnase_dataset]

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if self.valid_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.valid_dataset, shuffle=False, drop_last=False)

    def _loader(self, dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)

        if isinstance(dataset, list):
            sampler = MultiSourceBatchSampler(
                dataset_lengths=[len(ds) for ds in dataset],
                source_fracs=list(self.config["source_fracs"]),
                batch_size=self.batch_size,
                seed=self.config.get("random_seed"),
            )
            return DataLoader(ConcatDataset(dataset), batch_sampler=sampler, **kwargs)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
