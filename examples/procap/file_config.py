from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


VALID_CELL_TYPES = {"K562", "A673", "CACO2", "CALU3", "HUVEC", "MCF10A"}


@dataclass(frozen=True)
class FoldFilesConfig:
    proj_dir: Path
    cell_type: str
    data_type: str
    fold: int
    model_name: str
    timestamp: str
    use_unmappability_mask: bool

    @classmethod
    def create(
        cls,
        *,
        proj_dir: str | Path,
        cell_type: str = "K562",
        data_type: str = "procap",
        fold: int | str = 1,
        model_name: str,
        timestamp: str | None = None,
        use_unmappability_mask: bool = True,
    ) -> "FoldFilesConfig":
        cell_type = str(cell_type)
        if cell_type not in VALID_CELL_TYPES:
            raise ValueError(f"Unsupported cell_type={cell_type!r}. Expected one of {sorted(VALID_CELL_TYPES)}.")

        fold = int(fold)
        if fold < 1 or fold > 7:
            raise ValueError(f"fold must be in [1, 7], got {fold}.")

        timestamp = timestamp or datetime.now().strftime("%y%m%d_%H%M%S")
        cfg = cls(
            proj_dir=Path(proj_dir),
            cell_type=cell_type,
            data_type=str(data_type),
            fold=fold,
            model_name=str(model_name),
            timestamp=timestamp,
            use_unmappability_mask=bool(use_unmappability_mask),
        )
        cfg.validate()
        return cfg

    @property
    def genome_path(self) -> Path:
        return self.proj_dir / "genome" / "hg38.withrDNA.fasta"

    @property
    def chrom_size_path(self) -> Path:
        return self.proj_dir / "genome" / "hg38.withrDNA.chrom.sizes"

    @property
    def mask_bw_path(self) -> Path | None:
        if not self.use_unmappability_mask:
            return None
        return self.proj_dir / "annotations" / "hg38.k36.multiread.umap.bigWig"

    @property
    def processed_dir(self) -> Path:
        return self.proj_dir / "data" / self.data_type / "processed" / self.cell_type

    @property
    def all_peak_path(self) -> Path:
        return self.processed_dir / "peaks.bed.gz"

    @property
    def plus_bw_path(self) -> Path:
        return self.processed_dir / "5prime.pos.bigWig"

    @property
    def minus_bw_path(self) -> Path:
        return self.processed_dir / "5prime.neg.bigWig"

    @property
    def train_peak_path(self) -> Path:
        return self.processed_dir / f"peaks_fold{self.fold}_train.bed.gz"

    @property
    def val_peak_path(self) -> Path:
        return self.processed_dir / f"peaks_fold{self.fold}_val.bed.gz"

    @property
    def test_peak_path(self) -> Path:
        return self.processed_dir / f"peaks_fold{self.fold}_test.bed.gz"

    @property
    def train_val_peak_path(self) -> Path:
        return self.processed_dir / f"peaks_fold{self.fold}_train_and_val.bed.gz"

    @property
    def dnase_train_path(self) -> Path:
        return self.processed_dir / f"dnase_peaks_no_{self.data_type}_overlap_fold{self.fold}_train.bed.gz"

    @property
    def dnase_val_path(self) -> Path:
        return self.processed_dir / f"dnase_peaks_no_{self.data_type}_overlap_fold{self.fold}_val.bed.gz"

    @property
    def dnase_test_path(self) -> Path:
        return self.processed_dir / f"dnase_peaks_no_{self.data_type}_overlap_fold{self.fold}_test.bed.gz"

    @property
    def model_dir(self) -> Path:
        return self.proj_dir / "models" / self.model_name / self.data_type / self.cell_type / f"fold{self.fold}" / self.timestamp

    @property
    def checkpoint_dir(self) -> Path:
        return self.model_dir / "checkpoints"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / "metrics.tsv"

    @property
    def params_path(self) -> Path:
        return self.model_dir / "params_saved.yaml"

    @property
    def config_path(self) -> Path:
        return self.model_dir / "config_saved.yaml"

    @property
    def eval_dir(self) -> Path:
        return self.model_dir / "evals"

    def validate(self) -> None:
        required = [
            self.genome_path,
            self.chrom_size_path,
            self.all_peak_path,
            self.plus_bw_path,
            self.minus_bw_path,
            self.train_peak_path,
            self.val_peak_path,
            self.test_peak_path,
            self.train_val_peak_path,
            self.dnase_train_path,
            self.dnase_val_path,
            self.dnase_test_path,
        ]
        if self.mask_bw_path is not None:
            required.append(self.mask_bw_path)

        missing = [str(path) for path in required if not path.exists()]
        if missing:
            joined = "\n  ".join(missing)
            raise FileNotFoundError(f"Missing required PRO-cap file(s):\n  {joined}")

    def as_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "timestamp": self.timestamp,
            "proj_dir": str(self.proj_dir),
            "cell_type": self.cell_type,
            "data_type": self.data_type,
            "fold": self.fold,
            "model_name": self.model_name,
            "genome_path": str(self.genome_path),
            "chrom_size_path": str(self.chrom_size_path),
            "mask_bw_path": str(self.mask_bw_path) if self.mask_bw_path else None,
            "all_peak_path": str(self.all_peak_path),
            "plus_bw_path": str(self.plus_bw_path),
            "minus_bw_path": str(self.minus_bw_path),
            "train_peak_path": str(self.train_peak_path),
            "val_peak_path": str(self.val_peak_path),
            "test_peak_path": str(self.test_peak_path),
            "train_val_peak_path": str(self.train_val_peak_path),
            "dnase_train_path": str(self.dnase_train_path),
            "dnase_val_path": str(self.dnase_val_path),
            "dnase_test_path": str(self.dnase_test_path),
            "model_dir": str(self.model_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "params_path": str(self.params_path),
            "config_path": str(self.config_path),
            "eval_dir": str(self.eval_dir),
        }
