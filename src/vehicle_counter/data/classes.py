"""Load target classes and source mappings from configs/classes.yaml."""

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "classes.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_ids(config: dict) -> dict[str, int]:
    """Return {class_name: class_id}."""
    return {name: int(i) for i, name in config["names"].items()}


def source_mapping(config: dict, source: str) -> dict[str, int | None]:
    """Return {source_label: target_class_id or None}."""
    ids = class_ids(config)
    mapping = config["sources"][source]["mapping"]
    return {src: (ids[dst] if dst is not None else None) for src, dst in mapping.items()}
