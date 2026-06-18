from dataclasses import dataclass
import argparse

@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-3-8B"
    batch_size: int = 4
    lr: float = 2e-5
    precision: str = "bf16"
    strategy: str = "fsdp"
    max_steps: int = 10000
    grad_accum: int = 8
    warmup: int = 100
    grad_clip: float = 1.0
    @classmethod
    def from_args(cls):
        p = argparse.ArgumentParser()
        for n, f in cls.__dataclass_fields__.items():
            p.add_argument(f"--{n}", type=f.type, default=f.default)
        return cls(**{k:v for k,v in vars(p.parse_args()).items() if k in cls.__dataclass_fields__})
