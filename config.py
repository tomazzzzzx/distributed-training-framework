from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-3-8B"
    batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 3
    strategy: str = "fsdp"
    precision: str = "bf16"
    grad_clip: float = 1.0

    @classmethod
    def from_args(cls):
        import argparse
        p = argparse.ArgumentParser()
        for f in cls.__dataclass_fields__:
            p.add_argument(f"--{f}", default=getattr(cls, f))
        return cls(**vars(p.parse_args()))
