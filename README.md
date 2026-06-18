# Distributed Training Framework

Production-grade distributed training for AMD Instinct GPUs.

## Features
- FSDP2 + activation checkpointing
- Pipeline + tensor parallelism
- Mixed precision FP16/BF16/FP8
- Gradient accumulation + WandB

## Benchmarks
| Model | MI300X (8GPU) | vs A100 |
|-------|--------------|---------|
| Llama-3-70B | 1,847 tok/s | 2.1x |

## License: Apache 2.0
