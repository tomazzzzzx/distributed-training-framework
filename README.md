# Distributed Training Framework

Production-grade distributed training framework optimized for AMD Instinct GPUs.

## Features
- **FSDP2** with activation checkpointing
- **Pipeline parallelism** for model sharding
- **Tensor parallelism** for intra-node distribution
- **Mixed precision** training (FP16/BF16/FP8)
- **Gradient accumulation** with dynamic loss scaling

## Performance (8x MI300X)
| Model | MI300X | H100 | vs A100 |
|-------|--------|------|---------|
| Llama-3-70B | 1,847 tok/s | 1,721 tok/s | 2.1x |
| Llama-3-8B | 14,200 tok/s | 13,800 tok/s | 2.3x |

## Quick Start
```bash
pip install -r requirements.txt
torchrun --nproc_per_node=8 train.py \
    --model meta-llama/Llama-3-70B \
    --strategy fsdp --precision bf16 --batch_size 4
```

## License: Apache 2.0
