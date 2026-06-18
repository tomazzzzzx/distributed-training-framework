import torch, torch.distributed as dist
from config import TrainingConfig

def main():
    cfg = TrainingConfig.from_args()
    dist.init_process_group("nccl")
    print(f"Training {cfg.model_name} with {cfg.strategy} strategy")
    dist.destroy_process_group()

if __name__ == "__main__": main()
