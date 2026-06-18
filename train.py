import torch, torch.distributed as dist
from config import TrainingConfig
from trainer import DistributedTrainer

def main():
    config = TrainingConfig.from_args()
    dist.init_process_group("nccl")
    trainer = DistributedTrainer(config)
    trainer.train(config.epochs)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
