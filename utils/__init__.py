import os
import math
import torch
import numpy
import random
import logging
import matplotlib.pyplot as plt


LOGGER_NAME = "Classification"

LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s : %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
LOGGER.addHandler(stream_handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.use_deterministic_algorithms(True) # CUDA >= 10.2


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_env():
    """
    .env
        MASTER_ADDR : master ip
        MASTER_PORT : master port
    """
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TORCH_USE_CUDA_DSA'] = '1'


def get_losses(loss_history, step):
    losses = []
    length = len(loss_history)
    chunk_step = math.ceil(length / step)
    for i in range(0, length, chunk_step):
        cur = [loss for _, loss in loss_history[i: i+chunk_step]]
        losses.append(sum(cur) / len(cur))
    
    return losses


def save_loss_history(base_path, train_loss_history, valid_loss_history, step):
    train_losses = get_losses(train_loss_history, step)
    valid_losses = get_losses(valid_loss_history, step)

    plt.figure(figsize=(step/2, 8))
    plt.title(f"Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.plot(range(1, step + 1), train_losses, marker='o', label="train")
    plt.plot(range(1, step + 1), valid_losses, marker='o', label="validation")

    plt.xticks(range(1, step + 1))
    plt.legend()
    plt.savefig(f"{base_path}/train_loss.png", bbox_inches='tight', dpi=300)