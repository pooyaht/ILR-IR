import argparse
from pyHGT.model import TypeGAT
import torch
import torch.nn as nn
import numpy as np
import os


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--data",
                      default="./data/ICEWS14_forecasting", help="data directory")
    args = args.parse_args()
    return args


args = parse_args()


def load_data_info():
    data_path = args.data

    with open(os.path.join(data_path, 'stat.txt'), 'r') as fr:
        for line in fr:
            line_split = line.split()
            num_e, num_r = int(line_split[0]), int(line_split[1])

    return num_e, num_r


def print_model_summary():
    num_e, num_r = load_data_info()
    embedding_size = 200

    relation_embeddings = torch.FloatTensor(
        np.random.randn(num_r * 2, embedding_size))

    model = TypeGAT(num_e, num_r * 2, relation_embeddings, embedding_size)

    print("=" * 50)
    print("MODEL ARCHITECTURE")
    print("=" * 50)
    print(model)
    print()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print("PARAMETER SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print()

    print("=" * 50)
    print("DETAILED PARAMETER BREAKDOWN")
    print("=" * 50)
    for name, param in model.named_parameters():
        print(f"{name:30} | Shape: {str(param.shape):20} | Params: {param.numel():,}")

    print()
    print("=" * 50)
    print("DATA INFO")
    print("=" * 50)
    print(f"Number of entities: {num_e:,}")
    print(f"Number of relations: {num_r:,}")
    print(f"Total relations (with inverse): {num_r * 2:,}")
    print(f"Embedding dimension: {embedding_size}")


if __name__ == "__main__":
    print_model_summary()
