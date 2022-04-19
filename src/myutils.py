import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import f1_score
import argparse
import importlib

def parse_configs():
    parser = argparse.ArgumentParser(description='5329')
    parser.add_argument('--config', type=str, default='base',
                        help='Name of the config file')
    args = parser.parse_args()
    config = importlib.import_module('configs.'+args.config).Config()
    return config

def evaluate(predictions, labels):
    if len(predictions)==0:
        return 0
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    samples_f1 = f1_score(labels, predictions, average='samples')
    return micro_f1, macro_f1, weighted_f1, samples_f1