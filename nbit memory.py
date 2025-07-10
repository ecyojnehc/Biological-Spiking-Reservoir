import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import LIF
from snntorch import functional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np
from numpy import load

import sys
import os
sys.path.append(os.path.relpath("robot_trajectories.py"))

from warnings import warn
from snntorch.surrogate import atan

import pandas as pd
from copy import deepcopy
import random
import seaborn as sns

def generate_parameters(recall_L):
    while True:
        length = np.random.randint(2 * recall_L + 1, 25)
        low = recall_L + 1
        high = length - recall_L
        if low < high:
            cue = np.random.randint(low, high)
            return length, cue

def generate_inputs (length,cue):

    signal_inputs=np.zeros((length,1))
    signal_inputs[cue]=1

    random_sequence=np.zeros((length,1))
    for t in range(length):
         random_sequence[t]=np.random.uniform(low=0,high=1)
    return signal_inputs,random_sequence

def generate_time_series_inputs(signal_inputs, random_sequence):
    return np.hstack((random_sequence, signal_inputs))

def recall_sequence (random_sequence, signal_inputs,recall_L):
        length=len(signal_inputs)
        outputs=np.zeros((length,1))
        for t in range(length):
             if signal_inputs[t]==1:
                outputs[t+1:t+1+recall_L]=random_sequence[t-recall_L:t]
                return outputs

recall_L=5
seq_length,cue_sig=generate_parameters(recall_L)
sig_inp,seq_inp=generate_inputs(seq_length,cue_sig)
X = generate_time_series_inputs(sig_inp, seq_inp)
Y = recall_sequence(seq_inp, sig_inp, recall_L)


print("Input X (shape {}):\n".format(X.shape), X)
print("\nOutput Y (shape {}):\n".format(Y.shape), Y)


class SequenceRecallDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=500, recall_L=5):
        self.samples = []
        for _ in range(num_samples):
            seq_length, cue = generate_parameters(recall_L)
            sig_inp, seq_inp = generate_inputs(seq_length, cue)
            X = generate_time_series_inputs(sig_inp, seq_inp)
            Y = recall_sequence(seq_inp, sig_inp, recall_L)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
dataset = SequenceRecallDataset(num_samples=400, recall_L=5)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

