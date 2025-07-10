
# creates connection matrix with specific sparseness for linear layers
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

n_exc = 8
n_inh = 2
p_EE = 0.16
p_EI = 0.205
p_II = 0.284
p_IE = 0.252


def initialize_matrix (n_exc,n_inh):
    n_total=n_exc+n_inh
    init_matrix=np.zeros((n_total,n_total))
    return init_matrix



def generate_WM_1I(weight_matrix, n_exc, p_EE, p_EI, p_II, p_IE):
    n_total = weight_matrix.shape[0]
    n_inh = n_total - n_exc


    EE = (np.random.rand(n_exc, n_exc) < p_EE).astype(int)
    EI = (np.random.rand(n_exc, n_inh) < p_EI).astype(int)
    II = (np.random.rand(n_inh, n_inh) < p_II).astype(int)
    IE = (np.random.rand(n_inh, n_exc) < p_IE).astype(int)

  
    weight_matrix[:n_exc, :n_exc] = EE
    weight_matrix[:n_exc, n_exc:] = EI
    weight_matrix[n_exc:, n_exc:] = II
    weight_matrix[n_exc:, :n_exc] = IE

    total_non0_conn= np.count_nonzero(weight_matrix)

    mu = -0.64
    sigma = 0.51
    non_zero_indices = np.where(weight_matrix != 0)
    log_normal_values = np.random.lognormal(mean=mu, sigma=sigma, size=total_non0_conn)
    weight_matrix[non_zero_indices]=log_normal_values

    return weight_matrix



def generate_WM_3I(weight_matrix, n_exc, n_inh, n_iPV, n_iSst, n_iHtr):
    n_total = weight_matrix.shape[0]
    n_inh = n_total - n_exc

    e_e: 0.16 
    e_PV: 0.395 
    e_Sst: 0.182
    e_Htr: 0.105
    PV_e: 0.411 
    PV_PV: 0.451 
    PV_Sst: 0.03
    PV_Htr: 0.22
    Sst_e: 0.424
    Sst_PV: 0.857
    Sst_Sst: 0.082
    Sst_Htr: 0.77
    Htr_e: 0.087
    Htr_PV: 0.02
    Htr_Sst: 0.0625
    Htr_Htr: 0.028

    #excitatory connections: EE, EPV, ESST, EHTR
    weight_matrix[:n_exc, :n_exc] = (np.random.rand(n_exc, n_exc) < e_e).astype(int) 
    weight_matrix[:n_exc, n_exc:n_exc+n_iPV] = (np.random.rand(n_exc, n_iPV) < e_PV).astype(int)
    weight_matrix[:n_exc,n_exc+n_iPV:n_exc+n_iPV+n_iSst] = (np.random.rand(n_exc, n_iSst) < e_Sst).astype(int)
    weight_matrix[:n_exc,n_exc+n_iPV+n_iSst:] = (np.random.rand(n_exc, n_iHtr) < e_Htr).astype(int)

    #iPV connections: PVE, PVPV, PVSST, PVHTR
    weight_matrix[n_exc:n_exc +n_iPV,:n_exc] = (np.random.rand(n_iPV, n_exc) < PV_e).astype(int)
    weight_matrix[n_exc:n_exc +n_iPV,n_exc:n_exc +n_iPV] = (np.random.rand(n_iPV, n_iPV) < PV_PV).astype(int)
    weight_matrix[n_exc:n_exc +n_iPV,n_exc+n_iPV:n_exc+n_iPV+n_iSst] = (np.random.rand(n_iPV, n_iSst) < PV_Sst).astype(int)
    weight_matrix[n_exc:n_exc +n_iPV,n_exc+n_iPV+n_iSst:] = (np.random.rand(n_iPV, n_iHtr) < PV_Htr).astype(int)
  
    #iSST connections: SSTE, SSTPV, SSTSST, SSTHTR
    weight_matrix[n_exc+n_iPV:n_exc+n_iPV+n_iSst,:n_exc] = (np.random.rand(n_iSst, n_exc) < Sst_e).astype(int)
    weight_matrix[n_exc+n_iPV:n_exc+n_iPV+n_iSst,n_exc:n_exc+n_iPV] = (np.random.rand(n_iSst, n_iPV) < Sst_PV).astype(int)
    weight_matrix[n_exc+n_iPV:n_exc+n_iPV+n_iSst,n_exc+n_iPV:n_exc+n_iPV+n_iSst] = (np.random.rand(n_iSst, n_iSst) < Sst_Sst).astype(int)
    weight_matrix[n_exc+n_iPV:n_exc+n_iPV+n_iSst,n_exc+n_iPV+n_iSst:] = (np.random.rand(n_iSst, n_iHtr) < Sst_Htr).astype(int)

    #iHTR connections: HTRE, HTRPV, HTRSST< HTRHTR
    weight_matrix[n_exc+n_iPV+n_iSst:,:n_exc] = (np.random.rand(n_iHtr, n_exc) < Htr_e).astype(int)
    weight_matrix[n_exc+n_iPV+n_iSst:,n_exc:n_exc+n_iPV] = (np.random.rand(n_iHtr, n_iPV) < Htr_PV).astype(int)
    weight_matrix[n_exc+n_iPV+n_iSst:,n_exc+n_iPV:n_exc+n_iPV+n_iSst]= (np.random.rand(n_iHtr, n_iSst) < Htr_Sst).astype(int)
    weight_matrix[n_exc+n_iPV+n_iSst:,n_exc+n_iPV+n_iSst:] = (np.random.rand(n_iHtr, n_iHtr) < Htr_Htr).astype(int)


    total_non0_conn= np.count_nonzero(weight_matrix)
    mu = -0.64
    sigma = 0.51
    non_zero_indices = np.where(weight_matrix != 0)
    log_normal_values = np.random.lognormal(mean=mu, sigma=sigma, size=total_non0_conn)
    weight_matrix[non_zero_indices]=log_normal_values

    return weight_matrix



#matrix = initialize_matrix(n_exc, n_inh)
#final_matrix = generate_WM_1I(matrix, n_exc, p_EE, p_EI, p_II, p_IE)

#print(final_matrix)