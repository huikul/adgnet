import copy
import os, yaml
import math
import numpy as np
import pickle
import random
import argparse
import sys
import logging
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from neural_network_grasp import ADG_Net, get_branch_names, get_all_parameters, set_requires_grad
# from accelerate import Accelerator
# from accelerate import DistributedDataParallelKwargs
from torch_geometric.utils import dense_to_sparse
import multiprocessing
import shutil
#
#
#
path_cfg_yaml = "cfg/01_nn_train.yaml"
with open(path_cfg_yaml, 'r') as f:
    cfg_all = yaml.load(f.read(), Loader=yaml.FullLoader)

device = torch.device(cfg_all['training_params']['device'] if torch.cuda.is_available() else "cpu")
NN_model = ADG_Net(cfg=cfg_all['NN_params'])
NN_model = NN_model.to(device)

'''
pesudo code: load dataset

'''

def compute_loss_rgbd_model(model, lst_in, lst_ref):
    image, vector_pose, vector_tactile, vector_joints, Gindex_tac_dofs = lst_in
    pose_ref, qg_ref, suc_ref, joints_ref = lst_ref
    #
    out_pose, out_gq, out_success, out_joints = \
        model(image, vector_tactile, vector_joints, Gindex_tac_dofs, 'rgbd_model')
    #
    loss_pose = F.mse_loss(out_pose, pose_ref)
    loss_gq = F.mse_loss(out_gq, qg_ref)
    loss_success = F.mse_loss(out_success, suc_ref)
    loss_dofs = F.mse_loss(out_joints, joints_ref)
    #
    return {
        'loss_total': loss_pose + loss_gq + loss_success + loss_dofs,
        'losses': {
            'loss_pose': loss_pose,
            'loss_gq': loss_gq,
            'loss_success': loss_success,
            'loss_dofs': loss_dofs,
        },
        'pred': {
            'g_pose': out_pose,
            'gq': out_gq,
            'g_success': out_success,
            'g_dofs': out_joints
        }
    }


def evaluate_rgbd_model(model, lst_in):
    image, vector_pose, vector_tactile, vector_joints, Gindex_tac_dofs = lst_in
    #
    out_pose, out_gq, out_success, out_joints = \
        model(image, vector_tactile, vector_joints, Gindex_tac_dofs, 'rgbd_model')
    #
    return {
        'loss_total': None,
        'losses': {
            'loss_pose': None,
            'loss_gq': None,
            'loss_success': None,
            'loss_dofs': None
        },
        'pred': {
            'g_pose': out_pose,
            'gq': out_gq,
            'g_success': out_success,
            'g_dofs': out_joints
        }
    }


def compute_loss_multi_model(model, lst_in, lst_ref):
    image, vector_pose, vector_tactile, vector_joints, Gindex_tac_dofs = lst_in
    pose_ref, qg_ref, suc_ref, joints_ref = lst_ref
    #
    out_pose, out_gq, out_success, out_joints = \
        model(image, vector_tactile, vector_joints, Gindex_tac_dofs, 'multi_model')
    #
    # loss_coor = F.mse_loss(out_coor, coor_ref)
    # loss_dir = F.mse_loss(out_dir, dir_ref)
    loss_gq = F.mse_loss(out_gq, qg_ref)
    loss_success = F.mse_loss(out_success, suc_ref)
    loss_dofs = F.mse_loss(out_joints, joints_ref)
    #
    return {
        'loss_total': loss_success + loss_gq + loss_dofs,
        'losses': {
            'loss_gq': loss_gq,
            'loss_success': loss_success,
            'loss_dofs': loss_dofs
        },
        'pred': {
            'g_pose': out_pose,
            'gq': out_gq,
            'g_success': out_success,
            'g_dofs': out_joints
        }
    }

def evaluate_multi_model(model, lst_in):
    image, vector_pose, vector_tactile, vector_joints, Gindex_tac_dofs = lst_in
    #
    out_pose, out_gq, out_success, out_joints = \
        model(image, vector_tactile, vector_joints, Gindex_tac_dofs, 'multi_model')
    #
    return {
        'loss_total': None,
        'losses': {
            'loss_gq': None,
            'loss_success': None,
            'loss_dofs': None
        },
        'pred': {
            'g_pose': out_pose,
            'gq': out_gq,
            'g_success': out_success,
            'g_dofs': out_joints
        }
    }


def train():
    pass


def test():
    pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # NOTE: this is a test for ADG-Net I in our paper
    ''''''
    Majd_tac_dofs = np.array(cfg_all['graph']['graph_dof_tac'])
    non_zero_indices = np.argwhere(Majd_tac_dofs == 1)
    for i, j in non_zero_indices:   # For each non-zero element, mirror its position
        Majd_tac_dofs[j, i] = 1  # Mirror the position
    graph_index_tac_dofs = torch.from_numpy(Majd_tac_dofs).to(device)
    graph_index_tac_dofs, _ = dense_to_sparse(graph_index_tac_dofs)

    #
    rgbd_img = torch.ones([1, 3, 240, 240]).to(device)
    pose = torch.ones([1, 6]).to(device)
    dofs = torch.ones([1, 22]).to(device)
    tactiles = torch.ones([1, 5]).to(device)
    qg_ref = torch.ones([1, 1]).to(device)
    suc_ref = torch.ones([1, 1]).to(device)
    # pose_ref, qg_ref, suc_ref, joints_ref = lst_ref

    # Calculate the total number of parameters
    total_params = count_parameters(NN_model)
    print("Total number of parameters:", total_params)

    eva_rgb_model = \
        compute_loss_rgbd_model(NN_model, lst_in=[rgbd_img, pose, tactiles, dofs, graph_index_tac_dofs], lst_ref=[pose, qg_ref, suc_ref, dofs])

    pred_rgb_model = \
        evaluate_rgbd_model(NN_model, lst_in=[rgbd_img, pose, tactiles, dofs, graph_index_tac_dofs])

    eva_multi_model = \
        compute_loss_multi_model(NN_model, lst_in=[rgbd_img, pose, tactiles, dofs, graph_index_tac_dofs], lst_ref=[pose, qg_ref, suc_ref, dofs])

    pred_multi_model = \
        evaluate_multi_model(NN_model, lst_in=[rgbd_img, pose, tactiles, dofs, graph_index_tac_dofs])

    ###################################################

if __name__ == '__main__':
    main()
