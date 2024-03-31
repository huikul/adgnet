import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torchvision.models as models
# from torchvision.transforms import ToTensor, Resize
from cbam import CBAM
import numpy as np

class VectorBranch(nn.Module):
    def __init__(self, info_dim=1, emb_size=768):
        super(VectorBranch, self).__init__()
        self.proj = nn.Linear(info_dim, emb_size)
    def forward(self, x):
        return self.proj(x)

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in


class ResidualBlock_1D(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

class ResCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.cbam = CBAM(out_channels, 16)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.bn2(self.conv2(x))

        x = F.leaky_relu(x)
        x = self.cbam(x)

        return x + x_in


# Multi-Branch Transformer Model
class ADG_Net(nn.Module):
    def __init__(self, cfg):
        super(ADG_Net, self).__init__()
        #
        self.imgnet_conv0 = nn.Conv2d(cfg['input']['dim_img'][0],
                                      cfg['structure']['img_branch']['channel_size'][0],
                                      kernel_size=cfg['structure']['img_branch']['kernel_size'][0],
                                      stride=cfg['structure']['img_branch']['strides'][0],
                                      padding=(cfg['structure']['img_branch']['kernel_size'][0] - 1)//2)
        self.imgnet_bn0 = nn.BatchNorm2d(cfg['structure']['img_branch']['channel_size'][0])
        #
        self.imgnet_conv1 = nn.Conv2d(cfg['structure']['img_branch']['channel_size'][0],
                                      cfg['structure']['img_branch']['channel_size'][1],
                                      kernel_size=cfg['structure']['img_branch']['kernel_size'][1],
                                      stride=cfg['structure']['img_branch']['strides'][1],
                                      padding=(cfg['structure']['img_branch']['kernel_size'][1] - 1)//2)
        self.imgnet_bn1 = nn.BatchNorm2d(cfg['structure']['img_branch']['channel_size'][1])
        #
        self.imgnet_conv2 = nn.Conv2d(cfg['structure']['img_branch']['channel_size'][1],
                                      cfg['structure']['img_branch']['channel_size'][2],
                                      kernel_size=cfg['structure']['img_branch']['kernel_size'][2],
                                      stride=cfg['structure']['img_branch']['strides'][2],
                                      padding=(cfg['structure']['img_branch']['kernel_size'][2] - 1)//2)
        self.imgnet_bn2 = nn.BatchNorm2d(cfg['structure']['img_branch']['channel_size'][2])
        #
        self.dof_tactile1 = GCNConv(1, cfg['structure']['share_branch']['features'])
        self.dof_tactile2 = GCNConv(cfg['structure']['share_branch']['features'], cfg['structure']['share_branch']['features'])
        self.dof_tactile3 = GCNConv(cfg['structure']['share_branch']['features'], cfg['structure']['share_branch']['features'])
        self.dof_tactile4 = GCNConv(cfg['structure']['share_branch']['features'], cfg['structure']['share_branch']['features'])
        self.doc_tac_fc = nn.Linear(27 * cfg['structure']['share_branch']['features'], cfg['structure']['share_branch']['features'])
        #
        # shared layers
        self.share_cbam1 = nn.Sequential(
            ResCBAMBlock(cfg['structure']['img_branch']['channel_size'][2], cfg['structure']['img_branch']['channel_size'][3]),
            ResCBAMBlock(cfg['structure']['img_branch']['channel_size'][3], cfg['structure']['img_branch']['channel_size'][3]),
            ResCBAMBlock(cfg['structure']['img_branch']['channel_size'][3], cfg['structure']['img_branch']['channel_size'][3])
        )
        # pose
        self.pose_cbam1 = nn.Sequential(
            ResCBAMBlock(cfg['structure']['img_branch']['channel_size'][3], cfg['structure']['coor_branch']['channel_size'][0]),
            ResCBAMBlock(cfg['structure']['coor_branch']['channel_size'][0], cfg['structure']['coor_branch']['channel_size'][1])
        )
        self.pose_fc1 = nn.Linear(20 * 20 * 32, cfg['structure']['share_branch']['features'])
        self.pose_fc2 = nn.Linear(cfg['structure']['share_branch']['features'], cfg['output']['dim_pose'])
        # gq
        self.gq_cbam1 = nn.Sequential(
            ResCBAMBlock(cfg['structure']['img_branch']['channel_size'][3], cfg['structure']['gq_branch']['channel_size'][0]),
            ResCBAMBlock(cfg['structure']['gq_branch']['channel_size'][0], cfg['structure']['gq_branch']['channel_size'][1]),
        )
        self.gq_fc1 = nn.Linear(20 * 20 * 32, cfg['structure']['share_branch']['features'])
        self.gq_fc2 = nn.Linear(cfg['structure']['share_branch']['features'], cfg['output']['dim_gq'])
        self.gq_fc3 = nn.Linear(cfg['structure']['share_branch']['features'], cfg['output']['dim_gq'])
        # success
        self.suc_cbam1 = nn.Sequential(
            ResCBAMBlock(cfg['structure']['img_branch']['channel_size'][3], cfg['structure']['suc_branch']['channel_size'][0]),
            ResCBAMBlock(cfg['structure']['suc_branch']['channel_size'][0], cfg['structure']['suc_branch']['channel_size'][1])
        )
        self.suc_fc1 = nn.Linear(20 * 20 * 32, cfg['structure']['share_branch']['features'])
        self.suc_fc2 = nn.Linear(cfg['structure']['share_branch']['features'], cfg['output']['dim_suc'])
        self.suc_fc3 = nn.Linear(cfg['structure']['share_branch']['features'], cfg['output']['dim_suc'])
        # dofs
        self.dofs_cbam1 = nn.Sequential(
            ResCBAMBlock(cfg['structure']['img_branch']['channel_size'][3], cfg['structure']['dofs_branch']['channel_size'][0]),
            ResCBAMBlock(cfg['structure']['dofs_branch']['channel_size'][0], cfg['structure']['dofs_branch']['channel_size'][1])
        )
        self.dofs_fc1 = nn.Linear(20 * 20 * 32, cfg['structure']['share_branch']['features'])
        self.dofs_fc2 = nn.Linear(cfg['structure']['share_branch']['features'], cfg['output']['dim_dofs'])
        self.dofs_fc3 = nn.Linear(cfg['structure']['share_branch']['features'], cfg['output']['dim_dofs'])
        #
        #
        self.dropout_pose = nn.Dropout(p=cfg['dropout'])
        self.dropout_gq = nn.Dropout(p=cfg['dropout'])
        self.dropout_success = nn.Dropout(p=cfg['dropout'])
        self.dropout_dofs = nn.Dropout(p=cfg['dropout'])

    def forward_rgbd_model(self, image, vector_tactile, vector_joints, graph_tac_dofs):
        '''
        tac_dofs = torch.cat((vector_tactile, vector_joints), dim=1).unsqueeze(1).unsqueeze(3)
        G_tac_dofs = F.leaky_relu(self.dof_tactile1(tac_dofs, graph_tac_dofs))
        G_tac_dofs = F.leaky_relu(self.dof_tactile2(G_tac_dofs, graph_tac_dofs))
        G_tac_dofs = F.leaky_relu(self.dof_tactile3(G_tac_dofs, graph_tac_dofs))
        G_tac_dofs = F.leaky_relu(self.dof_tactile4(G_tac_dofs, graph_tac_dofs))
        G_tac_dofs = G_tac_dofs.view(G_tac_dofs.size(0), -1)
        G_tac_dofs = self.doc_tac_fc(G_tac_dofs)
        '''
        # print(G_tac_dofs.shape)
        #
        in_img = F.leaky_relu(self.imgnet_bn0(self.imgnet_conv0(image)))
        in_img = F.leaky_relu(self.imgnet_bn1(self.imgnet_conv1(in_img)))
        in_img = F.leaky_relu(self.imgnet_bn2(self.imgnet_conv2(in_img)))
        # print(in_img.shape)
        share_info = self.share_cbam1(in_img)
        # pose
        out_pose = self.pose_cbam1(share_info)
        out_pose = self.dropout_pose(out_pose)
        out_pose = out_pose.view(out_pose.size(0), -1)
        # print(out_coor.shape)
        out_pose = self.pose_fc1(out_pose)
        out_pose = self.pose_fc2(out_pose)
        out_pose = torch.sigmoid(out_pose)
        # gq
        out_gq = self.gq_cbam1(share_info)
        out_gq = self.dropout_gq(out_gq)
        out_gq = out_gq.view(out_gq.size(0), -1)
        out_gq = self.gq_fc1(out_gq)
        out_gq = self.gq_fc2(out_gq)
        out_gq = torch.sigmoid(out_gq)
        # success
        out_suc = self.suc_cbam1(share_info)
        out_suc = self.dropout_success(out_suc)
        out_suc = out_suc.view(out_suc.size(0), -1)
        out_suc = self.suc_fc1(out_suc)
        out_suc = self.suc_fc2(out_suc)
        out_suc = torch.sigmoid(out_suc)
        # dofs
        out_dofs = self.dofs_cbam1(share_info)
        out_dofs = self.dropout_dofs(out_dofs)
        out_dofs = out_dofs.view(out_dofs.size(0), -1)
        out_dofs = self.dofs_fc1(out_dofs)
        out_dofs = self.dofs_fc2(out_dofs)
        out_dofs = torch.sigmoid(out_dofs)
        #
        return out_pose, out_gq, out_suc, out_dofs

    def forward_multi_model(self, image, vector_tactile, vector_joints, graph_tac_dofs):
        tac_dofs = torch.cat((vector_tactile, vector_joints), dim=1).unsqueeze(1).unsqueeze(3)
        G_tac_dofs = F.leaky_relu(self.dof_tactile1(tac_dofs, graph_tac_dofs))
        G_tac_dofs = F.leaky_relu(self.dof_tactile2(G_tac_dofs, graph_tac_dofs))
        G_tac_dofs = F.leaky_relu(self.dof_tactile3(G_tac_dofs, graph_tac_dofs))
        G_tac_dofs = F.leaky_relu(self.dof_tactile4(G_tac_dofs, graph_tac_dofs))
        print(G_tac_dofs.shape)
        G_tac_dofs = G_tac_dofs.view(G_tac_dofs.size(0), -1)
        print(G_tac_dofs.shape)
        G_tac_dofs = self.doc_tac_fc(G_tac_dofs)
        #
        in_img = F.leaky_relu(self.imgnet_bn0(self.imgnet_conv0(image)))
        in_img = F.leaky_relu(self.imgnet_bn1(self.imgnet_conv1(in_img)))
        in_img = F.leaky_relu(self.imgnet_bn2(self.imgnet_conv2(in_img)))
        share_info = self.share_cbam1(in_img)
        # pose
        out_pose = self.pose_cbam1(share_info)
        out_pose = out_pose.view(out_pose.size(0), -1)
        out_pose = self.pose_fc1(out_pose)
        out_pose = self.pose_fc2(out_pose)
        out_pose = torch.sigmoid(out_pose)
        # gq
        out_gq = self.gq_cbam1(share_info)
        out_gq = out_gq.view(out_gq.size(0), -1)
        print(out_gq.shape)

        out_gq = self.gq_fc1(out_gq)
        out_gq = self.dropout_gq(out_gq)
        out_gq = self.gq_fc3(out_gq + G_tac_dofs)
        out_gq = torch.sigmoid(out_gq)
        # success
        out_suc = self.suc_cbam1(share_info)
        out_suc = out_suc.view(out_suc.size(0), -1)
        out_suc = self.suc_fc1(out_suc)
        out_suc = self.dropout_success(out_suc)
        out_suc = self.suc_fc3(out_suc + G_tac_dofs)
        out_suc = torch.sigmoid(out_suc)
        # dofs
        out_dofs = self.dofs_cbam1(share_info)
        out_dofs = out_dofs.view(out_dofs.size(0), -1)
        out_dofs = self.dofs_fc1(out_dofs)
        out_dofs = self.dropout_dofs(out_dofs)
        # print(out_dofs.shape)
        # print(G_tac_dofs.shape)
        out_dofs = self.dofs_fc3(out_dofs + G_tac_dofs)
        out_dofs = torch.sigmoid(out_dofs)
        #
        return out_pose, out_gq, out_suc, out_dofs

    def forward(self, image, vector_tactile, vector_joints, Gindex_tac_dofs, fd_mode='rgbd_model'):
        """
        :param image:
        :param vector_pixel_coor:
        :param vector_grasp_dir:
        :param vector_tactile:
        :param vector_joints:
        :param fd_mode:             rgbd_model -> input rgbd img       multi_model ->
        :return:
        """
        if fd_mode == 'rgbd_model':
            out_pose, out_gq, out_success, out_joints = \
                self.forward_rgbd_model(image, vector_tactile, vector_joints, Gindex_tac_dofs)
        else:
            out_pose, out_gq, out_success, out_joints = \
                self.forward_multi_model(image, vector_tactile, vector_joints, Gindex_tac_dofs)
        return out_pose, out_gq, out_success, out_joints


def set_requires_grad(model, branch_names, requires_grad):
    """
    :param model:
    :param branch_names:
    :param requires_grad: bool
    :return:
    """
    for name, param in model.named_parameters():
        # if any(branch_name in name for branch_name in branch_names):
        #     param.requires_grad = requires_grad
        results = []
        for branch_name in branch_names:
            if branch_name in name:
                results.append(True)
            else:
                results.append(False)
        if any(results):
            param.requires_grad = requires_grad
    pass


def get_branch_names(model):
    branch_names = []
    for name, module in model.named_children():
        branch_names.append(name)
    return branch_names


def get_all_parameters(model):
    return dict(model.named_parameters())


if __name__ == '__main__':

    path_cfg_yaml = "cfg/01_nn_train.yaml"
    with open(path_cfg_yaml, 'r') as f:
        cfg_all = yaml.load(f.read(), Loader=yaml.FullLoader)
    ADG_Net(cfg=cfg_all['NN_params'])


    pass



