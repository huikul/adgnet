import os
import glob
import pickle
import logging
# import pcl
# import torch
import time
import torch.utils.data
# import torch.nn as nn
import numpy as np
import random
import cv2
import random
# from vstsim.grasping import grasp_info
from torch_geometric.utils import dense_to_sparse

class ShadowHandGraspDataset_RT(torch.utils.data.Dataset):
    """
    generate a real-time dataset and save mask
    """
    def __init__(self, dir_img_rgbd, dir_grasp_info, cfg):
        """
        :param dir_synthetic_msk:
        :param path_source_data:
        :param flg_dp_bg_0:        bool    the default pixels in depth image is 0 or 1
        :param thresh_max_num:
        """
        self.shape_img = cfg['shape_img']
        self.cfg = cfg
        if cfg['threshold_max_grasps'] < 1:
            cfg['threshold_max_grasps'] = 1
        #
        ''''''
        self.rgb_mean_vals = np.array(cfg['mean_vals'])
        self.rgb_std_vals = np.array(cfg['std_vals'])

        self.rgb_mean_arr = np.ones([3, self.shape_img[0], self.shape_img[1]]).astype('f')
        self.rgb_mean_arr[0] *= self.rgb_mean_vals[0]
        self.rgb_mean_arr[1] *= self.rgb_mean_vals[1]
        self.rgb_mean_arr[2] *= self.rgb_mean_vals[2]

        self.rgb_std_arr = np.ones([3, self.shape_img[0], self.shape_img[1]]).astype('f')
        self.rgb_std_arr[0] *= self.rgb_std_vals[0]
        self.rgb_std_arr[1] *= self.rgb_std_vals[1]
        self.rgb_std_arr[2] *= self.rgb_std_vals[2]
        #
        self.dir_img_rgbd = dir_img_rgbd
        self.dir_grasp_info = dir_grasp_info
        self.threshold_gq: float = cfg['threshold_gq']
        self.dp_bg: float = cfg['dp_background']
        self.rgb_bg: float = cfg['rgb_background']
        '''
        path_img_rgbd = self.merge_all_route(dir_img_rgbd)
        self.grasps_list_all = []
        for cnt_path, dir_load in enumerate(path_img_rgbd):
            self.grasps_list_all += self.get_file_name(dir_load)
        '''
        #
        self.index_pixel_info = str(cfg['index_coor'])
        self.index_tactile_info = str(cfg['index_tactile'])
        self.index_dofs_info = str(cfg['index_dofs'])
        #
        # self.lst_all_grasps = self.merge_all_route(dir_img_rgbd)
        self.lst_all_grasps = self.get_all_path(self.dir_img_rgbd)
        if cfg['flg_random_shuffle']:
            random_seed: int = cfg['seed_random_shuffle']
            random.seed(random_seed)
            random.shuffle(self.lst_all_grasps)

        if cfg['flg_check_completeness']:
            self.check_completeness()
        self.conditional_selection()
        #
        self.amount = 0
        self.amount = len(self.lst_all_grasps)
        if self.amount > cfg['threshold_max_grasps']:
            self.amount = int(cfg['threshold_max_grasps'])
            self.lst_all_grasps = self.lst_all_grasps[0:self.amount]

        self.lst_dir_dp_img = []
        self.lst_dir_seg_img = []
        self.lst_dir_pixel_info = []
        self.lst_dir_grasp_info = []
        for cnt_grasp, dir_img_rgb in enumerate(self.lst_all_grasps):
            str_obj_grasp_info, str_obj_pixel_info, str_obj_name, str_scale, str_time, path_img_root = self.extract_info(dir_img_rgb)
            self.lst_dir_dp_img.append(path_img_root + '/detail/' + str_obj_pixel_info + '_dp.npy')
            self.lst_dir_seg_img.append(path_img_root + '/detail/' + str_obj_pixel_info + '_seg.npy')
            self.lst_dir_pixel_info.append(path_img_root + '/detail/' + str_obj_pixel_info + '.pickle')
            self.lst_dir_grasp_info.append(self.dir_grasp_info + '/' + str_obj_grasp_info + '.pickle')

        print('Dataset has been loaded.')
        print('Num. data: ', self.amount)

    def get_all_path(self, lst_path):
        lst_file = []
        for i, path in enumerate(lst_path):
            tmp_lst_file = self.get_folder_name(path)
            tmp_lst_file.pop()
            lst_file += self.merge_all_route(tmp_lst_file)
        return lst_file

    def merge_all_route(self, lst_route):
        file_list = []
        for i, route in enumerate(lst_route):
            file_list += self.get_file_name(route)
        return file_list

    def get_folder_name(self, file_dir_):
        """
        :param file_dir_: root dir of target documents  e.g.: home_dir + "/dataset/ycb_meshes_google/backup/003_typical"
        :return: all folder dirs, only for folders and ignore files
        """
        file_list = []
        for root, dirs, files in os.walk(file_dir_):
            if root.count('/') == file_dir_.count('/') + 1:
                file_list.append(root)
        file_list.sort()
        return file_list

    def get_file_name(self, file_dir):
        file_list = []
        for root, dirs, files in os.walk(file_dir):
            # print(root)  # current path
            if root.count('/') == file_dir.count('/'):
                for name in files:
                    str = file_dir + '/' + name
                    file_list.append(str)
        file_list.sort()
        return file_list

    def extract_info(self, name_document):
        pos_name = name_document.find('name')
        pos_scale = name_document.find('scale')
        pos_time = name_document.find('ttt')
        pos_augtime = name_document.find('augtime')
        str_end = name_document.find('.png')
        str_obj_grasp_info = name_document[pos_name: pos_augtime - 1]
        str_obj_pixel_info = name_document[pos_name: str_end - 4]
        str_obj_name = name_document[pos_name + 5: pos_scale - 1]
        str_scale = name_document[pos_scale + 6: pos_time - 1]
        str_time = name_document[pos_time + 3: pos_augtime - 1]
        pos_root_start = name_document.find('/' + str_obj_name + '/')
        path_img_root = name_document[0: pos_root_start]
        return str_obj_grasp_info, str_obj_pixel_info, str_obj_name, str_scale, str_time, path_img_root

    def access_info(self, ind_grasp):
        msk_scenario = np.load(self.lst_dir_seg_img[ind_grasp])
        rgba_scenario = cv2.imread(self.lst_all_grasps[ind_grasp], cv2.IMREAD_UNCHANGED)
        rgb_scenario = cv2.cvtColor(rgba_scenario, cv2.COLOR_RGBA2RGB)
        rgb_scenario = rgb_scenario.transpose(2, 0, 1)

        dp_senario = np.load(self.lst_dir_dp_img[ind_grasp])
        grasp_info = pickle.load(open(self.lst_dir_grasp_info[ind_grasp], 'rb'))
        pixel_info = pickle.load(open(self.lst_dir_pixel_info[ind_grasp], 'rb'))

        indexes_bg = np.where(msk_scenario < 1)
        dp_senario[indexes_bg] = 1.0 * self.dp_bg
        rgb_scenario = rgb_scenario.astype('f')
        rgb_scenario[0, :, :][indexes_bg] = 255.0 * self.rgb_bg
        rgb_scenario[1, :, :][indexes_bg] = 255.0 * self.rgb_bg
        rgb_scenario[2, :, :][indexes_bg] = 255.0 * self.rgb_bg

        rgbd_img = np.vstack((rgb_scenario, dp_senario[np.newaxis, :, :]))
        rgb_norm = rgb_scenario / 255.0
        # rgb_norm = ((rgb_scenario / 255.0) - self.rgb_mean_arr) / self.rgb_std_arr
        # mean = np.mean(rgb_norm[0])
        # std = np.std(rgb_norm[0])
        rgbd_img_norm = np.vstack((rgb_norm, dp_senario[np.newaxis, :, :]))
        #
        coor_pixel = pixel_info[self.index_pixel_info]
        # dir_grasp = np.array([0.0, grasp_info['gripper_status'][self.index_dofs_info][20], 0.0]).astype('f')
        dir_grasp = (grasp_info['gripper_status'][self.index_dofs_info][20]).astype('f')
        tactile_info = grasp_info['info_biotac'][self.index_tactile_info]
        tactile_info = np.linalg.norm(tactile_info.reshape(5, 3), axis=1)
        dof_pose = grasp_info['gripper_status'][self.index_dofs_info][0:20]
        flg_success = np.array([grasp_info['grasp_status']['success']]).astype('f')
        gq = np.array([grasp_info['grasp_status']['quality_save']]).astype('f')
        return msk_scenario.astype('f'), rgbd_img.astype('f'), rgbd_img_norm.astype('f'), coor_pixel, dir_grasp, tactile_info, dof_pose, gq, flg_success

    def __len__(self):
        return self.amount

    def __getitem__(self, index):
        msk_scenario, rgbd_scenario, rgbd_img_norm, coor_pixel, dir_grasp, tactile_info, dof_pose, gq, flg_success = \
            self.access_info(index)
        return msk_scenario, rgbd_scenario, rgbd_img_norm, coor_pixel, dir_grasp, tactile_info, dof_pose, gq, flg_success

    def enumerate_all_grasps(self):
        cnt_max = float(self.amount)
        for i in range(self.amount):
            self.__getitem__(i)
            print("Enumerate progress: {:.2f}%".format(i/cnt_max * 100.0))

    def check_completeness(self):
        cnt_max = float(len(self.lst_all_grasps))
        for i in range(len(self.lst_all_grasps) - 1, -1, -1):
            str_obj_grasp_info, str_obj_pixel_info, str_obj_name, str_scale, str_time = self.extract_info(self.lst_all_grasps[i])
            dir_img_dp = self.dir_img_info + '/' + str_obj_pixel_info + '_dp.npy'
            dir_img_seg = self.dir_img_info + '/' + str_obj_pixel_info + '_seg.npy'
            dir_pixel_info = self.dir_img_info + '/' + str_obj_pixel_info + '.pickle'
            dir_grasp_info = self.dir_grasp_info + '/' + str_obj_grasp_info + '.pickle'

            if (os.path.exists(self.lst_all_grasps[i]) is False) or \
                    (os.path.exists(dir_img_dp) is False) or \
                    (os.path.exists(dir_img_seg) is False) or \
                    (os.path.exists(dir_pixel_info) is False) or \
                    (os.path.exists(dir_grasp_info) is False):
                del self.lst_all_grasps[i]
                print("delete incomplete data: rgb_img")
                continue
            #
            with open(dir_grasp_info, 'rb') as file:
                grasp_info = pickle.load(file)
            if 'info_biotac' in grasp_info:
                pass
            else:
                del self.lst_all_grasps[i]
                print("delete incomplete data: info_biotac")
                continue
            #
            with open(dir_pixel_info, 'rb') as file:
                pixel_info = pickle.load(file)
            if 'grasp_pixel' in pixel_info:
                pass
            else:
                del self.lst_all_grasps[i]
                print("delete incomplete data: grasp_pixel")
                continue
            #
            cnt_max = float(len(self.lst_all_grasps))
            print("Check completeness: {:.2f}%".format((len(self.lst_all_grasps) - i) / cnt_max * 100.0))

    def conditional_selection(self):
        cnt_max = float(len(self.lst_all_grasps))
        for i in range(len(self.lst_all_grasps) - 1, -1, -1):
            str_obj_grasp_info, str_obj_pixel_info, str_obj_name, str_scale, str_time, path_img_root = self.extract_info(self.lst_all_grasps[i])
            dir_img_dp = path_img_root+ '/detail/' + str_obj_pixel_info + '_dp.npy'
            dir_img_seg = path_img_root + '/detail/' + str_obj_pixel_info + '_seg.npy'
            dir_pixel_info = path_img_root + '/detail/' + str_obj_pixel_info + '.pickle'
            dir_grasp_info = self.dir_grasp_info + '/' + str_obj_grasp_info + '.pickle'

            with open(dir_grasp_info, 'rb') as file:
                grasp_info = pickle.load(file)
            if self.cfg['flg_success']:
                if not grasp_info['grasp_status']['success']:
                    del self.lst_all_grasps[i]
                    continue
            if grasp_info['grasp_status']['quality_save'] < self.cfg['threshold_gq']:
                del self.lst_all_grasps[i]
                continue
            if self.cfg['flg_single_tac']:
                if not('single_tac' in grasp_info['info_biotac']):
                    del self.lst_all_grasps[i]
                    continue
            #
            cnt_max = float(len(self.lst_all_grasps))
            print("Conditional selection: {:.2f}%".format((len(self.lst_all_grasps) - i) / cnt_max * 100.0))

    def view_grasp(self, ind_grasp):
        # msk_scenario = np.load(self.lst_dir_seg_img[ind_grasp])
        rgb_scenario = cv2.imread(self.lst_all_grasps[ind_grasp])
        # rgb_scenario = rgb_scenario.transpose(2, 0, 1)
        _, rgbd_scenario, _, coor_pixel, _, _, _, _ = self.access_info(i)
        rgb_scenario = rgbd_scenario[0:3, :, :]
        rgb_scenario = rgb_scenario.transpose(1, 2, 0).astype('uint8')
        dp_scenario = rgbd_scenario[3, :, :] * 255.0
        dp_scenario = dp_scenario.astype('uint8')
        #
        center_coordinates = (int(coor_pixel[0] * 240), int(coor_pixel[1] * 240))  # Center of the circle (x, y)
        radius = 3  # Radius of the circle
        color = (0, 0, 255)  # Color in BGR format (red in this case)
        thickness = 2  # Thickness of the circle outline (in pixels)
        cv2.circle(rgb_scenario, center_coordinates, radius, color, thickness)
        cv2.imshow('RGB Image', rgb_scenario)
        cv2.imshow('Depth Image', dp_scenario)
        # Wait for a key press and close the window when a key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #

    def view_segmentation(self, ind_grasp):
        msk_scenario = np.load(self.lst_dir_seg_img[ind_grasp])
        if np.max(msk_scenario) > 0:
            print(np.min(msk_scenario), np.max(msk_scenario))
            cnt_correct = 1
        else:
            cnt_correct = 0
        # print("cnt_correct", cnt_correct)
        return cnt_correct


