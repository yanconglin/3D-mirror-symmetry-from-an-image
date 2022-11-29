import json
import math
import os
import os.path as osp
import random
import sys
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import scipy.spatial.distance as scipy_spatial_dist
from sym.models.sphere.sphere_utils import gold_spiral_sampling_patch
from sym.config import CI, CM



class ShapeNetDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f"{rootdir}/{split}.txt", dtype=str)
        random.seed(0)
        random.shuffle(filelist)
        print(f"total n{split}:", len(filelist))
        filelist = [f for f in filelist if "03636649" not in f]
        if (
                split == "train"
                and hasattr(CI, "only_car_plane_chair")
                and CI.only_car_plane_chair
        ):
            filelist = [
                f
                for f in filelist
                if "02691156" in f or "02958343" in f or "03001627" in f
            ]
        if "train" in split: filelist = filelist[0:int(len(filelist) * CI.percentage)]
        if "valid" in split: filelist = filelist[0:100]
        if "test" in split: filelist = filelist[0:1000]
        self.filelist = [f"{rootdir}/{f}" for f in filelist]
        self.filelist2 = filelist
        self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prefix = self.filelist[idx]
        image = cv2.imread(f"{prefix}.png", -1).astype(np.float32) / 255.0
        image = np.rollaxis(image, 2).copy()
        depth = cv2.imread(f"{prefix}_depth0001.exr", -1).astype(np.float32)[:, :, 0:1]
        depth[depth > 20] = 0
        depth[depth < 0] = 0
        depth[depth != depth] = 0
        depth = np.rollaxis(depth, 2).copy()
        with open(f"{prefix}.json") as f:
            js = json.load(f)
        Rt, K_ = np.array(js["RT"]), np.array(js["K"])
        K = np.eye(4)
        K[:3, :3] = K_[np.ix_([0, 1, 3], [0, 1, 2])]

        oprefix = self.filelist2[idx]
        fname = np.zeros([60], dtype="uint8")
        fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

        depth_scale = 1 / abs(Rt[2][3]) if CM.detection.enabled else 1

        input_dict = {
            "fname": torch.tensor(fname).byte(),
            "image": torch.tensor(image).float(),
            "depth": torch.tensor(depth).float() * depth_scale,
            "K": torch.tensor(K).float(),
            "Rt": torch.tensor(Rt).float(),
        }
        if CM.detection.enabled:
            assert CM.num_sym == 1
            # # # GT # # #
            w0_ = LA.inv(Rt).T @ np.array([0, 1, 0, 0])
            # find plane normal s.t. w0 @ x + 1 = 0
            w0 = w0_[:3] / w0_[3]
            # normalize so that w[2]=1
            w0 = w0 / w0[2]
            pts0 = w0 / LA.norm(w0)
            S0 = K @ w2S(w0) @ LA.inv(K)
            P0 = w2P(w0) @ LA.inv(K)

            pts, Ss, Ps, ws = [],  [], [], []
            edge_index = []
            label = []
            for l in range(CM.n_levels):
                # print('level',  l, CM.theta[l], CM.num_nodes[l])
                if l==0:
                    pt_anchor = sample_sphere(pts0, 0, np.pi/2)
                    assert CM.theta[l]==90.0
                    pts_ = gold_spiral_sampling_patch(pt_anchor,  CM.theta[l]*np.pi/180., CM.num_nodes[l])
                else:
                    pts_ = gold_spiral_sampling_patch(pt_anchor,  CM.theta[l]*np.pi/180., CM.num_nodes[l])


                ws_ = pts_ / pts_[:, 2:3]
                Ss_ = [K @ w2S(w) @ LA.inv(K) for w in ws_]
                Ps_ = [w2P(w) @ LA.inv(K) for w in ws_]
                Ss_ = np.array(Ss_)
                Ps_ = np.array(Ps_)

                dist_cos_arc_ = cos_cdis(pts_, pts_, semi_sphere=True)
                # print('dist_cos_arc_', dist_cos_arc_.shape, dist_cos_arc_.min(axis=1))
                if CM.num_neighbors[l]<CM.num_nodes[l]:
                    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
                    topk_min_idx_ = np.argpartition(dist_cos_arc_, kth=CM.num_neighbors[l], axis=1)[:, 0:CM.num_neighbors[l]]
                    # print('topk_min_idx_', topk_min_idx_.shape)
                else:
                    topk_min_idx_ = np.argsort(dist_cos_arc_, axis=1)
                    # print('topk_min_idx_', topk_min_idx_.shape)
                neighbors_ = topk_min_idx_.flatten()
                centers_ = np.arange(0, CM.num_nodes[l]).repeat(CM.num_neighbors[l])
                edge_index_ = np.vstack([centers_, neighbors_])
                # if l>0: edge_index_ = np.vstack([centers_, neighbors_]) + sum(CM.num_nodes[0:l])

                dist_cos_arc_ = cos_cdis(pts0[None], pts_, semi_sphere=True)
                w0_idx_ = np.argmin(dist_cos_arc_.flatten())
                label_ = np.zeros((CM.num_nodes[l]), dtype=np.float32)
                label_[w0_idx_] = 1.0
                pt_anchor = pts_[w0_idx_]

                pts.append(pts_)
                ws.append(ws_)
                Ps.append(Ps_)
                Ss.append(Ss_)
                edge_index.append(edge_index_)
                label.append(label_)

            pts = np.concatenate(pts, axis=0)
            ws = np.concatenate(ws, axis=0)
            Ps = np.concatenate(Ps, axis=0)
            Ss = np.concatenate(Ss, axis=0)
            edge_index = np.concatenate(edge_index, axis=1)
            label = np.concatenate(label, axis=0)
            # print('pts, ws, Ps, Ss, edge_index, label', pts.shape, ws.shape, Ps.shape, Ss.shape, edge_index.shape, label.shape)

            input_dict["pts"] = torch.tensor(pts).float()
            input_dict["ws"] = torch.tensor(ws).float()
            input_dict["Ss"] = torch.tensor(Ss).float()
            input_dict["Ps"] = torch.tensor(Ps).float()
            input_dict["edge_index"] = torch.tensor(edge_index).long()

            input_dict["w0"] = torch.tensor(w0).float()
            input_dict["pts0"] = torch.tensor(pts0).float()
            input_dict["S0"] = torch.tensor(S0).float()
            input_dict["P0"] = torch.tensor(P0).float()
            input_dict["label"] = torch.tensor(label).float()

        return input_dict


class ShapeNetDataset_test(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f"{rootdir}/{split}.txt", dtype=str)
        random.seed(0)
        random.shuffle(filelist)
        print(f"total n{split}:", len(filelist))
        filelist = [f for f in filelist if "03636649" not in f]
        if (
                split == "train"
                and hasattr(CI, "only_car_plane_chair")
                and CI.only_car_plane_chair
        ):
            filelist = [
                f
                for f in filelist
                if "02691156" in f or "02958343" in f or "03001627" in f
            ]

        assert "test" in split
        filelist = filelist[0:1000]
        self.filelist = [f"{rootdir}/{f}" for f in filelist]
        self.filelist2 = filelist
        self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prefix = self.filelist[idx]
        image = cv2.imread(f"{prefix}.png", -1).astype(np.float32) / 255.0
        image = np.rollaxis(image, 2).copy()
        depth = cv2.imread(f"{prefix}_depth0001.exr", -1).astype(np.float32)[:, :, 0:1]
        depth[depth > 20] = 0
        depth[depth < 0] = 0
        depth[depth != depth] = 0
        depth = np.rollaxis(depth, 2).copy()
        with open(f"{prefix}.json") as f:
            js = json.load(f)
        Rt, K_ = np.array(js["RT"]), np.array(js["K"])
        K = np.eye(4)
        K[:3, :3] = K_[np.ix_([0, 1, 3], [0, 1, 2])]

        oprefix = self.filelist2[idx]
        fname = np.zeros([60], dtype="uint8")
        fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

        depth_scale = 1 / abs(Rt[2][3]) if CM.detection.enabled else 1

        input_dict = {
            "fname": torch.tensor(fname).byte(),
            "image": torch.tensor(image).float(),
            "depth": torch.tensor(depth).float() * depth_scale,
            "K": torch.tensor(K).float(),
            "Rt": torch.tensor(Rt).float(),
        }
        if CM.detection.enabled:
            assert CM.num_sym == 1
            # # # GT # # #
            w0_ = LA.inv(Rt).T @ np.array([0, 1, 0, 0])
            # find plane normal s.t. w0 @ x + 1 = 0
            w0 = w0_[:3] / w0_[3]
            # normalize so that w[2]=1
            w0 = w0 / w0[2]
            pts0 = w0 / LA.norm(w0)
            S0 = K @ w2S(w0) @ LA.inv(K)
            P0 = w2P(w0) @ LA.inv(K)

            input_dict["w0"] = torch.tensor(w0).float()
            input_dict["pts0"] = torch.tensor(pts0).float()
            input_dict["S0"] = torch.tensor(S0).float()
            input_dict["P0"] = torch.tensor(P0).float()

            ############# inference ################################
            pts = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=CM.theta[0]*np.pi/180., num_pts=CM.num_nodes[0])
            ws = pts / pts[:, 2:3]
            Ss = [K @ w2S(w) @ LA.inv(K) for w in ws]
            Ps = [w2P(w) @ LA.inv(K) for w in ws]
            Ss = np.array(Ss)
            Ps = np.array(Ps)
            dist_cos_arc = cos_cdis(pts, pts, semi_sphere=True)
            # print('dist_cos_arc_', dist_cos_arc_.shape, dist_cos_arc_.min(axis=1))
            if CM.num_neighbors[0]<CM.num_nodes[0]:
                # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
                topk_min_idx = np.argpartition(dist_cos_arc, kth=CM.num_neighbors[0], axis=1)[:, 0:CM.num_neighbors[0]]
                # print('topk_min_idx', topk_min_idx.shape)
            else:
                topk_min_idx = np.argsort(dist_cos_arc, axis=1)
                # print('topk_min_idx', topk_min_idx.shape)

            neighbors = topk_min_idx.flatten()
            centers = np.arange(0, CM.num_nodes[0]).repeat(CM.num_neighbors[0])
            edge_index = np.vstack([centers, neighbors])
            # if l>0: edge_index_ = np.vstack([centers_, neighbors_]) + sum(CM.num_nodes[0:l])

            input_dict["pts"] = torch.tensor(pts).float()
            input_dict["ws"] = torch.tensor(ws).float()
            input_dict["Ss"] = torch.tensor(Ss).float()
            input_dict["Ps"] = torch.tensor(Ps).float()
            input_dict["edge_index"] = torch.tensor(edge_index).long()
        return input_dict


class Pix3dDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        with open(f"{rootdir}/pix3d_info.json", "r") as fin:
            data_lists = json.load(fin)
        data_valid = set(np.loadtxt(f"{rootdir}/pix3d-valid.txt", dtype=str))

        data_lists = [
            d
            for d in data_lists
            if not d["truncated"] and not d["occluded"] and d["img"][:-4] in data_valid
        ]
        random.seed(0)
        random.shuffle(data_lists)
        print(f"total n{split}:", len(data_lists))
        if self.split == "train":
            data_lists = [data_lists[i] for i in range(len(data_lists)) if i % 10 != 0]
            data_lists = data_lists[0:int(len(data_lists) * CI.percentage)]
            self.size = len(data_lists) * 2
        elif self.split == "valid":
            data_lists = [data_lists[i] for i in range(len(data_lists)) if i % 10 == 0]
            # data_lists = data_lists[0:100]
            self.size = len(data_lists)
        else:
            raise NotImplementedError

        self.data_lists = data_lists

        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        do_flip = False
        if self.split == "train":
            do_flip = idx % 2 == 1
            idx //= 2

        data_item = self.data_lists[idx]
        fimage = osp.join(self.rootdir, data_item["img"])
        fdepth = osp.join(self.rootdir, data_item["depth"])
        fmask = osp.join(self.rootdir, data_item["mask"])

        image = cv2.imread(fimage, -1).astype(np.float32) / 255.0
        image = np.rollaxis(image, 2).copy()
        mask = cv2.imread(fmask).astype(np.float32)[None, :, :, 0] / 255.0
        image = np.concatenate([image, mask])
        depth = cv2.imread(fdepth, -1).astype(np.float32)[:, :, 0:1]
        depth[depth > 20] = 0
        depth[depth < 0] = 0
        depth[depth != depth] = 0
        depth = np.rollaxis(depth, 2).copy()
        # print('image, mask, depth', image.shape, mask.shape, depth.shape)
        Rt = np.array(data_item["Rt"])
        K = np.array(data_item["K"])        
        K[:3] *= -1 
        # print('Rt', Rt.shape, Rt)
        # print('K', K.shape, K)

        if do_flip:
            K = np.diagflat([-1, 1, 1, 1]) @ K
            image = image[:, :, ::-1].copy()
            depth = depth[:, :, ::-1].copy()

        if CM.detection.enabled:
            assert CM.num_sym == 1
            # # # GT # # #
            w0_ = LA.inv(Rt).T @ np.array([1, 0, 0, 0])
            # find plane normal s.t. w0 @ x + 1 = 0
            w0 = w0_[:3] / w0_[3]
            # normalize so that w[2]=1
            w0 = w0 / w0[2]
            pts0 = w0 / LA.norm(w0)
            S0 = K @ w2S(w0) @ LA.inv(K)
            P0 = w2P(w0) @ LA.inv(K)

            E0 = w2E(w0)  # [3,3]
            F0 = LA.inv(K).T @ E0 @ LA.inv(K)  # [4,4]
            F0 = F0[0:3, 0:3]

            depth_scale = 1 / abs(Rt[2][3])
            # depth_scale = 1 / (w0 @ Rt[:3, 3])
            # print('depth scale', depth_scale)

            pts, Ss, Ps, Fs, ws = [],  [], [], [], []
            edge_index = []
            label = []
            for l in range(CM.n_levels):
                # print('level',  l, CM.theta[l], CM.num_nodes[l])
                if l==0:
                    pt_anchor = sample_sphere(pts0, 0, np.pi/2)
                    assert CM.theta[l]==90.0
                    pts_ = gold_spiral_sampling_patch(pt_anchor,  CM.theta[l]*np.pi/180., CM.num_nodes[l])
                else:
                    pts_ = gold_spiral_sampling_patch(pt_anchor,  CM.theta[l]*np.pi/180., CM.num_nodes[l])

                ws_ = pts_ / pts_[:, 2:3]
                Ss_ = [K @ w2S(w) @ LA.inv(K) for w in ws_]
                Ps_ = [w2P(w) @ LA.inv(K) for w in ws_]
                Fs_ = [LA.inv(K).T @ w2E(w) @ LA.inv(K) for w in ws_]
                Ss_ = np.array(Ss_)
                Ps_ = np.array(Ps_)
                Fs_ = np.array(Fs_)
                Fs_ = Fs_[:, 0:3, 0:3]

                dist_cos_arc_ = cos_cdis(pts_, pts_, semi_sphere=True)
                if CM.num_neighbors[l]<CM.num_nodes[l]:
                    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
                    topk_min_idx_ = np.argpartition(dist_cos_arc_, kth=CM.num_neighbors[l], axis=1)[:, 0:CM.num_neighbors[l]]
                    # print('topk_min_idx_', topk_min_idx_.shape)
                else:
                    topk_min_idx_ = np.argsort(dist_cos_arc_, axis=1)
                    # print('topk_min_idx_', topk_min_idx_.shape)
                neighbors_ = topk_min_idx_.flatten()
                centers_ = np.arange(0, CM.num_nodes[l]).repeat(CM.num_neighbors[l])
                edge_index_ = np.vstack([centers_, neighbors_])
                # if l>0: edge_index_ = np.vstack([centers_, neighbors_]) + sum(CM.num_nodes[0:l])

                dist_cos_arc_ = cos_cdis(pts0[None], pts_, semi_sphere=True)
                w0_idx_ = np.argmin(dist_cos_arc_.flatten())
                # w0_err_ = np.min(dist_cos_arc_.flatten())
                # print('w0_err_', w0_err_*180./np.pi)
                label_ = np.zeros((CM.num_nodes[l]), dtype=np.float32)
                label_[w0_idx_] = 1.0
                pt_anchor = pts_[w0_idx_]

                pts.append(pts_)
                ws.append(ws_)
                Ps.append(Ps_)
                Ss.append(Ss_)
                Fs.append(Fs_)
                edge_index.append(edge_index_)
                label.append(label_)

            pts = np.concatenate(pts, axis=0)
            ws = np.concatenate(ws, axis=0)
            Ps = np.concatenate(Ps, axis=0)
            Ss = np.concatenate(Ss, axis=0)
            Fs = np.concatenate(Fs, axis=0)
            edge_index = np.concatenate(edge_index, axis=1)
            label = np.concatenate(label, axis=0)
            # print('pts, ws, Ps, Ss, edge_index, label', pts.shape, ws.shape, Ps.shape, Ss.shape, edge_index.shape, label.shape)
            oprefix = data_item["mask"][5:-4]
            fname = np.zeros([60], dtype="uint8")
            fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

            input_dict = {}
            input_dict["fname"] = torch.tensor(fname).byte()
            input_dict["image"] = torch.tensor(image).float()
            input_dict["depth"] = torch.tensor(depth).float() * depth_scale
            input_dict["S0"] = torch.tensor(S0).float()
            input_dict["K"] = torch.tensor(K).float()
            input_dict["Rt"] = torch.tensor(Rt).float()

            input_dict["pts"] = torch.tensor(pts).float()
            input_dict["ws"] = torch.tensor(ws).float()
            input_dict["Ss"] = torch.tensor(Ss).float()
            input_dict["Ps"] = torch.tensor(Ps).float()
            input_dict["Fs"] = torch.tensor(Fs).float()
            input_dict["edge_index"] = torch.tensor(edge_index).long()

            input_dict["w0"] = torch.tensor(w0).float()
            input_dict["pts0"] = torch.tensor(pts0).float()
            input_dict["S0"] = torch.tensor(S0).float()
            input_dict["P0"] = torch.tensor(P0).float()
            input_dict["F0"] = torch.tensor(F0).float()
            input_dict["label"] = torch.tensor(label).float()

        return input_dict



class Pix3dDataset_test(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        with open(f"{rootdir}/pix3d_info.json", "r") as fin:
            data_lists = json.load(fin)
        data_valid = set(np.loadtxt(f"{rootdir}/pix3d-valid.txt", dtype=str))

        data_lists = [
            d
            for d in data_lists
            if not d["truncated"] and not d["occluded"] and d["img"][:-4] in data_valid
        ]
        random.seed(0)
        random.shuffle(data_lists)
        if "test" in self.split:
            data_lists = [data_lists[i] for i in range(len(data_lists)) if i % 10 == 0]
            # data_lists = data_lists[0:100]
            self.size = len(data_lists)
        else:
            raise NotImplementedError

        self.data_lists = data_lists

        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        do_flip = False
        if self.split == "train":
            do_flip = idx % 2 == 1
            idx //= 2

        data_item = self.data_lists[idx]
        # print('data_item',data_item)
        fimage = osp.join(self.rootdir, data_item["img"])
        fdepth = osp.join(self.rootdir, data_item["depth"])
        fmask = osp.join(self.rootdir, data_item["mask"])

        image = cv2.imread(fimage, -1).astype(np.float32) / 255.0
        image = np.rollaxis(image, 2).copy()
        mask = cv2.imread(fmask).astype(np.float32)[None, :, :, 0] / 255.0
        image = np.concatenate([image, mask])
        depth = cv2.imread(fdepth, -1).astype(np.float32)[:, :, 0:1]
        depth[depth > 20] = 0
        depth[depth < 0] = 0
        depth[depth != depth] = 0
        depth = np.rollaxis(depth, 2).copy()
        # print('image, mask, depth', image.shape, mask.shape, depth.shape)
        Rt = np.array(data_item["Rt"])
        K = np.array(data_item["K"])
        K[:3] *= -1  # do not understand what this does in the original code
        # print('Rt', Rt.shape, Rt)
        # print('K', K.shape, K)

        if do_flip:
            K = np.diagflat([-1, 1, 1, 1]) @ K
            image = image[:, :, ::-1].copy()
            depth = depth[:, :, ::-1].copy()

        if CM.detection.enabled:
            assert CM.num_sym == 1
            # # # GT # # #
            w0_ = LA.inv(Rt).T @ np.array([1, 0, 0, 0])
            # find plane normal s.t. w0 @ x + 1 = 0
            w0 = w0_[:3] / w0_[3]
            # normalize so that w[2]=1
            w0 = w0 / w0[2]
            pts0 = w0 / LA.norm(w0)
            S0 = K @ w2S(w0) @ LA.inv(K)
            P0 = w2P(w0) @ LA.inv(K)

            E0 = w2E(w0)  # [3,3]
            F0 = LA.inv(K).T @ E0 @ LA.inv(K)  # [4,4]
            F0 = F0[0:3, 0:3]

            depth_scale = 1 / abs(Rt[2][3])
            # depth_scale = 1 / (w0 @ Rt[:3, 3])
            # print('depth scale', depth_scale)
            ############# inference ################################
            pts = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=CM.theta[0]*np.pi/180., num_pts=CM.num_nodes[0])
            ws = pts / pts[:, 2:3]
            Ss = [K @ w2S(w) @ LA.inv(K) for w in ws]
            Ps = [w2P(w) @ LA.inv(K) for w in ws]
            Ss = np.array(Ss)
            Ps = np.array(Ps)
            dist_cos_arc = cos_cdis(pts, pts, semi_sphere=True)
            # print('dist_cos_arc_', dist_cos_arc_.shape, dist_cos_arc_.min(axis=1))
            if CM.num_neighbors[0]<CM.num_nodes[0]:
                # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
                topk_min_idx = np.argpartition(dist_cos_arc, kth=CM.num_neighbors[0], axis=1)[:, 0:CM.num_neighbors[0]]
                # print('topk_min_idx', topk_min_idx.shape)
            else:
                topk_min_idx = np.argsort(dist_cos_arc, axis=1)
                # print('topk_min_idx', topk_min_idx.shape)

            neighbors = topk_min_idx.flatten()
            centers = np.arange(0, CM.num_nodes[0]).repeat(CM.num_neighbors[0])
            edge_index = np.vstack([centers, neighbors])
            # if l>0: edge_index_ = np.vstack([centers_, neighbors_]) + sum(CM.num_nodes[0:l])

            # print('pts, ws, Ps, Ss, edge_index, label', pts.shape, ws.shape, Ps.shape, Ss.shape, edge_index.shape, label.shape)
            oprefix = data_item["mask"][5:-4]
            fname = np.zeros([60], dtype="uint8")
            fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

            input_dict = {}
            input_dict["fname"] = torch.tensor(fname).byte()
            input_dict["image"] = torch.tensor(image).float()
            input_dict["depth"] = torch.tensor(depth).float() * depth_scale
            input_dict["S0"] = torch.tensor(S0).float()
            input_dict["K"] = torch.tensor(K).float()
            input_dict["Rt"] = torch.tensor(Rt).float()

            input_dict["pts"] = torch.tensor(pts).float()
            input_dict["ws"] = torch.tensor(ws).float()
            input_dict["Ss"] = torch.tensor(Ss).float()
            input_dict["Ps"] = torch.tensor(Ps).float()
            input_dict["edge_index"] = torch.tensor(edge_index).long()

            input_dict["w0"] = torch.tensor(w0).float()
            input_dict["pts0"] = torch.tensor(pts0).float()
            input_dict["S0"] = torch.tensor(S0).float()
            input_dict["P0"] = torch.tensor(P0).float()
            input_dict["F0"] = torch.tensor(F0).float()
            input_dict["idx"] = idx
           
        return input_dict




def sample_sphere_test(v, alpha, num_pts):
    def orth(v):
        x, y, z = v
        o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
        o /= LA.norm(o)
        return o

    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.linspace(1, num_pts, num_pts)
    phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    w = (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T
    return w


def sample_sphere(v, theta0, theta1):
    def orth(v):
        x, y, z = v
        o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
        o /= LA.norm(o)
        return o

    costheta = random.uniform(math.cos(theta1), math.cos(theta0))
    phi = random.random() * math.pi * 2
    v1 = orth(v)
    v2 = np.cross(v, v1)
    r = math.sqrt(1 - costheta ** 2)
    w = v * costheta + r * (v1 * math.cos(phi) + v2 * math.sin(phi))
    return w / LA.norm(w)


def sample_reflection(v, alpha, num_points):
    ws = sample_sphere(v, alpha, num_points)
    ws /= ws[:, 2:]
    return ws


def w2S(w):
    S = np.eye(4)
    S[:3, :3] = np.eye(3) - 2 * np.outer(w, w) / np.sum(w ** 2)
    S[:3, 3] = -2 * w / np.sum(w ** 2)
    return S


def to_label(w, w0):
    theta = math.acos(np.clip(abs(w @ w0) / LA.norm(w) / LA.norm(w0), -1, 1))
    return [theta < theta0 for theta0 in CM.detection.theta]


def w2P(w):
    # p = np.eye(4)
    # p[:3, :3] = np.eye(3) - np.outer(w, w) / np.sum(w ** 2)
    # p[:3, 3] = - w / np.sum(w ** 2)
    P = np.zeros((4))
    P[:3] = w
    P[3] = 1.0
    return P / LA.norm(w)


def w2E(w):
    R = np.eye(3) - 2 * np.outer(w, w) / np.sum(w ** 2)
    tx, ty, tz = -2 * w / np.sum(w ** 2)
    T = np.array(
        [[ 0, -tz, ty,],
         [ tz, 0, -tx,],
         [ -ty, tx,  0]]
        )
    # print('R', R.shape, R)
    # print('t', tx, ty, tz)
    # print('T', T.shape, T)
    E = np.zeros((4,4))
    E[0:3, 0:3] = T @ R
    E[3,3] = 1.0
    # print('E', E)
    return E


def cos_cdis(x, y, semi_sphere=True):
    # input: x: mxp, y: nxp
    # output: y, mxn
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    ### compute cosine distance
    ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
    dist_cos = scipy_spatial_dist.cdist(x, y, 'cosine')  # num_nodes_ x num_nodes
    # ### map to: same 1, opposite -1, orthorgonal 0, dist = AB/(|A||B|)
    dist_cos *= -1.0
    dist_cos += 1.0

    if semi_sphere is True: dist_cos = np.abs(dist_cos)
    dist_cos_arc = np.arccos(dist_cos)
    return dist_cos_arc


