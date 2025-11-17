import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
from torch_points3d.modules.KPConv.plyutils import read_ply
from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)




TreeInstanceLoader_NUM_CLASSES = 2

INV_OBJECT_LABEL = {
    0: "non-tree",
    1: "tree",
}


OBJECT_COLOR = np.asarray(
    [
        [179, 116, 81],
        [77, 174, 84],
        [0, 0, 0],
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

def object_name_to_label(object_class):
    """convert from object name in NPPM3D to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["unclassified"])
    return object_label

def read_tree_instance_loader_format(train_file, label_out=True, verbose=False, debug=False):
    """extract data from a tree_instance_loader file"""
    raw_path = train_file
    data = read_ply(raw_path)
    xyz = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
    if not label_out:
        return xyz
    semantic_labels = data['semantic_seg'].astype(np.int64)-1
    instance_labels = data['treeID'].astype(np.int64)+1
    return (
        torch.from_numpy(xyz),
        torch.from_numpy(semantic_labels),
        torch.from_numpy(instance_labels),
    )


def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, "TreeInstanceLoader")
    PlyData([el], byte_order=">").write(file)
    
def to_eval_ply(pos, pre_label, gt, file):
    assert len(pre_label.shape) == 1
    assert len(gt.shape) == 1
    assert pos.shape[0] == pre_label.shape[0]
    assert pos.shape[0] == gt.shape[0]
    pos = np.asarray(pos)
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "u16"), ("gt", "u16")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["preds"] = np.asarray(pre_label)
    ply_array["gt"] = np.asarray(gt)
    PlyData.write(file)
    
def to_ins_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    max_instance = np.max(np.asarray(label)).astype(np.int32)+1
    rd_colors = np.random.randint(255, size=(max_instance,3), dtype=np.uint8)
    colors = rd_colors[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    PlyData.write(file)





class TreeInstanceLoaderOriginalFused(InMemoryDataset):
    """ Original TreeInstanceLoader dataset. Each area is loaded individually and can be processed using a pre_collate transform.
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single area by default.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    forest regions: list of str
        @TreeInstanceLoader: specifies from which forest region(s) data files should be used for training and validation, [] means taking data files from all forest regions
    test_area: list
        @TreeInstanceLoader: during training/running train.py: [] means taking all specified test files (i.e. all files with name ending in "test" for testing, otherwise list of ints indexing into which of these specified test files to use
        @TreeInstanceLoader: during evaluation/running eval.py: paths to files to test model on
    split: str
        can be one of train, trainval, val or test
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    num_classes = TreeInstanceLoader_NUM_CLASSES

    def __init__(
        self,
        root,
        grid_size,
        forest_regions=[],
        test_area=[],
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.forest_regions = forest_regions
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self.grid_size = grid_size
     
        super(TreeInstanceLoaderOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            
            if split == "train":
                path = self.processed_paths[0]
            elif split == "val":
                path = self.processed_paths[1]
            elif split == "test":
                path = self.processed_paths[2]
            elif split == "trainval":
                path = self.processed_paths[3]
            else:
                raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))
            self._load_data(path)

            if split == "test":
                if self.test_area == []:
                    "PROBLEM: In branch split==test even though test_area was not initialized yet"
                self.raw_test_data = [torch.load(self.raw_areas_paths[test_area_i]) for test_area_i in self.test_area]

        else:
            self.process_test(test_area)
            path = self.processed_path
            self._load_data(path)
            self.raw_test_data = [torch.load(raw_area_path) for raw_area_path in self.raw_areas_paths]

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        """returns list of paths to the .ply raw data files we use"""
        if self.forest_regions == []:
            return glob.glob(self.raw_dir + '/**/*.ply', recursive=True)
        else:
            raw_files_list = []
            for region in self.forest_regions:
                raw_files_list += glob.glob(self.raw_dir + "/" + region + "/*.ply", recursive=False)
            return raw_files_list

    @property
    def processed_dir(self):
        """returns path to the directory which contains the processed data files,
               e.g. path/to/project/OutdoorPanopticSeg_V2/data/tree_instance_loaderfused/processed_0.2"""
        processed_dir_prefix = 'processed_' + str(self.grid_size)
        if self.forest_regions != []:
            processed_dir_prefix += "_" + str(self.forest_regions)

        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            return osp.join(self.root, processed_dir_prefix)
        else:
            return osp.join(self.root, processed_dir_prefix+'_test')

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        """returns list of paths to .pt files saved in self.processed_dir and created from the .ply raw data files"""
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            if not hasattr(self, "num_datafiles"):
                self.num_datafiles = len(self.raw_file_names)
            return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(self.num_datafiles)]
        else:
            return [os.path.join(self.processed_dir, 'raw_area_'+os.path.split(f)[-1].split('.')[0]+'.pt') for f in self.test_area]


    @property
    def processed_file_names(self):
        """return list of paths to all kinds of files in the processed directory"""
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            return (
            ["{}.pt".format(s) for s in ["train", "val", "test", "trainval"]]
            + self.raw_areas_paths
            + [self.pre_processed_path]
        )
        else:
            return ['processed_'+os.path.split(f)[-1].split('.')[0]+'.pt' for f in self.test_area]

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    @property
    def num_features(self):
        feats = self[0].x
        if feats is not None:
            return feats.shape[-1]
        return 0

    def process(self):
        """Takes the given .ply files, processes them and saves the newly created files in self.processed_dir.
        This method is used during training/running train.py."""

        if not os.path.exists(self.pre_processed_path):
            input_ply_files = self.raw_file_names

            data_list = [[] for _ in range(len(input_ply_files))]
            for area_num, file_path in enumerate(input_ply_files):
                area_name = os.path.split(file_path)[-1]
                xyz, semantic_labels, instance_labels = read_tree_instance_loader_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug
                )

                data = Data(pos=xyz, y=semantic_labels)
                data.validation_set = False
                data.test_set = False
                if area_name[-7:-4]=="val":
                    data.validation_set = True
                elif area_name[-8:-4]=="test":
                    data.test_set = True
                    self.test_area.append(area_num)

                if self.keep_instance:
                    data.instance_labels = instance_labels

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                print("area_num:")
                print(area_num)
                print("data:")
                print(data)
                data_list[area_num].append(data)
            print("data_list")
            print(data_list)
            raw_areas = cT.PointCloudFusion()(data_list)
            print("raw_areas")
            print(raw_areas)
            for i, area in enumerate(raw_areas):
                torch.save(area, self.raw_areas_paths[i])


            for area_datas in data_list:
                if self.pre_transform is not None:
                    area_datas = self.pre_transform(area_datas)
            torch.save(data_list, self.pre_processed_path)
        else:
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        train_data_list = []
        val_data_list = []
        trainval_data_list = []
        test_data_list = []
        for list in data_list:
            for data in list:
                validation_set = data.validation_set
                del data.validation_set
                test_set = data.test_set
                del data.test_set
                if validation_set:
                    val_data_list.append(data)
                elif test_set:
                    test_data_list.append(data)
                else:
                    train_data_list.append(data)
        trainval_data_list = val_data_list + train_data_list

        print("train_data_list:")
        print(train_data_list)
        print("test_data_list:")
        print(test_data_list)
        print("val_data_list:")
        print(val_data_list)
        print("trainval_data_list:")
        print(trainval_data_list)
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)

        self._save_data(train_data_list, val_data_list, test_data_list, trainval_data_list)

    def process_test(self, test_area):
        """Takes the .ply files specified in data:fold: [...] in the file conf/eval.yaml as test files, processes them and saves the newly created files in self.processed_dir.
        This method is used during evaluation/running eval.py.
        @TreeInstanceLoader: Method is extended so that we can evaluate on more than one test file."""

        self.processed_path = osp.join(self.processed_dir,'processed_test.pt')


        test_data_list = []
        for i, file_path in enumerate(test_area):
            area_name = os.path.split(file_path)[-1]
            processed_area_path = osp.join(self.processed_dir, self.processed_file_names[i])
            if not os.path.exists(processed_area_path):
                xyz, semantic_labels, instance_labels = read_tree_instance_loader_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug
                )
                data = Data(pos=xyz, y=semantic_labels)
                if self.keep_instance:
                    data.instance_labels = instance_labels
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                print("area_name:")
                print(area_name)
                print("data:")
                print(data)
                test_data_list.append([data])
                torch.save(data, processed_area_path)
            else:
                data = torch.load(processed_area_path)
                test_data_list.append([data])

        raw_areas = cT.PointCloudFusion()(test_data_list)
        for i, area in enumerate(raw_areas):
            torch.save(area, self.raw_areas_paths[i])

        if self.debug:
            return

        print("test_data_list:")
        print(test_data_list)

        new_test_data_list = [listelem[0] for listelem in test_data_list]
        test_data_list = new_test_data_list
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            test_data_list = self.pre_collate_transform(test_data_list)
        torch.save(test_data_list, self.processed_path)


    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(self.collate(trainval_data_list), self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


class TreeInstanceLoaderSphere(TreeInstanceLoaderOriginalFused):
    """ Small variation of TreeInstanceLoaderOriginalFused that allows random sampling of spheres
    within an Area during training and validation. Spheres have a radius of 8m. If sample_per_epoch is not specified, spheres
    are taken on a 0.16m grid.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 4 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=8, grid_size=0.12, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=grid_size, mode="last")
        super().__init__(root, grid_size, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def len(self):
        return len(self)

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def process(self):
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            super().process()
        else:
            super().process_test(self.test_area)

    def download(self):
        super().download()

    def _get_random(self):
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.SphereSampling.KDTREE_KEY
                )
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class TreeInstanceLoaderCylinder(TreeInstanceLoaderSphere):
    def _get_random(self):
        while True:
            chosen_label = np.random.choice(self._labels, p=self._label_counts)
            valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
            centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
            centre = valid_centres[centre_idx]
            area_data = self._datas[centre[3].int()]
            cylinder_sampler = cT.CylinderSampling(self._radius, centre[:3], align_origin=False)
            cylinder_area = cylinder_sampler(area_data)
            if (torch.any(cylinder_area.y==1)).item():
                return cylinder_area

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.CylinderSampling.KDTREE_KEY
                )
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                setattr(data, cT.CylinderSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridCylinderSampling(self._radius, self._radius, center=False)
            self._test_spheres = []
            self._num_spheres = []
            for i, data in enumerate(self._datas):
                test_spheres = grid_sampler(data)
                self._test_spheres = self._test_spheres + test_spheres
                self._num_spheres = self._num_spheres + [len(test_spheres)]


class TreeInstanceLoaderFusedDataset(BaseDataset):
    """ Wrapper around NPM3DSphere that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = TreeInstanceLoaderCylinder if sampling_format == "cylinder" else TreeInstanceLoaderSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data
        
    @property
    def test_data_spheres(self):
        return self.test_dataset[0]._test_spheres

    @property
    def test_data_num_spheres(self):
        return self.test_dataset[0]._num_spheres

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save NPM3D predictions to disk using NPM3D color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.instance_seg_tracker import SegmentationTracker
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
