import os
import cv2
import torch
import numpy as np
from tasks.aneurysm.datasets.base_dataset import BaseDataset
from utils.tools.logger import Logger as Log
import os.path as osp
from abc import ABCMeta
import copy
import random


class ANEURYSM_CLS(BaseDataset, metaclass=ABCMeta):
    CLASSES = ['fp', 'aneurysm']

    def __init__(self, cfg, stage):
        self.cfg = cfg
        self.pipeline = self.init_pipline(stage)
        super(ANEURYSM_CLS, self).__init__(cfg, stage)

    def load_annotations(self):
        ann_file = os.path.join(self.data_prefix, self.ann_file)
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split() for line in lines]  # add support for [' ' '\t']
        self.eval_patch_info_list = lines
        data_infos = []
        for index in range(len(lines)):
            path, label = lines[index]
            filename = path
            data_info = {}
            data_info['img_info'] = {'filename': filename}
            data_info['img_prefix'] = self.data_prefix
            data_info['gt_label'] = np.array(float(label), dtype=np.int64)
            data_infos.append(data_info)

        return data_infos

    def init_pipline(self, stage):
        if stage in ['train', 'val', 'test']:
            transforms = [
                LoadImageFromFile(new_shape=(36,36,36), to_float32=True),
                PhotoMetricDistortionMultipleSlices(brightness_delta=32, contrast_range=(0.8, 1.2)),
                TensorNormCropRotateFlip(crop_size=32, move=2, train=True),
                ToTensor(keys=['img']),
                ToTensor(keys=['gt_label']),
                Collect(keys=['img', 'gt_label']),
            ]
        # elif stage in ['test']:
        #     transforms = [
        #         LoadImageFromFile(new_shape=(36,36,36), to_float32=True),
        #         TensorNormCropRotateFlip(crop_size=32, move=2, train=False),
        #         ToTensor(keys=['img']),
        #         Collect(keys=['img',]),
        #     ]
        else:
            raise Exception(f'unknown stage= {stage} in pipline init.')

        return Compose(transforms)

    def train_init(self):
        self.ann_file = self.cfg.TRAIN.DATA.TRAIN_LIST
        self.data_prefix = self.cfg.TRAIN.DATA.TRAIN_IMAGE_DIR
        self.data_infos = self.load_annotations()
        self.num = len(self.data_infos)
        # self.gt = self.para_dict.get("gt", None)
        # self.flip = False

    def val_init(self):
        self.ann_file = self.cfg.TRAIN.DATA.VAL_LIST
        self.data_prefix = self.cfg.TRAIN.DATA.VALID_IMAGE_DIR
        self.data_infos = self.load_annotations()
        self.num = len(self.data_infos)

    def test_init(self):
        self.ann_file = self.cfg.TEST.DATA.TEST_LIST
        self.data_prefix = self.cfg.TEST.DATA.TEST_IMAGE_DIR
        self.data_infos = self.load_annotations()
        self.num = len(self.data_infos)

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        # if classes is None:
        #     return cls.CLASSES

        return cls.CLASSES

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [int(self.data_infos[idx]['gt_label'])]


# used tranforms
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self,
                 to_float32=False,
                 new_shape=None,):
        self.to_float32 = to_float32
        self.new_shape = new_shape

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = cv2.imread(filename, flags=0) # 0: gray; <0: origin with alpha(aRGB); >0: origin RGB without aplha
        if self.new_shape is not None:
            img = img.reshape(self.new_shape)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, ',)
        return repr_str


class PhotoMetricDistortionMultipleSlices(object):
    """Apply photometric distortion to multiple slice image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5)):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range

    def __call__(self, results):
        img = results['img']
        # random brightness, delta in range [x, y]
        if random.randint(0, 1):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = img + delta

        # random contrast, alpha in range [x, y]
        if random.randint(0, 1):
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            img = img * alpha
        img = img = np.clip(img, 0, 255)
        results['img'] = img.astype(np.uint8)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={})').format(self.brightness_delta, self.contrast_range)
        return repr_str


class TensorNormCropRotateFlip:

    def __init__(self, crop_size, move=5, train=True, copy_channels=False, mean=None, std=None, rotation=True):
        self.size = (crop_size, ) * 3 if isinstance(crop_size, int) else crop_size
        self.move = move
        self.copy_channels = copy_channels
        self.train = train
        self.mean = mean
        self.std = std
        self.rotation = rotation

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            voxel = results[key]
        shape = voxel.shape
        extra_dim = 0 if len(shape) == 3 else 1
        # norm
        if self.mean is None:
            voxel = voxel / 255. - 0.5
        else:
            voxel = (voxel - self.mean) / self.std
        # crop, rotation, reflection.
        if self.train:
            if self.move is not None:
                center = random_center(shape[-3:], self.move)
            else:
                center = np.array(shape[-3:]) // 2
            if voxel.shape[-3:] == self.size:
                voxel_ret = voxel
            else:
                voxel_ret = crop(voxel, center, self.size, extra_dim)
            if self.rotation:
                angle = np.random.randint(4, size=3)
                voxel_ret = do_rotation(voxel_ret, angle=angle, extra_dim=extra_dim)

            axis = np.random.randint(4) - 1
            voxel_ret = reflection(voxel_ret, axis=axis, extra_dim=extra_dim)
        else:
            center = np.array(shape[-3:]) // 2
            if voxel.shape[-3:] == self.size:
                voxel_ret = voxel
            else:
                voxel_ret = crop(voxel, center, self.size, extra_dim)
        if self.copy_channels:
            output = np.stack([voxel_ret, voxel_ret, voxel_ret], 0).astype(np.float32)
        else:
            if extra_dim == 0:
                output = np.expand_dims(voxel_ret, 0).astype(np.float32)
            else:
                output = voxel_ret.astype(np.float32)
        results['img'] = output
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(LIDC_Transform)'
        return repr_str

def do_rotation(array, angle, extra_dim=0):
    '''Using Euler angles method.
       angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0 + extra_dim, extra_dim + 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0 + extra_dim, extra_dim + 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1 + extra_dim, extra_dim + 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis, extra_dim=0):
    '''Flip tensor.
       axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis + extra_dim)
    else:
        ref = np.copy(array)
    return ref


def crop(array, zyx, dhw, extra_dim=0):
    z, y, x = zyx
    d, h, w = dhw
    if extra_dim == 0:
        cropped = array[z - d // 2:z + d // 2, y - h // 2:y + h // 2, x - w // 2:x + w // 2]
    elif extra_dim == 1:
        cropped = array[:, z - d // 2:z + d // 2, y - h // 2:y + h // 2, x - w // 2:x + w // 2]
    return cropped


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    zyx = np.array(shape) // 2 + offset
    return zyx


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')

class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img" and "gt_label".

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'flip',
            'flip_direction', 'img_norm_cfg')

    Returns:
        dict: The result dict contains the following keys

            - keys in ``self.keys``
            - ``img_metas`` if available
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'flip', 'flip_direction',
                            'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys},)'


class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (tuple, list))
        self.transforms = []
        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable, but got {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string

