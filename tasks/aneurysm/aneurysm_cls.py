import os
import time
import pickle
import numpy as np
import torch
from tasks.task import Task
from metrics.AUCEval import aucEval
from utils.tools.util import AverageMeter

from models.aneurysm_classification import DenseNet as DenseNet3D
from tasks.aneurysm.datasets.aneurysm_classification_dataset import ANEURYSM_CLS
from tasks.aneurysm.datasets.sampler import DefaultSampler, CTSampler
from .prefetcher import Data_Prefetcher
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools.util import count_time, get_now_datetime
from metrics.CommonEval import evaluate


from utils.config import cfg

def worker_init_fn(worker_id):
    np.random.seed(cfg.SEED)

class AneurysmCls(Task):

    def __init__(self):
        super(AneurysmCls, self).__init__()
        self.dataset_map = {}
        if cfg.TASK.STATUS == 'train':
            self.dataset_map['train'] = ANEURYSM_CLS(cfg, stage='train')
            self.dataset_map['val'] = ANEURYSM_CLS(cfg, stage='val')
            self.sampler_type = cfg.TRAIN.DATA.SAMPLER_TYPE

            # ret =  self.dataset_map['train'].__getitem__(0)


        elif cfg.TASK.STATUS == 'test':
            self.dataset_map['test'] = ANEURYSM_CLS(cfg, stage='test')



        #self.logger.info('Total train data: %d' % len(self.train_loader))
        #self.logger.info('Total validate data: %d' % len(self.val_loader))

        # self.logger.info('Total validate data: %d' % len(self.val_loader))

    def init_sampler(self, dataset, sampler_type, seed):
        # only for train stage
        assert sampler_type in ['default', 'ct'], 'unknown sampler type: {}'.format(sampler_type)
        # 需要提前设置好环境变量才能正确获取多GPU网络实际使用的显卡数量, 否则可能返回的是机器部署的全部GPU数
        # EG: os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        num_gpus = torch.cuda.device_count()
        if 'default' == sampler_type:
            self.logger.info('train with sampler: [default], used_gpus={}'.format(num_gpus))
            return DefaultSampler(dataset, num_replicas=num_gpus, seed=seed)
        elif 'ct' == sampler_type:
            num_per_ct = int(cfg.TRAIN.DATA.NUM_PER_CT)
            positive_fraction = float(cfg.TRAIN.DATA.POSITIVE_SAMPLE_FRACTION)
            self.logger.info('train with sampler: [ct] and [sample in per_ct={} with positive fraction={:.3f}], used_gpus={}'.format(num_per_ct, positive_fraction, num_gpus))
            return CTSampler(dataset, num_replicas=num_gpus, num_per_ct=num_per_ct, pos_fraction=positive_fraction, seed=seed)
        else:
            return None


    def get_model(self):
        if cfg.MODEL.NAME == 'densenet':
            self.net = DenseNet3D(cfg.MODEL.INPUT_CHANNEL, cfg.MODEL.NCLASS)
        else:
            super(AneurysmCls, self).get_model()

    @count_time
    def train(self, epoch):
        self.net.train()
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}
        aucEvalTrain = aucEval(cfg.MODEL.NCLASS)

        train_dataset = self.dataset_map['train']
        kwargs = {
            'worker_init_fn': worker_init_fn, 'num_workers': cfg.TRAIN.DATA.WORKERS,
            'batch_size': cfg.TRAIN.DATA.BATCH_SIZE,
            'pin_memory': True,
            # 'drop_last':True, # 自定义sampler控制
            # 'shuffle': True # 自定义sampler, 此选项必须关闭, 否则冲突
        }
        train_sampler = self.init_sampler(train_dataset, self.sampler_type, seed=epoch)
        train_data_loader = DataLoader(train_dataset, sampler=train_sampler, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(train_data_loader)))

        t0 = time.time()
        for batch_idx, data_dict in enumerate(train_data_loader):
            image = data_dict['img']
            label = data_dict['gt_label']

            target = label.long()
            image, target = image.cuda(), target.cuda()

            out = self.net(image)
            loss = self.criterion(out, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            t1 = time.time()
            aucEvalTrain.addBatch(out.max(1)[1].data, target.data)
            meters['loss'].update(loss.item(), image.size(0))
            meters['time'].update(t1-t0)

            if batch_idx % cfg.TRAIN.PRINT == 0:
                acc, pr, nr = aucEvalTrain.getMetric()
                self.logger.info('epoch=%04d, batch_idx=%06d, Time=%.2fs, loss=%.4f, Acc=%.2f%%, Neg-Recall=%.2f%%, Pos-Recall=%.2f%%,' % \
                                 (epoch, batch_idx, meters['time'].avg, meters['loss'].avg, acc, nr, pr))

            t0 = time.time()

            # if batch_idx >= 100:
            #     break

    @count_time
    @torch.no_grad()
    def validate(self):
        self.net.eval()
        meter_names = ['time', 'loss']
        meters = {name: AverageMeter() for name in meter_names}
        aucEvalVal = aucEval(cfg.MODEL.NCLASS)


        eval_dataset = self.dataset_map['val']
        kwargs = {
            'worker_init_fn': worker_init_fn, 'num_workers': cfg.TRAIN.DATA.WORKERS,
            'batch_size': cfg.TRAIN.DATA.BATCH_SIZE,
            'pin_memory': True, 'drop_last':False, 'shuffle': False
        }

        eval_data_loader = DataLoader(eval_dataset, **kwargs)

        preds = []
        t0 = time.time()
        for data_dict in tqdm(eval_data_loader, 'Eval Progress'):
            image = data_dict['img']
            label = data_dict['gt_label']
            target = label.long()
            image, target = image.cuda(), target.cuda()

            out = self.net(image)
            loss = self.criterion(out, target)
            aucEvalVal.addBatch(out.max(1)[1].data, target.data)
            preds += [torch.nn.functional.softmax(out, dim=1).cpu().numpy()]
            t1 = time.time()

            meters['loss'].update(loss.item(), image.size(0))
            meters['time'].update(t1-t0)

            t0 = time.time()

        acc, pr, nr = aucEvalVal.getMetric()
        self.logger.info('Eval: Time=%.3fms/batch, Loss=%.4f, Acc=%.4f%%, Neg-Recall=%.2f%%, Pos-Recall=%.2f%%,' % \
                         (meters['time'].avg * 1000, meters['loss'].avg, acc, nr, pr))

        ext_eval_results = evaluate(
            eval_dataset, results=preds, \
            metric_options={'topk': (1,)}, metric=cfg.METRIC_EXTEND, logger=self.logger)
        self.logger.info('Eval-extend: {}'.format(ext_eval_results))

        return {'Acc':acc, 'PR':pr, 'NR':nr, 'extend': ext_eval_results}

    @count_time
    @torch.no_grad()
    def test(self):
        self.net.eval()
        meter_names = ['time', 'loss']
        meters = {name: AverageMeter() for name in meter_names}
        aucEvalVal = aucEval(cfg.MODEL.NCLASS)


        eval_dataset = self.dataset_map['test']
        kwargs = {
            'worker_init_fn': worker_init_fn, 'num_workers': cfg.TRAIN.DATA.WORKERS,
            'batch_size': cfg.TRAIN.DATA.BATCH_SIZE,
            'pin_memory': True, 'drop_last':False, 'shuffle': False
        }

        eval_data_loader = DataLoader(eval_dataset, **kwargs)

        preds = []
        t0 = time.time()
        for data_dict in tqdm(eval_data_loader, 'Eval Progress'):
            image = data_dict['img']
            label = data_dict['gt_label']
            target = label.long()
            image, target = image.cuda(), target.cuda()

            out = self.net(image)
            loss = self.criterion(out, target)
            aucEvalVal.addBatch(out.max(1)[1].data, target.data)
            preds += [torch.nn.functional.softmax(out, dim=1).cpu().numpy()]
            t1 = time.time()

            meters['loss'].update(loss.item(), image.size(0))
            meters['time'].update(t1-t0)

            t0 = time.time()

        acc, pr, nr = aucEvalVal.getMetric()
        self.logger.info('Eval: Time=%.3fms/batch, Loss=%.4f, Acc=%.4f%%, Neg-Recall=%.2f%%, Pos-Recall=%.2f%%,' % \
                         (meters['time'].avg * 1000, meters['loss'].avg, acc, nr, pr))

        ext_eval_results = evaluate(
            eval_dataset, results=preds, \
            metric_options={'topk': (1,)}, metric=cfg.METRIC_EXTEND, logger=self.logger)
        self.logger.info('Eval-extend: {}'.format(ext_eval_results))

        SAVE_DIR = os.path.join('results', cfg.TASK.NAME, 'test', cfg.OUTPUT_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)
        eval_info_save_path = os.path.join(SAVE_DIR, 'eval_info_{}.pkl'.format(get_now_datetime()))
        with open(eval_info_save_path, 'wb') as fout:
            results = {
                'eval_result': ext_eval_results,
                'acc': acc, 'nr': nr, 'pr': pr,
                'ds_info': eval_dataset.data_infos,
            }
            pickle.dump(results, fout)
        self.logger.info('Eval result path: {}'.format(eval_info_save_path))