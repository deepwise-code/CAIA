import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import one_hot
from numbers import Number

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    # Modified from PyTorch-Ignite
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes**2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix


def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    num_classes = pred.size(1)
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_score = pred_score.flatten()
    pred_label = pred_label.flatten()

    gt_positive = one_hot(target.flatten(), num_classes)

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        pred_positive = one_hot(pred_label, num_classes)
        if thr is not None:
            pred_positive[pred_score <= thr] = 0
        class_correct = (pred_positive & gt_positive).sum(0)
        precision = class_correct / np.maximum(pred_positive.sum(0), 1.) * 100
        recall = class_correct / np.maximum(gt_positive.sum(0), 1.) * 100
        f1_score = 2 * precision * recall / np.maximum(
            precision + recall,
            torch.finfo(torch.float32).eps)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        elif average_mode == 'none':
            precision = precision.detach().cpu().numpy()
            recall = recall.detach().cpu().numpy()
            f1_score = f1_score.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores


def precision(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Precision.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=0.):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Recall.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=0.):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: F1 score.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Support.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res


def accuracy_numpy(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred = pred.float()
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x)
                 if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res


def get_aneu_eval_auc_indicator(pred_score_list, eval_patch_info_list, mtype=('add', 'mul'), verbose=0):
    """[summary]

    Args:
        pred_score_list ([list]): pred prob list
        eval_patch_info_list ([list]): 2d list of slice info desc and class
        mtype (str, list, optional): [description]. Defaults to 'add'.

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        [list]: [description] diff dim of auc info list
    """
    pred_count = len(pred_score_list)
    eval_list_count = len(eval_patch_info_list)
    assert pred_count == eval_list_count, 'aneu eval param length except: {} != {}'.format(pred_count, eval_list_count)

    # seg lesion/slice patch props auc calc
    auc_indicator_group_base_slice = {'stage1': [], 'stage2': []}
    auc_indicator_group_base_lesion = {'stage1': {}, 'stage2': {}}

    # 融合分组清单
    mtype_list = []
    if isinstance(mtype, (tuple, list)):
        mtype_list = mtype
    else:
        mtype_list = [mtype]
    for _mtype in mtype_list:
        merge_group_name = f'merge-{_mtype}'
        auc_indicator_group_base_slice.setdefault(merge_group_name, [])
        auc_indicator_group_base_lesion.setdefault(merge_group_name, {})

    slice_patch_count = {'pos': 0, 'neg': 0, 'all': 0}
    seg_lesion_count = {'pos': set(), 'neg': set(), 'all': set()}
    match_gt_lesion_count = set()
    gt_sample_lesion_count = set()
    case_count = {'all': set(), 'pos': set()}

    # stage1 slice patch and lesion match gt info
    for pred_score, (sub_path, _cls) in zip(pred_score_list, eval_patch_info_list):
        # eg: x151_y127_z172_w32_h32_d32_s0.39_seg_neg_5_0.png
        series_name, slice_patch_info = sub_path.split('/')
        *slice_info, seg_gt_label, pos_neg_label, inst_index, recall_inst_index = slice_patch_info.split('_')

        # all
        slice_patch_index = sub_path
        lesion_index = f'{series_name};{inst_index}'

        case_count['all'].add(series_name)

        if 'gt' == seg_gt_label:
            gt_lesion_key = f'{series_name};{inst_index}'  # var:inst_index same to var:recall_inst_index
            gt_sample_lesion_count.add(gt_lesion_key)
            case_count['pos'].add(series_name)
            # continue
        elif 'seg' == seg_gt_label:
            s1_score = float(slice_info[-1][1:])
            s2_score = pred_score
            # 1. slice patch level of seg props only
            slice_type = int(_cls)
            auc_indicator_group_base_slice['stage1'] += [(slice_patch_index, s1_score, slice_type)]
            auc_indicator_group_base_slice['stage2'] += [(slice_patch_index, s2_score, slice_type)]

            # 2. lesion level of seg props only
            lesion_recall_type = int(_cls)
            # 2.1 lesion in stage1
            auc_indicator_group_base_lesion['stage1'].setdefault(lesion_index, [-1, lesion_recall_type])
            cur_lesion_score_s1 = auc_indicator_group_base_lesion['stage1'][lesion_index][0]
            auc_indicator_group_base_lesion['stage1'][lesion_index][0] = max(cur_lesion_score_s1, s1_score)

            auc_indicator_group_base_lesion['stage2'].setdefault(lesion_index, [-1, lesion_recall_type])
            cur_lesion_score_s2 = auc_indicator_group_base_lesion['stage2'][lesion_index][0]
            auc_indicator_group_base_lesion['stage2'][lesion_index][0] = max(cur_lesion_score_s2, s2_score)

            # 3.merge of slice patch & lesion level
            for _mtype in mtype_list:
                merge_group_name = f'merge-{_mtype}'
                if 'add' == _mtype:
                    score_merge_s1_s2 = (s1_score + s2_score)/2
                elif 'mul' == _mtype:
                    score_merge_s1_s2 = float(np.sqrt(s1_score * s2_score))
                else:
                    raise Exception('unsport merge method for stage1 stage2: {}'.format(_mtype))
                auc_indicator_group_base_slice[merge_group_name] += [(slice_patch_index, score_merge_s1_s2, slice_type)]

                auc_indicator_group_base_lesion[merge_group_name].setdefault(lesion_index, [-1, lesion_recall_type])
                cur_lesion_score_ms1s2 = auc_indicator_group_base_lesion[merge_group_name][lesion_index][0]
                auc_indicator_group_base_lesion[merge_group_name][lesion_index][0] = \
                    max(cur_lesion_score_ms1s2, score_merge_s1_s2)

            # 4.count
            slice_patch_count['all'] += 1
            slice_patch_count[pos_neg_label] += 1

            seg_lesion_key = f'{series_name};{inst_index}'
            seg_lesion_count['all'].add(seg_lesion_key)
            seg_lesion_count[pos_neg_label].add(seg_lesion_key)

            if 'pos' == pos_neg_label:
                recall_inst_index_list = recall_inst_index.rsplit('.png', maxsplit=1)[0]
                recall_inst_index_list = recall_inst_index_list.split('&')
                for recall_inst_index in recall_inst_index_list:
                    match_gt_lesion_key = f'{series_name};{recall_inst_index}'
                    match_gt_lesion_count.add(match_gt_lesion_key)
        else:
            raise Exception('unknown origin type: {} in {}'.format(seg_gt_label, sub_path))

    seg_auc_info_list = []
    for group_name, seg_slice_info in auc_indicator_group_base_slice.items():
        pred_score_list = [_[1] for _ in seg_slice_info]
        label_list = [_[2] for _ in seg_slice_info]
        all_neg = all([0 == _ for _ in label_list])
        all_pos = all([1 == _ for _ in label_list])
        if all_neg or all_pos:
            auc_score = -1.00
        else:
            y_pred_list = np.array(pred_score_list)
            y_gt_list = np.array(label_list)
            auc_score = roc_auc_score(y_true=y_gt_list, y_score=y_pred_list)
        info = f'{group_name}-slice-auc={auc_score:.4f}'
        seg_auc_info_list += [info]

    for group_name, seg_lesion_info_map in auc_indicator_group_base_lesion.items():
        seg_lesion_info = list(seg_lesion_info_map.values())
        pred_score_list = [_[0] for _ in seg_lesion_info]
        label_list = [_[1] for _ in seg_lesion_info]
        all_neg = all([0 == _ for _ in label_list])
        all_pos = all([1 == _ for _ in label_list])
        if all_neg or all_pos:
            auc_score = -1.00
        else:
            y_pred_list = np.array(pred_score_list)
            y_gt_list = np.array(label_list)
            auc_score = roc_auc_score(y_true=y_gt_list, y_score=y_pred_list)
            info = f'{group_name}-lesion-auc={auc_score:.4f}'
        seg_auc_info_list += [info]

    # count show
    used_seg_slice_count = slice_patch_count
    used_seg_lesion_count_map = {k: len(v) for k, v in seg_lesion_count.items()}
    used_seg_macth_gt_lesion_count = len(match_gt_lesion_count)  # max match gt lesion
    used_gt_lesion_count = len(gt_sample_lesion_count)  # all gt lesion
    case_total = len(case_count['all'])  # all case count
    case_positive = len(case_count['pos'])
    case_negative = case_total - case_positive
    other_infos = [
        f'seg-slice-count={used_seg_slice_count}',
        f'seg-max-match-gt-lesion={used_seg_macth_gt_lesion_count}/{used_gt_lesion_count}',
        'seg-lesion-count={}=P{}+N{}'.format(*[used_seg_lesion_count_map[k] for k in ['all', 'pos', 'neg']]),
        'eval-case-count={}=P{}+N{}'.format(case_total, case_positive, case_negative),
    ]
    if verbose != 0:
        print('slice count in seg:', used_seg_slice_count)
        print('lesion count in seg:', used_seg_lesion_count_map)
        print('match gt lesion total:', used_seg_macth_gt_lesion_count, used_gt_lesion_count)

    return seg_auc_info_list, other_infos

def evaluate(aneu_cls_dataset, results, metric='accuracy', metric_options=None, indices=None, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        # todo: overwrite this metric function
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]


        eval_results = {}
        results = np.vstack(results)
        gt_labels = aneu_cls_dataset.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        topk = metric_options.get('topk', (1, 5))
        if not isinstance(topk, (list, tuple)):
            topk = [topk]

        if metric == 'accuracy':
            # loss
            loss = F.nll_loss(torch.log(torch.Tensor(results) + 1e-6), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 4)
            eval_results['ce_loss'] = loss
            # acc thr=0.5
            acc = accuracy(results, gt_labels, topk=topk)
            eval_results.update({f'acc_top-{k}': round(a.item()/100, 4)  for k, a in zip(topk, acc)})
        elif metric == 'auc_and_acc':
            # loss
            loss = F.nll_loss(torch.log(torch.Tensor(results) + 1e-6), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 4)
            eval_results['ce_loss'] = loss
            # acc thr=0.5
            acc = accuracy(results, gt_labels, topk=topk)
            eval_results.update({f'acc_top-{k}': round(a.item()/100, 4)  for k, a in zip(topk, acc)})

            # auc: for binary class
            if len(set(gt_labels)) == 1: # for only one cls data in valid dataset
                auc = -1.00
            else:
                auc = roc_auc_score(y_true=gt_labels, y_score=results[:,1])
            eval_results['auc'] = auc

            pred_labels = np.argmax(results, 1)
            print('\nconfusion_matrix: row-label=gt, col-label=pred')
            print(confusion_matrix(pred_labels, gt_labels))
        elif metric == 'all':
            # loss
            loss = F.nll_loss(torch.log(torch.Tensor(results) + 1e-6), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 4)
            eval_results['ce_loss'] = loss
            # acc thr=0.5
            acc = accuracy(results, gt_labels, topk=topk)
            eval_results.update({f'acc_top-{k}': round(a.item()/100, 4)  for k, a in zip(topk, acc)})

            # auc: for multi class, used multi binanry class auc score mean
            auc_pred_cls_map = {}
            gt_positive = one_hot(torch.from_numpy(gt_labels).flatten(), num_classes=len(aneu_cls_dataset.CLASSES))
            for _cls in aneu_cls_dataset.CLASSES:
                _idx = aneu_cls_dataset.CLASSES.index(_cls)
                if len(set(gt_positive[:, _idx])) == 1: # for only one cls data in valid dataset
                    _auc = -1.00
                else:
                    _auc = roc_auc_score(y_true=gt_positive[:, _idx], y_score=results[:, _idx])
                auc_pred_cls_map[_idx] = _auc
            auc_avg = sum(auc_pred_cls_map.values()) / (len(auc_pred_cls_map) + 1e-6)
            if len(aneu_cls_dataset.CLASSES) == 2:
                eval_results['auc'] = auc_avg
            else:
                eval_results['auc-avg'] = auc_avg
                eval_results.update({f'auc-cls{_cls}': _auc for _cls, _auc in auc_pred_cls_map.items()})

            # loop for every class recall/precision
            average_mode = 'none' # 'macro'
            precisions, recalls, _ =  precision_recall_f1(results, gt_labels, average_mode=average_mode, thrs=0.)
            for _cls, (_prec, _rec) in enumerate(zip(precisions, recalls)):
                eval_results.update({
                    'cls{}-recall'.format(_cls): round(_rec/100, 4),        # sensitivity/specificity
                    'cls{}-precision'.format(_cls): round(_prec/100, 4),
                })
        elif metric == 'auc_multi_cls_and_acc':
            # loss
            loss = F.nll_loss(torch.log(torch.Tensor(results) + 1e-6), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 4)
            eval_results['ce_loss'] = loss
            # acc thr=0.5
            acc = accuracy(results, gt_labels, topk=topk)
            eval_results.update({f'acc_top-{k}': round(a.item()/100, 4)  for k, a in zip(topk, acc)})

            # auc: for multi class, used multi binanry class auc score mean
            auc_pred_cls_map = {}
            for _cls in aneu_cls_dataset.CLASSES:
                _idx = aneu_cls_dataset.CLASSES.index(_cls)
                if len(set(gt_labels[:, _idx])) == 1: # for only one cls data in valid dataset
                    _auc = -1.00
                else:
                    _auc = roc_auc_score(y_true=gt_labels[:, _idx], y_score=results[:, _idx])
                auc_pred_cls_map[_cls] = _auc
            auc_avg = sum(auc_pred_cls_map.values()) / (len(auc_pred_cls_map) + 1e-6)
            if len(aneu_cls_dataset.CLASSES) == 2:
                eval_results['auc'] = auc_avg
            else:
                eval_results['auc-avg'] = auc_avg
                eval_results.update({'auc-cls{_cls}': _auc for _cls, _auc in auc_pred_cls_map.items()})

            pred_labels = np.argmax(results, 1)
            print('\nconfusion_matrix: row-label=gt, col-label=pred')
            print(confusion_matrix(pred_labels, gt_labels))
        elif metric == 'extend-aneurysm': # slice & lesion level auc
            # loss
            loss = F.nll_loss(torch.log(torch.Tensor(results) + 1e-6), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 4)
            eval_results['ce_loss'] = loss

            # acc thr=0.5
            acc = accuracy(results, gt_labels, topk=topk)
            eval_results.update({f'acc_top-{k}': round(a.item()/100, 4)  for k, a in zip(topk, acc)})

            # auc: for multi class, used multi binanry class auc score mean
            auc_pred_cls_map = {}
            gt_positive = one_hot(torch.from_numpy(gt_labels).flatten(), num_classes=len(aneu_cls_dataset.CLASSES))
            for _cls in aneu_cls_dataset.CLASSES:
                _idx = aneu_cls_dataset.CLASSES.index(_cls)
                if len(set(gt_positive[:, _idx])) == 1: # for only one cls data in valid dataset
                    _auc = -1.00
                else:
                    _auc = roc_auc_score(y_true=gt_positive[:, _idx], y_score=results[:, _idx])
                auc_pred_cls_map[_idx] = _auc
            auc_avg = round(sum(auc_pred_cls_map.values()) / (len(auc_pred_cls_map) + 1e-6), 4)
            if len(aneu_cls_dataset.CLASSES) == 2:
                eval_results['auc'] = auc_avg
            else:
                eval_results['auc-avg'] = auc_avg
                eval_results.update({f'auc-cls{_cls}': _auc for _cls, _auc in auc_pred_cls_map.items()})

            # loop for every class recall/precision
            average_mode = 'none' # 'macro'
            precisions, recalls, _ =  precision_recall_f1(results, gt_labels, average_mode=average_mode, thrs=0.)
            for _cls, (_prec, _rec) in enumerate(zip(precisions, recalls)):
                eval_results.update({
                    'cls{}-recall'.format(_cls): round(_rec/100, 4),        # sensitivity/specificity
                    'cls{}-precision'.format(_cls): round(_prec/100, 4),
                })

            seg_auc_info_list, other_infos = get_aneu_eval_auc_indicator(results[:, 1], aneu_cls_dataset.eval_patch_info_list, mtype=('add',))

            for eval_info in seg_auc_info_list:
                key, value = eval_info.split('=', maxsplit=1)
                eval_results[key] = value
            for eval_info in other_infos:
                key, value = eval_info.split('=', maxsplit=1)
                eval_results[key] = value
        else:
            raise ValueError(f'metric {metric} is not supported.')
        return eval_results