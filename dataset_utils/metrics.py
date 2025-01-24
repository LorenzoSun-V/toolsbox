import torch
import torch.nn.functional as F
import os.path as osp
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from utils import *


class Metric:
    """
        Class for computing evaluation metrics for YOLOv8 model.

        Attributes:
            p (list): Precision for each class. Shape: (nc,).
            r (list): Recall for each class. Shape: (nc,).
            f1 (list): F1 score for each class. Shape: (nc,).
            all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
            ap_class_index (list): Index of class for each AP score. Shape: (nc,).
            nc (int): Number of classes.

        Methods:
            ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
            ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
            mp(): Mean precision of all classes. Returns: Float.
            mr(): Mean recall of all classes. Returns: Float.
            map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
            map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
            map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
            mean_results(): Mean of results, returns mp, mr, map50, map.
            class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
            maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
            fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
            update(results): Update metric attributes with new evaluation results.
        """

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP50 at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP50 at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index = results


class DetMetrics:
    """
    This class is a utility class for computing detection metrics such as precision, recall, and mean average precision
    (mAP) of an object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (tuple of str): A tuple of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (tuple of str): A tuple of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
    """

    def __init__(self, save_dir=Path('.'), plot=False, on_plot=None, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(tp,
                               conf,
                               pred_cls,
                               target_cls,
                               plot=self.plot,
                               save_dir=self.save_dir,
                               names=self.names,
                               on_plot=self.on_plot)[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.fitness]))
    

class SegmentMetrics:
    """
    Calculates and aggregates detection and segmentation metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize a SegmentMetrics instance with a save directory, plot flag, callback function, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.seg = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "segment"

    def process(self, tp, tp_m, conf, pred_cls, target_cls):
        """
        Processes the detection and segmentation metrics over the given set of predictions.

        Args:
            tp (list): List of True Positive boxes.
            tp_m (list): List of True Positive masks.
            conf (list): List of confidence scores.
            pred_cls (list): List of predicted classes.
            target_cls (list): List of target classes.
        """
        results_mask = ap_per_class(
            tp_m,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """Returns a list of keys for accessing metrics."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self):
        """Return the mean metrics for bounding box and segmentation results."""
        return self.box.mean_results() + self.seg.mean_results()

    def class_result(self, i):
        """Returns classification results for a specified class index."""
        return self.box.class_result(i) + self.seg.class_result(i)

    @property
    def maps(self):
        """Returns mAP scores for object detection and semantic segmentation models."""
        return self.box.maps + self.seg.maps

    @property
    def fitness(self):
        """Get the fitness score for both segmentation and bounding box models."""
        return self.seg.fitness() + self.box.fitness()

    @property
    def ap_class_index(self):
        """Boxes and masks have the same ap_class_index."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns results of object detection model for evaluation."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.seg.curves_results


class DetValidator:
    def __init__(self, pred_path, gt_path, val_list_path, names, device="cuda:0"):
        """
        Args:
            pred_path (str): path to predicted txt files
            gt_path (str): path to ground truth txt files
            val_list_path (str): path to validation list txt file
            names (dict): class names dict, eg. {0:'SL', 1:'MS'}
        """
        self.device=device
        self.names = names
        self.pred_dict, self.gt_dict = self.get_pred_gt_dirs(pred_path, gt_path, val_list_path)
        self.nc = len(names)
        self.metrics = DetMetrics(names=names)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.stats = []
        self.seen = 0
    
    def update_metrics(self, preds, gts):
        self.seen +=1
        cls = gts[:, 0] if len(gts)>0 else torch.Tensor([]).to(self.device)
        nl, npr = len(cls), preds.shape[0]
        correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
        if npr == 0:
            if nl:
                self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls))
            return

        if nl:
            correct_bboxes = self._process_batch(preds, gts)
        self.stats.append((correct_bboxes, preds[:, -2], preds[:, -1], cls))
                  
    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        # print(stats)
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict
    
    def cal_metrics(self):
        for img in self.img_name:
            self.update_metrics(self.pred_dict[img], self.gt_dict[img])
        stats = self.get_stats()
        self.print_results()
    
    def print_results(self):
        """Prints training/validation set metrics per class."""
        print(('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)'))
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        print(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            print(
                f'WARNING ⚠️ no labels found in set, can not compute metrics without labels')

        # Print results per class
        if self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                print((pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i))))

    def _read_txt2tensor(self, txt_path):
        res = []
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.read().strip().splitlines():
                    r = []
                    for item in line.split():
                        r.append(float(item))
                    res.append(r)
        return torch.Tensor(res).to(self.device)
                        
    def get_pred_gt_dirs(self, pred_path, gt_path, val_list_path):
        """
        Read saved pred txt and gt txt,
        pred_dict = {
            'img_name': tensor([
                [x1, y1, x2, y2, conf, cls],
                [x1, y1, x2, y2, conf, cls],
                ...])
        }
        gt_dict = {
            'img_name': tensor([
                [cls, x1, y1, x2, y2],
                [cls, x1, y1, x2, y2],
                ...])
        }
        """
        pred_dict = {}
        gt_dict = {}
        self.img_name = []
        with open(val_list_path, 'r') as f:
            img_paths = f.readlines()
            for img_path in img_paths:
                img_name = osp.basename(img_path.strip()).rsplit('.', 1)[0]
                self.img_name.append(img_name)
                pred_dict[img_name] = self._read_txt2tensor(osp.join(pred_path, f"{img_name}.txt"))
                gt_dict[img_name] = self._read_txt2tensor(osp.join(gt_path, f"{img_name}.txt"))
        return pred_dict, gt_dict


class OBBValidator(DetValidator):
    def __init__(self, pred_path, gt_path, val_list_path, names, device="cuda:0"):
        """
        Args:
            pred_path (str): path to predicted txt files
            gt_path (str): path to ground truth txt files
            val_list_path (str): path to validation list txt file
            names (dict): class names dict, e.g., {0:'SL', 1:'MS'}
        """
        super().__init__(pred_path, gt_path, val_list_path, names, device)

    def _process_batch(self, detections, labels):
        """
        Override to use rotated IoU for matching.

        Args:
            detections (torch.Tensor): Predicted boxes, shape [N, 6] -> [x_center, y_center, w, h, angle, class].
            labels (torch.Tensor): Ground truth boxes, shape [M, 5] -> [class, x_center, y_center, w, h, angle].
        
        Returns:
            torch.Tensor: Correct prediction matrix for IoU levels.
        """
        iou = batch_probiou(labels[:, 1:], detections[:, :5])  # Calculate IoU between ground truth and predictions
        return self.match_predictions(detections[:, -1], labels[:, 0], iou)
    
    def _read_txt2tensor(self, txt_path):
        res = []
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.read().strip().splitlines():
                    r = []
                    for item in line.split():
                        # 如果class_map存在并且item是字符串，则根据字典转换为对应的float
                        if self.names and item in self.names.values():
                            item = list(self.names.keys())[list(self.names.values()).index(item)]  # 获取对应的float
                        r.append(float(item))  # 转换为float
                    res.append(r)
        return torch.Tensor(res).to(self.device)
    
    def get_pred_gt_dirs(self, pred_path, gt_path, val_list_path):
        """
        Read saved pred txt and gt txt,
        pred_dict = {
            'img_name': tensor([
                [x_center, y_center, w, h, radian, conf, cls],
                [x_center, y_center, w, h, radian, conf, cls],
                ...])
        }
        gt_dict = {
            'img_name': tensor([
                [x1, y1, x2, y2, x3, y3, x4, y4, cls_name],
                [x1, y1, x2, y2, x3, y3, x4, y4, cls_name],
                ...])
        }
        """
        pred_dict = {}
        gt_dict = {}
        self.img_name = []
        with open(val_list_path, 'r') as f:
            img_paths = f.readlines()
            for img_path in img_paths:
                img_name = osp.basename(img_path.strip()).rsplit('.', 1)[0]
                self.img_name.append(img_name)
                pred_dict[img_name] = self._read_txt2tensor(osp.join(pred_path, f"{img_name}.txt"))

                # 读取GT数据并转换为xywhr格式
                gt_tensor = self._read_txt2tensor(osp.join(gt_path, f"{img_name}.txt"))
                if len(gt_tensor)==0:
                    # 如果没有GT数据，则略过索引读取，直接把空列表赋值给gt_dict
                    gt_dict[img_name] = gt_tensor
                else:
                    gt_xywhr = xyxyxyxy2xywhr(gt_tensor[:, :8])  # 转换为 [x_center, y_center, width, height, radian]
                    # 将cls和转换后的坐标拼接到一起
                    gt_dict[img_name] = torch.cat((gt_tensor[:, 8].unsqueeze(1), gt_xywhr), dim=1)
        return pred_dict, gt_dict


class SegmentValidator(DetValidator):
    def __init__(self, pred_path, gt_path, val_list_path, names, device="cuda:0"):
        """
        Args:
            pred_path (str): path to predicted txt files
            gt_path (str): path to ground truth txt files
            val_list_path (str): path to validation list txt file
            names (dict): class names dict, e.g., {0:'SL', 1:'MS'}
        """
        super().__init__(pred_path, gt_path, val_list_path, names, device)
        self.metrics = SegmentMetrics(names=names)
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def update_metrics(self, preds, gts):
        self.seen += 1
        cls, bbox = torch.from_numpy(gts.pop("classes")).to(self.device), torch.from_numpy(gts.pop("boxes")).to(self.device)
 
        npr, nl = len(preds['classes']), len(cls)
        stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
        stat["target_cls"] = cls
        stat["target_img"] = cls.unique()
        if npr == 0:
            if nl:
                for k in self.stats.keys():
                    self.stats[k].append(stat[k])
            return
        
        gt_masks = torch.from_numpy(gts.pop("masks")).to(self.device)
        pred_masks = torch.from_numpy(preds.pop("masks")).to(self.device)
        stat["conf"] = torch.from_numpy(preds['confs']).to(self.device)
        stat["pred_cls"] = torch.from_numpy(preds['classes']).to(self.device)
        if nl:
            preds_bboxes = torch.from_numpy(np.hstack((preds['boxes'], preds['confs'][:, np.newaxis], preds['classes'][:, np.newaxis]))).to(self.device)
            
            stat["tp"] = self._process_batch(preds_bboxes, bbox, cls)
            stat["tp_m"] = self._process_batch(
                    preds_bboxes, bbox, cls, pred_masks, gt_masks, True, masks=True)
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def print_results(self):
        """Prints training/validation set metrics per class, including mask metrics."""
        print(('%22s' + '%11s' * 6 + '%11s' * 4) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95', 'Mask(P', 'R', 'mAP50', 'mAP50-95)'))
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        print(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            print(
                f'WARNING ⚠️ no labels found in set, can not compute metrics without labels')

        # Print results per class
        if self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                print((pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i))))

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=True, masks=False):
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detected bounding boxes and
                associated confidence scores and class indices. Each row is of the format [x1, y1, x2, y2, conf, class].
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground truth bounding box coordinates.
                Each row is of the format [x1, y1, x2, y2].
            gt_cls (torch.Tensor): Tensor of shape (M,) representing ground truth class indices.
            pred_masks (torch.Tensor | None): Tensor representing predicted masks, if available. The shape should
                match the ground truth masks.
            gt_masks (torch.Tensor | None): Tensor of shape (M, H, W) representing ground truth masks, if available.
            overlap (bool): Flag indicating if overlapping masks should be considered.
            masks (bool): Flag indicating if the batch contains mask data.

        Returns:
            (torch.Tensor): A correct prediction matrix of shape (N, 10), where 10 represents different IoU levels.

        Note:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.

        Example:
            ```python
            detections = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            gt_cls = torch.tensor([1, 0])
            correct_preds = validator._process_batch(detections, gt_bboxes, gt_cls)
            ```
        """
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                print(f"Warning: gt_masks shape {gt_masks.shape} != pred_masks shape {pred_masks.shape}, interpolate gt_masks to pred_masks shape")
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:
            iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def get_color_for_class(self, cls):
        """
        根据类别返回颜色。
    
        参数:
            cls (int): 类别索引。
    
        返回:
            tuple: BGR 颜色元组。
        """
        color = [(123, 45, 231), (23, 189, 12), (78, 234, 90), (112, 34, 190), (210, 123, 45), (56, 78, 198), (155, 22, 89), (201, 102, 145), (34, 178, 222), (99, 122, 33), (188, 56, 200), (77, 134, 211), (145, 202, 44), (22, 88, 155), (190, 33, 111), (12, 188, 233), (212, 144, 77), (88, 111, 199), (166, 33, 123), (231, 199, 44), (55, 200, 111), (133, 99, 222), (177, 233, 88), (44, 112, 188), (101, 210, 55), (189, 133, 77), (222, 44, 155), (77, 190, 123), (155, 88, 211), (200, 111, 144), (33, 145, 199), (111, 222, 66), (199, 77, 101), (66, 155, 233), (144, 211, 88), (233, 111, 177), (88, 188, 55), (177, 123, 200), (211, 155, 99), (101, 199, 133), (133, 233, 77), (188, 88, 166), (44, 144, 222), (111, 177, 101), (199, 123, 155), (77, 211, 188), (155, 111, 233), (222, 144, 88), (101, 188, 177), (133, 166, 211), (188, 111, 123), (233, 155, 77), (88, 222, 199), (177, 144, 133), (211, 188, 101), (101, 233, 166), (133, 199, 111), (188, 155, 144), (222, 177, 99), (88, 211, 123), (155, 188, 199), (233, 211, 77), (101, 166, 144), (133, 144, 200), (188, 177, 111), (222, 199, 88), (88, 233, 155), (177, 166, 123), (211, 222, 99), (101, 144, 188), (133, 111, 211), (188, 199, 144), (222, 166, 101), (88, 155, 177), (177, 211, 133), (211, 144, 166), (101, 111, 222), (133, 188, 199), (188, 123, 155), (222, 155, 111), (88, 199, 177), (177, 111, 200), (211, 166, 123), (101, 177, 144), (133, 222, 188), (188, 144, 111), (222, 188, 99), (88, 166, 155), (177, 199, 123), (211, 133, 177), (101, 222, 144), (133, 155, 211), (188, 200, 88), (222, 123, 199), (88, 144, 166), (177, 177, 133), (211, 111, 188), (101, 199, 155), (133, 123, 233), (188, 166, 111), (222, 101, 177), (88, 133, 211), (177, 188, 144), (211, 99, 200), (101, 155, 166), (133, 211, 123), (188, 199, 99), (222, 111, 155), (88, 177, 188), (177, 123, 133), (211, 144, 222), (101, 188, 155), (133, 111, 199)]
        return color[cls]

    def _read_txt2tensor_pred(self, txt_path):
        res = {}
        bin_path = txt_path.replace(".txt", ".bin")
        if osp.exists(txt_path) and osp.exists(bin_path):
            with open(txt_path, 'r') as f_txt:
                lb = [x.split() for x in f_txt.read().strip().splitlines() if len(x)]
                classes = np.array([x[-1] for x in lb], dtype=np.float32)
                confs = np.array([x[-2] for x in lb], dtype=np.float32)
                boxes = np.array([x[:4] for x in lb], dtype=np.float32)
            
            with open(bin_path, 'rb') as f_bin:
                num_masks = np.fromfile(f_bin, dtype=np.uint64, count=1)[0]
                masks = []

                for _ in range(num_masks):
                    # 读取每个 mask 的维度
                    rows = np.fromfile(f_bin, dtype=np.int32, count=1)[0]
                    cols = np.fromfile(f_bin, dtype=np.int32, count=1)[0]
                    
                    # 读取每个 mask 的数据
                    mask_data = np.fromfile(f_bin, dtype=np.float32, count=rows * cols)
                    mask = mask_data.reshape((rows, cols))
                    masks.append(mask)

            res['classes'] = classes                # array[cls, cls, cls,...]
            res['boxes'] = boxes                    # array[[x1, y1, x2, y2],...]
            res['confs'] = confs                    # array[conf, conf, conf,...]
            res['masks'] = np.array(masks)          # hxw, [[], [],...]
        else:
            res['classes'] = np.array([])
            res['boxes'] = np.array([[]])
            res['confs'] = np.array([])
            res['masks'] = np.array([[]])
        return res
    
    def _read_txt2tensor_gt(self, txt_path, imgsz):
        h, w = imgsz
        res = {}
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                classes = np.array([x[0] for x in lb], dtype=np.float32)
                segments_normalized = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                segments = [seg * [w, h] for seg in segments_normalized] # 横坐标*w, 纵坐标*h
                boxes = segments2xyxyboxes(segments)
                segments_resampled = resample_segments(segments)
                masks, boxes, classes = format_segments(segments_resampled, boxes, classes, w, h)

                # 将结果存储到 res 字典中
                res['classes'] = classes                # array[cls, cls, cls, ...]
                res['boxes'] = boxes                    # array[[x1, y1, x2, y2], ...]
                res['masks'] = masks                    # hxw, [[], [], ...]
        else:
            res['classes'] = np.array([])
            res['boxes'] = np.array([[]])
            res['masks'] = np.array([[]])
        return res

    def get_pred_gt_dirs(self, pred_path, gt_path, val_list_path):
        pred_dict = {}
        gt_dict = {}
        self.img_name = []
        print("Start loading...")
        with open(val_list_path, 'r') as f:
            img_paths = f.readlines()
            for img_path in tqdm(img_paths):
                img_path = img_path.strip()
                h, w = cv2.imread(img_path).shape[:2]
                img_name = osp.basename(img_path).rsplit('.', 1)[0]
                self.img_name.append(img_name)
                pred_dict[img_name] = self._read_txt2tensor_pred(osp.join(pred_path, f"{img_name}.txt"))
                gt_dict[img_name] = self._read_txt2tensor_gt(osp.join(gt_path, f"{img_name}.txt"), (h,w))
        print('Loading done!')
        return pred_dict, gt_dict


if __name__ == "__main__":
    # OBB 精度计算示例
    # obbval = OBBValidator(
    #     pred_path="/data/bt/hw_obb/cls2_hw_obb_v0.1/images/val_output",
    #     gt_path="/data/bt/hw_obb/cls2_hw_obb_v0.1/labels/val_original",
    #     val_list_path="/data/bt/hw_obb/cls2_hw_obb_v0.1/val.txt",
    #     names={0:'SL', 1:'MS'}) 
    # obbval.cal_metrics()

    # HBB 精度计算示例
    # detval = DetValidator(
    #     pred_path="/lorenzo/bt_repo/ultralytics/runs/detect/val2/labels",
    #     gt_path="/data/bt/xray/cls13_xray-sub_v1.0/gts",
    #     val_list_path="/data/bt/xray/cls13_xray-sub_v1.0/trainval/val.txt",
    #     names={0:'embedding_gs', 1: 'fracture_gs', 2: 'fall_gs', 3: 'damage_gs', 4: 'rust_gs', 5: 'embedding_fl', 
    #            6: 'fracture_fl', 7: 'glue_fl', 8: 'scratch_fl', 9: 'fall_fl', 10: 'brokenlen_gs', 11: 'fracture_full_gs', 12: 'rust_full_gs'})
    # detval.cal_metrics()

    # InstanceSegment 精度计算示例
    names = {}
    for i in range(80):
        names[i] = str(i)
    root_path = "/data/Datasets/public/coco_instanceseg"
    root_path = "/data/Datasets/public/coco_instanceseg/mini-test"
    segval = SegmentValidator(
        pred_path="/data/Datasets/public/coco_instanceseg/mini-test/images/output",
        gt_path=f"{root_path}/labels",
        val_list_path=f"{root_path}/val.txt",
        names=names)
    segval.cal_metrics()