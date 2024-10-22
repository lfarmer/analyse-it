# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
from matplotlib import pyplot as plt

from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample
from mmpose.visualization.opencv_backend_visualizer import OpencvBackendVisualizer

from visualisers import MMPoseKeypoint
from visualisers.person import PosePerson


def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales

@VISUALIZERS.register_module()
class Pose2dWithAnglesVisualizer(OpencvBackendVisualizer):
    """Pose2dWithAnglesVisualizer

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to ``None``
        vis_backends (list, optional): Visual backend config list. Defaults to
            ``None``
        save_dir (str, optional): Save file dir for all storage backends.
            If it is ``None``, the backend storage will not save any data.
            Defaults to ``None``
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to ``'green'``
        kpt_color (str, tuple(tuple(int)), optional): Color of keypoints.
            The tuple of color should be in BGR order. Defaults to ``'red'``
        link_color (str, tuple(tuple(int)), optional): Color of skeleton.
            The tuple of color should be in BGR order. Defaults to ``None``
        line_width (int, float): The width of lines. Defaults to 1
        radius (int, float): The radius of keypoints. Defaults to 4
        show_keypoint_weight (bool): Whether to adjust the transparency
            of keypoints according to their score. Defaults to ``False``
        alpha (int, float): The transparency of bboxes. Defaults to ``1.0``
    """
    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = 'green',
                 kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = 'red',
                 link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (255, 255, 255),
                 skeleton: Optional[Union[List, Tuple]] = None,
                 line_width: Union[int, float] = 1,
                 radius: Union[int, float] = 3,
                 show_keypoint_weight: bool = False,
                 backend: str = 'opencv',
                 alpha: float = 1.0):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            backend=backend)

        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.skeleton = skeleton
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight
        # Set default value. When calling
        # `PoseLocalVisualizer().set_dataset_meta(xxx)`,
        # it will override the default value.
        self.dataset_meta = {}

    @staticmethod
    def draw_angle_data_table(self,
                              image: np.ndarray,
                              person: PosePerson):
        vis_height, vis_width = image.shape[:2]

        plt.ioff()

        fig, ax = plt.subplots(1, 1, figsize=(vis_width * 0.01, vis_height * 0.01))

        left_hip_angle = person.get_joint_angle(person.get_left_hip_angle_keypoints())
        right_hip_angle = person.get_joint_angle(person.get_right_hip_angle_keypoints())
        left_knee_angle = person.get_joint_angle(person.get_left_knee_angle_keypoints())
        right_knee_angle = person.get_joint_angle(person.get_right_knee_angle_keypoints())
        left_ankle_angle = person.get_joint_angle(person.get_left_ankle_angle_keypoints())
        right_ankle_angle = person.get_joint_angle(person.get_right_ankle_angle_keypoints())

        data = np.array([[left_hip_angle], [right_hip_angle], [left_knee_angle], [right_knee_angle], [left_ankle_angle], [right_ankle_angle]])
        columns = "Angle"
        rows = ("Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle")
        # ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=data, colLabels=columns, colLoc='center', rowLabels=rows, rowLoc='center',  loc='center')

        # convert figure to numpy array
        fig.tight_layout()
        fig.canvas.draw()

        pred_img_data = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8)

        if not pred_img_data.any():
            pred_img_data = np.full((vis_height, vis_width, 3), 255)
        else:
            width, height = fig.get_size_inches() * fig.get_dpi()
            pred_img_data = pred_img_data.reshape(
                int(height),
                int(width), 3)

        plt.close(fig)
        return pred_img_data

    def set_dataset_meta(self,
                         dataset_meta: Dict,
                         skeleton_style: str = 'mmpose'):
        """Assign dataset_meta to the visualizer. The default visualization
        settings will be overridden.

        Args:
            dataset_meta (dict): meta information of dataset.
            :param skeleton_style:
        """
        if dataset_meta.get(
                'dataset_name') == 'coco' and skeleton_style == 'openpose':
            dataset_meta = parse_pose_metainfo(
                dict(from_file='configs/_base_/datasets/coco_openpose.py'))

        if isinstance(dataset_meta, dict):
            self.dataset_meta = dataset_meta.copy()
            self.bbox_color = dataset_meta.get('bbox_color', self.bbox_color)
            self.kpt_color = dataset_meta.get('keypoint_colors',
                                              self.kpt_color)
            self.link_color = dataset_meta.get('skeleton_link_colors',
                                               self.link_color)
            self.skeleton = dataset_meta.get('skeleton_links', self.skeleton)
        # sometimes self.dataset_meta is manually set, which might be None.
        # it should be converted to a dict at these times
        if self.dataset_meta is None:
            self.dataset_meta = {}

    def _draw_instances_bbox(self, image: np.ndarray,
                             instances: InstanceData) -> np.ndarray:
        """Draw bounding boxes and corresponding labels of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            self.draw_bboxes(
                bboxes,
                edge_colors=self.bbox_color,
                alpha=self.alpha,
                line_widths=self.line_width)
        else:
            return self.get_image()

        if 'labels' in instances and self.text_color is not None:
            classes = self.dataset_meta.get('classes', None)
            labels = instances.labels

            positions = bboxes[:, :2]
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'

                if isinstance(self.bbox_color,
                              tuple) and max(self.bbox_color) > 1:
                    facecolor = [c / 255.0 for c in self.bbox_color]
                else:
                    facecolor = self.bbox_color

                self.draw_texts(
                    label_text,
                    pos,
                    colors=self.text_color,
                    font_sizes=int(13 * scales[i]),
                    vertical_alignments='bottom',
                    bboxes=[{
                        'facecolor': facecolor,
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    def _draw_instances_kpts(self,
                             image: np.ndarray,
                             person: PosePerson,
                             instances: InstanceData,
                             kpt_thr: float = 0.3,
                             show_kpt_idx: bool = False,
                             skeleton_style: str = 'mmpose'):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        self.set_image(image)
        img_h, img_w, _ = image.shape

        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)

            if 'keypoint_scores' in instances:
                scores = instances.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])


            for kpts, score, visible in zip(keypoints, scores,
                                            keypoints_visible):
                kpts = np.array(kpts, copy=False)

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                # draw links
                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        body_part1 = person.body_parts[sk[0]]
                        body_part2 = person.body_parts[sk[1]]
                        if body_part1.not_visible and body_part2.not_visible:
                            continue

                        if (body_part1.visible_within_image(img_h, img_w)
                                or body_part2.visible_within_image(img_h, img_w)
                                or body_part1.score < kpt_thr
                                or body_part2.score < kpt_thr
                                or MMPoseKeypoint.is_face_id(body_part1.skeleton_id)
                                or MMPoseKeypoint.is_face_id(body_part2.skeleton_id)
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue

                        x = np.array((int(body_part1.keypoint[0]), int(body_part2.keypoint[0])))
                        y = np.array((int(body_part1.keypoint[1]), int(body_part2.keypoint[1])))

                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            color = tuple(int(c) for c in color)

                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0, min(1, 0.5 * (body_part1.score + body_part2.score)))

                        self.draw_lines(x, y, color, line_widths=self.line_width)

                # draw each point on image
                for bid, body_part in enumerate(person.body_parts):
                    if body_part.score < kpt_thr or body_part.not_visible or kpt_color[bid] is None:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[bid]
                    kp_x = body_part.keypoint[0]
                    kp_y = body_part.keypoint[1]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, body_part.score))
                    self.draw_circles(
                        body_part.keypoint,
                        radius=np.array([self.radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=self.radius)
                    if show_kpt_idx:
                        kp_x += self.radius
                        kp_y -= self.radius
                        self.draw_texts(
                            str(bid),
                            body_part.keypoint,
                            colors=color,
                            font_sizes=self.radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')

        return self.get_image()

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: PoseDataSample,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_heatmap: bool = False,
                       draw_bbox: bool = False,
                       show_kpt_idx: bool = True,
                       skeleton_style: str = 'mmpose',
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       kpt_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier
            image (np.ndarray): The image to draw
            data_sample (:obj:`PoseDataSample`, optional): The data sample
                to visualize
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            draw_pred (bool): Whether to draw Prediction PoseDataSample.
                Defaults to ``True``
            draw_bbox (bool): Whether to draw bounding boxes. Default to
                ``False``
            draw_heatmap (bool): Whether to draw heatmaps. Defaults to
                ``False``
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``
            show (bool): Whether to display the drawn image. Default to
                ``False``
            wait_time (float): The interval of show (s). Defaults to 0
            out_file (str): Path to output file. Defaults to ``None``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            step (int): Global step value to record. Defaults to 0
        """

        gt_img_data = None
        pred_img_data = None
        person = PosePerson(image,
                            data_sample.pred_instances,
                            self.skeleton,
                            self.dataset_meta.get('keypoint_id2name'),
                            'mmpose')
        person.build()

        if draw_pred:
            pred_img_data = image.copy()
            # draw bboxes & keypoints
            if 'pred_instances' in data_sample:
                pred_img_data = self._draw_instances_kpts(
                    pred_img_data, person, data_sample.pred_instances, kpt_thr,
                    show_kpt_idx, skeleton_style)
                if draw_bbox:
                    pred_img_data = self._draw_instances_bbox(
                        pred_img_data, data_sample.pred_instances)

        pred_table_data = self.draw_angle_data_table(self,
                                                     image.copy(),
                                                     person)

        # merge visualization results
        if gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = np.concatenate((pred_img_data, pred_table_data), axis=1)
            # drawn_img = pred_img_data

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            # save drawn_img to backends
            self.add_image(name, drawn_img, step)

        return self.get_image()
