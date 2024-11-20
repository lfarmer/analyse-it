import math
from typing import Tuple

import cv2
import numpy as np
from mmengine.structures import InstanceData

from visualisers.body_part import BodyPart
from visualisers.mmpose_keypoint import MMPoseKeypoint


class PosePerson:
    def __init__(self,
                 image: np.ndarray,
                 keypoints: InstanceData,
                 keypoints_visible,
                 scores,
                 bbox,
                 skeleton,
                 skeleton_id2name,
                 skeleton_style):
        self.image = image
        self.keypoints = keypoints
        self.keypoints_visible = keypoints_visible
        self.scores = scores
        self.bbox = bbox
        self.skeleton = skeleton
        self.skeleton_id2name = skeleton_id2name
        self.skeleton_style = skeleton_style
        self.body_parts = []

    def build(self):

        img_h, img_w, _ = self.image.shape

        kpts = np.array(self.keypoints, copy=False)
        for sk_id, sk in enumerate(self.skeleton_id2name):
            self.body_parts.append(
                    BodyPart(name=self.skeleton_id2name[sk_id], skeleton_id=sk_id, keypoint=(kpts[sk_id, 0], kpts[sk_id, 1]), score=self.scores[sk_id], not_visible=not self.keypoints_visible[sk_id]))

    def get_body_part(self, mmpose_keypoint: MMPoseKeypoint):
        return self.body_parts[mmpose_keypoint.value]

    def get_size(self):
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @staticmethod
    def get_joint_angle(joints: Tuple[BodyPart, BodyPart, BodyPart]):
        ang = math.degrees(
            math.atan2(joints[2].keypoint[1] - joints[1].keypoint[1], joints[2].keypoint[0] - joints[1].keypoint[0]) - math.atan2(joints[0].keypoint[1] - joints[1].keypoint[1],
                                                                                      joints[0].keypoint[0] - joints[1].keypoint[0]))
        return int(ang + 360) if ang < 0 else int(ang)

    def is_facing_left(self):
        return self.get_body_part(MMPoseKeypoint.NOSE).keypoint[0] < self.get_body_part(MMPoseKeypoint.RIGHT_EAR).keypoint[0]

    def get_left_hip_angle_keypoints(self):
        return [self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE), self.get_body_part(
            MMPoseKeypoint.LEFT_HIP), self.get_body_part(
            MMPoseKeypoint.LEFT_SHOULDER)] if self.is_facing_left() else [self.get_body_part(
            MMPoseKeypoint.LEFT_SHOULDER), self.get_body_part(MMPoseKeypoint.LEFT_HIP), self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE)]

    def get_right_hip_angle_keypoints(self):
        return [self.get_body_part(MMPoseKeypoint.RIGHT_KNEE), self.get_body_part(
            MMPoseKeypoint.RIGHT_HIP), self.get_body_part(
            MMPoseKeypoint.RIGHT_SHOULDER)] if self.is_facing_left() else [self.get_body_part(
            MMPoseKeypoint.RIGHT_SHOULDER), self.get_body_part(MMPoseKeypoint.RIGHT_HIP), self.get_body_part(
            MMPoseKeypoint.RIGHT_KNEE)]

    def get_left_knee_angle_keypoints(self):
        return [self.get_body_part(MMPoseKeypoint.LEFT_HIP), self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE), self.get_body_part(
            MMPoseKeypoint.LEFT_ANKLE)] if self.is_facing_left() else [self.get_body_part(
            MMPoseKeypoint.LEFT_ANKLE), self.get_body_part(MMPoseKeypoint.LEFT_KNEE), self.get_body_part(
            MMPoseKeypoint.LEFT_HIP)]

    def get_right_knee_angle_keypoints(self):
        return [self.get_body_part(MMPoseKeypoint.RIGHT_HIP), self.get_body_part(
            MMPoseKeypoint.RIGHT_KNEE), self.get_body_part(
            MMPoseKeypoint.RIGHT_ANKLE)] if self.is_facing_left() else [self.get_body_part(
            MMPoseKeypoint.RIGHT_ANKLE), self.get_body_part(MMPoseKeypoint.RIGHT_KNEE), self.get_body_part(
            MMPoseKeypoint.RIGHT_HIP)]

    def get_left_ankle_angle_keypoints(self):
        return [self.get_body_part(MMPoseKeypoint.LEFT_BIG_TOE), self.get_body_part(
            MMPoseKeypoint.LEFT_ANKLE), self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE)] if self.is_facing_left() else [self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE), self.get_body_part(MMPoseKeypoint.LEFT_ANKLE), self.get_body_part(
            MMPoseKeypoint.LEFT_BIG_TOE)]

    def get_right_ankle_angle_keypoints(self):
        return [self.get_body_part(MMPoseKeypoint.RIGHT_BIG_TOE), self.get_body_part(
            MMPoseKeypoint.RIGHT_ANKLE), self.get_body_part(
            MMPoseKeypoint.RIGHT_KNEE)] if self.is_facing_left() else [self.get_body_part(
            MMPoseKeypoint.RIGHT_KNEE), self.get_body_part(MMPoseKeypoint.RIGHT_ANKLE), self.get_body_part(
            MMPoseKeypoint.RIGHT_BIG_TOE)]
