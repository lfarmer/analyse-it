import math
from typing import Tuple

import numpy as np
from mmengine.structures import InstanceData

from visualisers.body_part import BodyPart
from visualisers.mmpose_keypoint import MMPoseKeypoint


class PosePerson:
    def __init__(self,
                 image: np.ndarray,
                 instances: InstanceData,
                 skeleton,
                 skeleton_id2name,
                 skeleton_style):
        self.image = image
        self.instances = instances
        self.skeleton = skeleton
        self.skeleton_id2name = skeleton_id2name
        self.skeleton_style = skeleton_style
        self.body_parts = []

    def build(self):

        img_h, img_w, _ = self.image.shape

        if 'keypoints' in self.instances:
            keypoints = self.instances.get('transformed_keypoints',
                                           self.instances.keypoints)

            if 'keypoint_scores' in self.instances:
                scores = self.instances.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if 'keypoints_visible' in self.instances:
                keypoints_visible = self.instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            for kpts, score, visible in zip(keypoints, scores,
                                            keypoints_visible):
                kpts = np.array(kpts, copy=False)

                # draw links
                for sk_id, sk in enumerate(self.skeleton_id2name):
                    self.body_parts.append(
                            BodyPart(name=self.skeleton_id2name[sk_id], skeleton_id=sk_id, keypoint=(kpts[sk_id, 0], kpts[sk_id, 1]), score=score[sk_id], not_visible=not visible[sk_id]))

    def get_body_part(self, mmpose_keypoint: MMPoseKeypoint):
        return self.body_parts[mmpose_keypoint.value]

    @staticmethod
    def get_joint_angle(joints: Tuple[BodyPart, BodyPart, BodyPart]):
        ang = math.degrees(
            math.atan2(joints[2].keypoint[1] - joints[1].keypoint[1], joints[2].keypoint[0] - joints[1].keypoint[0]) - math.atan2(joints[0].keypoint[1] - joints[1].keypoint[1],
                                                                                      joints[0].keypoint[0] - joints[1].keypoint[0]))
        return int(ang + 360) if ang < 0 else int(ang)

    def is_facing_left(self):
        if (self.get_body_part(MMPoseKeypoint.LEFT_EAR).not_visible):
            return self.get_body_part(MMPoseKeypoint.NOSE).keypoint[0] < \
                self.get_body_part(MMPoseKeypoint.RIGHT_EAR).keypoint[0]
        if (self.get_body_part(MMPoseKeypoint.RIGHT_EAR).not_visible):
            return self.get_body_part(MMPoseKeypoint.NOSE).keypoint[0] > \
                self.get_body_part(MMPoseKeypoint.LEFT_EAR).keypoint[0]

    def get_left_hip_angle_keypoints(self):
        return self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE), self.get_body_part(
            MMPoseKeypoint.LEFT_HIP), self.get_body_part(
            MMPoseKeypoint.LEFT_SHOULDER) if self.is_facing_left() else self.get_body_part(
            MMPoseKeypoint.LEFT_SHOULDER), self.get_body_part(MMPoseKeypoint.LEFT_HIP), self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE)

    def get_right_hip_angle_keypoints(self):
        return self.get_body_part(MMPoseKeypoint.RIGHT_KNEE), self.get_body_part(
            MMPoseKeypoint.RIGHT_HIP), self.get_body_part(
            MMPoseKeypoint.RIGHT_SHOULDER) if self.is_facing_left() else self.get_body_part(
            MMPoseKeypoint.RIGHT_SHOULDER), self.get_body_part(MMPoseKeypoint.RIGHT_HIP), self.get_body_part(
            MMPoseKeypoint.RIGHT_KNEE)

    def get_left_knee_angle_keypoints(self):
        return self.get_body_part(MMPoseKeypoint.LEFT_HIP), self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE), self.get_body_part(
            MMPoseKeypoint.LEFT_ANKLE) if self.is_facing_left() else self.get_body_part(
            MMPoseKeypoint.LEFT_ANKLE), self.get_body_part(MMPoseKeypoint.LEFT_KNEE), self.get_body_part(
            MMPoseKeypoint.LEFT_HIP)

    def get_right_knee_angle_keypoints(self):
        return self.get_body_part(MMPoseKeypoint.RIGHT_HIP), self.get_body_part(
            MMPoseKeypoint.RIGHT_KNEE), self.get_body_part(
            MMPoseKeypoint.RIGHT_ANKLE) if self.is_facing_left() else self.get_body_part(
            MMPoseKeypoint.RIGHT_ANKLE), self.get_body_part(MMPoseKeypoint.RIGHT_KNEE), self.get_body_part(
            MMPoseKeypoint.RIGHT_HIP)

    def get_left_ankle_angle_keypoints(self):
        return self.get_body_part(MMPoseKeypoint.LEFT_BIG_TOE), self.get_body_part(
            MMPoseKeypoint.LEFT_ANKLE), self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE) if self.is_facing_left() else self.get_body_part(
            MMPoseKeypoint.LEFT_KNEE), self.get_body_part(MMPoseKeypoint.LEFT_ANKLE), self.get_body_part(
            MMPoseKeypoint.LEFT_BIG_TOE)

    def get_right_ankle_angle_keypoints(self):
        return self.get_body_part(MMPoseKeypoint.RIGHT_BIG_TOE), self.get_body_part(
            MMPoseKeypoint.RIGHT_ANKLE), self.get_body_part(
            MMPoseKeypoint.RIGHT_KNEE) if self.is_facing_left() else self.get_body_part(
            MMPoseKeypoint.RIGHT_KNEE), self.get_body_part(MMPoseKeypoint.RIGHT_ANKLE), self.get_body_part(
            MMPoseKeypoint.RIGHT_BIG_TOE)
