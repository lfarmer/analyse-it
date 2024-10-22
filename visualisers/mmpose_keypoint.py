from enum import Enum

class MMPoseKeypoint(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    HEAD = 17
    NECK = 18
    HIP = 19
    LEFT_BIG_TOE = 20
    RIGHT_BIG_TOE = 21
    LEFT_SMALL_TOE = 22
    RIGHT_SMALL_TOE = 23
    LEFT_HEAL = 24
    RIGHT_HEAL = 25

    @staticmethod
    def is_face_id(skeleton_id: int):
        return skeleton_id in [MMPoseKeypoint.NOSE.value, MMPoseKeypoint.LEFT_EYE.value, MMPoseKeypoint.RIGHT_EYE.value]