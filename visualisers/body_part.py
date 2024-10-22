class BodyPart:
    def __init__(self,
                 name,
                 skeleton_id,
                 keypoint,
                 score,
                 not_visible):
        self.name = name
        self.skeleton_id = skeleton_id
        self.keypoint = keypoint
        self.score = score
        self.not_visible = not_visible

    def visible_within_image(self, image_height, image_width):
        return (self.keypoint[0] <= 0 or self.keypoint[0] >= image_width
                or self.keypoint[1] <= 0 or self.keypoint[1] >= image_height)

    def keypoints_to_int(self):
        return int(self.keypoint[0]), int(self.keypoint[1])