import logging
from mmengine.logging import print_log
from mmpose.apis import MMPoseInferencer

img_path = 'demo/image/demo.jpeg'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer(pose2d='body26', show_progress=True, device='cpu')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, out_dir="demo/image")
result = min(result_generator)
