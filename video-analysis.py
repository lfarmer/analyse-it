from mmpose.apis import MMPoseInferencer
from argparse import ArgumentParser

from visualisers import Pose2dWithAnglesVisualizer

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='Video path')

    args = parser.parse_args()
    return args

def main():
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parse_args()

    assert args.input != ''

    inferencer = MMPoseInferencer(pose2d='body26', show_progress=True, device='cpu')
    inferencer.inferencer.visualizer = Pose2dWithAnglesVisualizer(vis_backends=[dict(type='LocalVisBackend')])
    inferencer.inferencer.visualizer.set_dataset_meta(inferencer.inferencer.model.dataset_meta)

    result_generator = inferencer(args.input, out_dir="output/video")
    results = [result for result in result_generator]

if __name__ == '__main__':
    main()
