from mmpose.apis import MMPoseInferencer
from argparse import ArgumentParser

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

    # instantiate the inferencer using the model alias
    inferencer = MMPoseInferencer(pose2d='body26', show_progress=True, device='cpu')

    # The MMPoseInferencer API employs a lazy inference approach,
    # creating a prediction generator when given input
    result_generator = inferencer(args.input, out_dir="output/video")
    results = [result for result in result_generator]

if __name__ == '__main__':
    main()
