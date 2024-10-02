import logging
from mmengine.logging import print_log
from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

# build the model from a config file and a checkpoint file
# cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

cfg_options = None

model = init_model(
    "demo/image/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
    "demo/image/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth",
    device="cpu",
    cfg_options=cfg_options)

# init visualizer
model.cfg.visualizer.radius = 3
model.cfg.visualizer.alpha = 0.8
model.cfg.visualizer.line_width = 1

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(
    model.dataset_meta, skeleton_style="mmpose")

# inference a single image
batch_results = inference_topdown(model, "demo/image/demo.jpeg")
results = merge_data_samples(batch_results)

# show the results
img = imread("demo/image/demo.jpeg", channel_order='rgb')
visualizer.add_datasample(
    'result',
    img,
    data_sample=results,
    draw_gt=False,
    draw_bbox=True,
    kpt_thr=0.3,
    draw_heatmap=False,
    show_kpt_idx=False,
    skeleton_style="mmpose",
    show=False,
    out_file="demo/image/output.jpeg")

print_log(
    f'the output image has been saved at demo/image/output.jpeg',
    logger='current',
    level=logging.INFO)
