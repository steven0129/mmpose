# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Dict
from PIL import Image
from datetime import datetime

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases

import random
import os
import numpy as np
import json
import copy

filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        help='Input image/video path or folder path.')

    # init args
    parser.add_argument(
        '--pose2d',
        type=str,
        default=None,
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose2d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--pose3d',
        type=str,
        default=None,
        help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose3d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Display the progress bar during inference.')

    # The default arguments for prediction filtering differ for top-down
    # and bottom-up models. We assign the default arguments according to the
    # selected pose2d model
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break

    # call args
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='Whether to draw the bounding boxes.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=filter_args['bbox_thr'],
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--pose-based-nms',
        type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'],
        help='Whether to use pose-based NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization.')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--black-background',
        action='store_true',
        help='Plot predictions on a black image')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        default='',
        help='Directory for saving visualized results.')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default='',
        help='Directory for saving inference results.')
    parser.add_argument(
        '--show-alias',
        action='store_true',
        help='Display all the available model aliases.')
    parser.add_argument(
        '--save-as-coco',
        type=str,
        default=None,
        help='Folder path to save prediction result as COCO')
    parser.add_argument(
        '--return-vis',
        action='store_true',
        default=False
    )

    call_args = vars(parser.parse_args())

    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights',
        'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    display_alias = call_args.pop('show_alias')

    return init_args, call_args, display_alias


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    """Display the available model aliases and their corresponding model
    names."""
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f'{"ALIAS".ljust(max_alias_length+2)}MODEL_NAME')
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length+2)}{model_aliases[alias]}')


def main():
    init_args, call_args, display_alias = parse_args()
    if display_alias:
        model_alises = get_model_aliases(init_args['scope'])
        display_model_aliases(model_alises)
    else:
        inferencer = MMPoseInferencer(**init_args)
        
        if(call_args['save_as_coco'] is not None):
            output_folder = call_args['save_as_coco']
            output_training_annotations = {
                "info": {
                    "description": "VirtualMouse",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "date_created": datetime.now().strftime('%Y/%m/%d')
                },
                "licenses": "",
                "images": [],
                "annotations": [],
                "categories": [{
                    "supercategory": "hand",
                    "id": 1,
                    "name": "hand",
                    "keypoints": [
                        "wrist",
                        "thumb1",
                        "thumb2",
                        "thumb3",
                        "thumb4",
                        "forefinger1",
                        "forefinger2",
                        "forefinger3",
                        "forefinger4",
                        "middle_finger1",
                        "middle_finger2",
                        "middle_finger3",
                        "middle_finger4",
                        "ring_finger1",
                        "ring_finger2",
                        "ring_finger3",
                        "ring_finger4",
                        "pinky_finger1",
                        "pinky_finger2",
                        "pinky_finger3",
                        "pinky_finger4"
                    ]
                }]
            }

            output_val_annotations = copy.deepcopy(output_training_annotations)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                os.makedirs(f'{output_folder}/images')
                os.makedirs(f'{output_folder}/annotations')
                os.makedirs(f'{output_folder}/vis')

                output_data = []

                for idx, (ori_inputs, _, results) in enumerate(inferencer(**call_args)):
                    filename = ''.join(call_args['inputs'].split('.')[:-1]) + f'-{idx}.png'
                    input_img = Image.fromarray(ori_inputs[0].astype(np.uint8)[..., ::-1])
                    
                    if(len(results['predictions']) > 0):
                        keypoints = results['predictions'][0][0]['keypoints']
                        bbox = results['predictions'][0][0]['bbox'][0]
                        visualization = Image.fromarray(results['visualization'][0])
                        new_keypoints = []

                        for point in keypoints:
                            new_keypoints.extend(point)
                            new_keypoints.extend([1])

                        visualization.save(f'{output_folder}/vis/{filename}')
                        input_img.save(f'{output_folder}/images/{filename}')
                        output_data.append({
                            "images": {
                                "file_name": f'images/{filename}',
                                "height": ori_inputs[0].shape[0],
                                "width": ori_inputs[0].shape[1],
                                "id": idx
                            },
                            "annotations": {
                                "bbox": bbox,
                                "keypoints": new_keypoints,
                                "category_id": 1,
                                "id": idx,
                                "image_id": idx
                            }
                        })

                random.Random(0).shuffle(output_data)

                training_num = int(len(output_data) * 0.8)
                training_data = output_data[:training_num]
                val_data = output_data[training_num:]

                output_training_annotations['images'] = list(map(lambda x: x['images'], training_data))
                output_training_annotations['annotations'] = list(map(lambda x: x['annotations'], training_data))
                output_val_annotations['images'] = list(map(lambda x: x['images'], val_data))
                output_val_annotations['annotations'] = list(map(lambda x: x['annotations']), val_data)


                with open(f'{output_folder}/annotations/training_annotations.json', 'w', encoding='utf-8') as F:
                    json.dump(output_training_annotations, F, ensure_ascii=False, indent=4)

                with open(f'{output_folder}/annotations/val_annotations.json', 'w', encoding='utf-8') as F:
                    json.dump(output_val_annotations, F, ensure_ascii=False, indent=4)

            else:
                print(f'{output_folder} has already existed.')
        else:
            for _ in inferencer(**call_args):
                pass


if __name__ == '__main__':
    main()
