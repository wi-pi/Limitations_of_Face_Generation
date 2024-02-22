import json
import argparse
import os
import torch.nn.functional as F
import glob

from compute_clip_metrics import SEGAComputeMetrics
from l1l2 import eval_distance
from PIL import Image
from dinov2_similarity import Dinov2Similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-images-path", type=str, required=True)
    parser.add_argument("--source-images-path", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True,
                        help="Full path to the json file to store the metrics, without extension")
    args = parser.parse_args()

    # Load the clip model
    cm = SEGAComputeMetrics(download_path=".")
    dino = Dinov2Similarity()

    all_attributes = ['facial_hair_0','smile_0','young_0','old_0','sunglasses_0'] 
    attribute_for_edit_clip_metric={ "smile": "smiling",
                                    "sunglasses": "sunglasses",
                                    "young": "youthful face",
                                    "old": "elderly face",
                                    "facial_hair": "face with facial hair"
                                }


    demographics = ['asian_man',
                    'asian_woman',
                    'white_man',
                    'white_woman',
                    'black_man',
                    'black_woman',
                    'indian_man',
                    'indian_woman']

    all_metrics = {}
    for attr in all_attributes:
        attribute = '_'.join(attr.split("_")[:-1])
        print(f"Getting metrics for the attribute {attribute}")
        for demo in demographics:
            print(f"Now for the demographic {demo}")
            race = demo.split("_")[0]
            gender = demo.split("_")[1]
            # Construct the template captions
            if race == "asian" or race == "indian":
                source_caption = f"A photo of the face of an {race} {gender}"
                if attribute in ['cleanshaven', 'old', 'young']:
                    target_caption = f"A photo of the face of a cleanshaven {race} {gender}"
                elif attribute == 'male':
                    target_caption = f"A photo of the face of an {race} man"
                elif attribute == 'female':
                    target_caption = f"A photo of the face of an {race} woman"
                elif attribute == 'no_operation':
                    target_caption = source_caption
                else:
                    target_caption = f"A photo of the face of an {race} {gender} with {' '.join(attribute.split('_'))}"
            else:
                source_caption = f"A photo of the face of a {race} {gender}"
                if attribute in ['cleanshaven', 'old', 'young']:
                    target_caption = f"A photo of the face of a cleanshaven {race} {gender}"
                elif attribute == 'male':
                    target_caption = f"A photo of the face of a {race} man"
                elif attribute == 'female':
                    target_caption = f"A photo of the face of a {race} woman"
                elif attribute == 'no_operation':
                    target_caption = source_caption
                else:
                    target_caption = f"A photo of the face of a {race} {gender} with {' '.join(attribute.split('_'))}"

            names = os.listdir(f"{args.source_images_path}/{demo}")

            for name in names:
                num=len(glob.glob(f"{args.source_images_path}/{demo}/{name}/*.jpg"))
                for j in range(num):
                    src_image_path=f"{args.source_images_path}/{demo}/{name}/{name}_000{j}.jpg"
                    tgt_image_path=f"{args.target_images_path}/{attribute}/{demo}/{name}/{name}_000{j}.jpg"
                    src_clip_score, tgt_clip_score, clip_direction, clip_image = cm.compute_clip_metrics_individual(
                        source_img_path=src_image_path,
                        target_img_path=tgt_image_path,
                        source_caption=source_caption, target_caption=target_caption)
                    edit_clip_similarity=cm.compute_clip_edit_similarity(tgt_image_path,attribute_for_edit_clip_metric[attribute])
                    dino_image_similarity = dino.get_similarity(src_img_path=src_image_path,
                                                                tgt_img_path=tgt_image_path)
                    l1loss = eval_distance(src_image_path=src_image_path,
                                           tgt_image_path=tgt_image_path,
                                           metric='l1')
                    l2loss = eval_distance(src_image_path=src_image_path,
                                           tgt_image_path=tgt_image_path,
                                           metric='l2')
                    all_metrics[f"{attribute}/{demo}/{name}_000{j}"] = dict(src_clip_similarity=src_clip_score,
                                                                     tgt_clip_similarity=tgt_clip_score,
                                                                     clip_direction=clip_direction, clip_image=clip_image,
                                                                     l1loss=l1loss, l2loss=l2loss,
                                                                     dino_image_similarity=dino_image_similarity,
                                                                     clip_edit_image_similarity=edit_clip_similarity)

    with open(args.json_path, 'w') as fout:
        json.dump(all_metrics, fout)

