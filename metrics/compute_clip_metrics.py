"""
Class for computing CLIP Metrics on LFW and Mbrush data
"""
import json
from PIL import Image
from tqdm.auto import tqdm
import torch
import torchvision.transforms as transforms
from einops import rearrange
import numpy as np
from clip_similarity import ClipSimilarity
from typing import List, Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

transform = transforms.Compose([
    transforms.PILToTensor()
])


class ComputeMetrics():
    def __init__(self, source_imgs_folder: str, source_caption_file: str, filtered_indices_file: str, data="mbrush",
                 name: str = "ViT-L/14", device="cuda:1", download_path="./"):
        """
        Loads the model, and other necessary initialization variables
        :param source_imgs_folder: Path to the source image folder
        :param source_caption_file: Path to the source image captions json file
        :param filtered_indices_file: Path to the Mbrush filtered indices file
        :param data : Identifier required for which dataset we are computing CLIP metrics, "mbrush" or "lfw" or "lfw80"
        :param name: Name of the CLIP model to be loaded
        :param device: Device to store the CLIP model
        :param download_path: Path where the CLIP model will be downloaded to
        """
        self.clip_similarity = ClipSimilarity(name=name, device=device, download_path=download_path).to(device)
        print(f"CLIP model loaded to device: {device} and stored in {download_path}")
        self.device = device
        self.load_source_utils(source_imgs_folder, source_caption_file, filtered_indices_file)
        self.all_metrics : List[Dict] = []
        self.data=data
    def load_source_utils(self, source_imgs_folder: str, source_caption_file: str, filtered_indices_file: str):
        """
        Loads source captions and other necessary initializations are taken care here
        """
        self.source_imgs_folder_path = source_imgs_folder
        self.source_caption_file_path = source_caption_file
        self.filtered_indices_file_path = filtered_indices_file
        self.source_captions = self.load_json(self.source_caption_file_path)  # dict
        self.filtered_indices = self.load_json(self.filtered_indices_file_path)

    def load_json(self, json_path):
        """
        Helper to load json file, takes the path to the json file and returns the json object
        """
        with open(json_path) as fin:
            json_obj = json.load(fin)
        return json_obj

    def load_utils(self, target_imgs_folder: str, target_caption_file: str):
        """
        Loads target related utilities
        """
        self.target_imgs_folder_path = target_imgs_folder
        self.target_caption_file_path = target_caption_file
        self.target_captions = self.load_json(self.target_caption_file_path)  # dict

    def load_image_pil(self, image_path):
        """
        Helper to load the image according to CLIP requirements
        Takes Image path and returns a torch tensor of shape (B,C,H,W) normalized to [0,1]
        """
        img = Image.open(image_path).convert("RGB")
        image_0 = img.resize((256, 256), Image.Resampling.LANCZOS)
        # print(np.asarray(image_0)[0])
        # return rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w").unsqueeze(0)
        # #[0,1] gives better results
        return rearrange(torch.tensor(np.array(image_0)).float() / 255, "h w c -> c h w").unsqueeze(0)

    def compute_clip_metrics(self, target_imgs_folder: str,
                             target_caption_file: str,
                             gc : str,
                        ):  # We compute for all the 115 images in the filtered Mbrush dataset
        """
        Computes the four CLIP metrics and appends them to the list all_metrics
        :param target_imgs_folder: Path to target images folder
        :param target_caption_file: Path to target caption file
        :param gc:
        :return:
        """
        self.load_utils(target_imgs_folder, target_caption_file)
        self.num_samples = len(self.source_captions)
        self.sim_0_avg = 0
        self.sim_1_avg = 0
        self.sim_direction_avg = 0
        self.sim_image_avg = 0
        self.count = 0

        pbar = tqdm(total=self.num_samples)
        if self.data == "mbrush":
            for i in self.filtered_indices:
                # print(f"Computing CLIP metrics for the image {i}.png")
                source_img = self.load_image_pil(f"{self.source_imgs_folder_path}/{i}.png")
                target_img = self.load_image_pil(f"{self.target_imgs_folder_path}/{i}.png")

                sim_0, sim_1, sim_direction, sim_image = self.clip_similarity(
                    source_img.to(self.device), target_img.to(self.device),
                    [self.source_captions[f"{str(i)}"].strip()],
                    [self.target_captions[f"{str(i)}.png"].strip()]
                )

                self.sim_0_avg += sim_0.item()
                self.sim_1_avg += sim_1.item()
                self.sim_direction_avg += sim_direction.item()
                self.sim_image_avg += sim_image.item()
                self.count += 1
                pbar.update(self.count)
        elif self.data== "lfw80":
            for img_name in self.filtered_indices:
                # print(f"Computing CLIP metrics for the image {i}.png")
                person_name = "_".join(img_name.split("_")[:-1])
                source_img = self.load_image_pil(f"{self.source_imgs_folder_path}/{person_name}/{img_name}")
                target_img = self.load_image_pil(f"{self.target_imgs_folder_path}/{person_name}/{img_name}")

                sim_0, sim_1, sim_direction, sim_image = self.clip_similarity(
                    source_img.to(self.device), target_img.to(self.device),
                    [self.source_captions[f"{img_name}"].strip()],
                    [self.target_captions[f"{img_name}"].strip()]
                )

                self.sim_0_avg += sim_0.item()
                self.sim_1_avg += sim_1.item()
                self.sim_direction_avg += sim_direction.item()
                self.sim_image_avg += sim_image.item()
                self.count += 1
                pbar.update(self.count)
        pbar.close()

        metrics_dict = {f"guidance_combination": gc,
                        f"clip_sim_source": self.sim_0_avg / self.count,
                        f"clip_sim_target": self.sim_1_avg / self.count,
                        f"clip_image-to-image": self.sim_image_avg / self.count,
                        f"clip_directional": self.sim_direction_avg / self.count
                        }
        self.all_metrics.append((metrics_dict))
        # print(self.sim_0_avg / count, self.sim_1_avg / count, self.sim_direction_avg / count,
        # self.sim_image_avg / count)

    def write_metrics_to_json_csv_and_plot(self, json_path: str, plot_folder : str):
        """
        1.Creates a json file of list of dicts, each dict containing the metrics of each guidance combination run.
        Stored at json_path.
        2. CSV file stored in the same location as the json file, with same name
        3. Text guidance and Image guidance specific plots of the clip directional similarity and image-to-image similarity
            Figures of the plots are stored at plot_folder and named as text_guidance_plot.png and image_guidance_plot.png
        :param json_path: Full path to the json file
        :param plot_folder: Folder where image and text guidance plots are created
        """
        #Path(f"{json_path}.json").touch(exist_ok=True)
        with open(f"{json_path}.json", 'w') as fout:
            json.dump(self.all_metrics, fout)

        df = pd.DataFrame(self.all_metrics)
        df['text_guidance'] = df['guidance_combination'].apply(lambda x: x.split("_")[-2])
        df['image_guidance'] = df['guidance_combination'].apply(lambda x: x.split("_")[-1])
        """
        if self.data == "lfw80":
            df['text_guidance'] = [str(x) for x in np.arange(4.0, 8.5, 0.5).tolist() * 3]
            df['image_guidance'] = [str(1.0)] * 9 + [str(1.5)] * 9 + [str(2.0)] * 9
        elif self.data == "mbrush":
            df['text_guidance'] = [str(x) for x in np.arange(4.0, 8.0, 0.5).tolist() * 3]
            df['image_guidance'] = [str(1.0)] * 8 + [str(1.5)] * 8 + [str(2.0)] * 8
        """
        df.to_csv(f"{json_path}.csv", sep='\t', index=False)
        plt.figure()
        tg_plot = sns.lineplot(df, x="clip_directional", y="clip_image-to-image", hue="text_guidance")
        tg_plot.figure.suptitle(f"Attribute {json_path.split('/')[-1]}, Text_Guidance wise")
        fig = tg_plot.get_figure()
        fig.savefig(f"{plot_folder}/text_guidance_plot.png")
        plt.figure()
        ig_plot = sns.lineplot(df, x="clip_directional", y="clip_image-to-image", hue="image_guidance")
        ig_plot.figure.suptitle(f"Attribute {json_path.split('/')[-1]}, Image_Guidance wise")
        fig = ig_plot.get_figure()
        fig.savefig(f"{plot_folder}/image_guidance_plot.png")

    def write_metrics_to_json_csv(self, json_path: str, plot_folder : str):
        """
        1.Creates a json file of list of dicts, each dict containing the metrics of each guidance combination run.
        Stored at json_path.
        2. CSV file stored in the same location as the json file, with same name
        3. Text guidance and Image guidance specific plots of the clip directional similarity and image-to-image similarity
            Figures of the plots are stored at plot_folder and named as text_guidance_plot.png and image_guidance_plot.png
        :param json_path: Full path to the json file
        :param plot_folder: Folder where image and text guidance plots are created
        """
        #Path(f"{json_path}.json").touch(exist_ok=True)
        with open(f"{json_path}.json", 'w') as fout:
            json.dump(self.all_metrics, fout)

        df = pd.DataFrame(self.all_metrics)
        df['text_guidance'] = df['guidance_combination'].apply(lambda x: x.split("_")[-2])
        df['image_guidance'] = df['guidance_combination'].apply(lambda x: x.split("_")[-1])
        df.to_csv(f"{json_path}.csv", sep='\t', index=False)


    def reset_all_metrics_list(self):
        """
        Resets the all_metrics list to an empty list, useful for lfw runs as each attribute requires a plot
        """
        self.all_metrics=[]

class SEGAComputeMetrics(ComputeMetrics):
    def __init__(self, name: str = "ViT-L/14", device="cuda:1", download_path="./"):
        self.clip_similarity = ClipSimilarity(name=name, device=device, download_path=download_path).to(device)
        self.device=device
        print(f"CLIP model loaded to device: {device} and stored in {download_path}")

    def compute_clip_metrics_individual(self, source_img_path: object, target_img_path: object,
                                        source_caption: object, target_caption: object) -> object:

        source_img=self.load_image_pil(source_img_path)
        target_img=self.load_image_pil(target_img_path)
        sim_0, sim_1, sim_direction, sim_image = self.clip_similarity(
            source_img.to(self.device), target_img.to(self.device),
            [source_caption],
            [target_caption]
        )

        return sim_0.item(), sim_1.item(), sim_direction.item(), sim_image.item()

    def compute_clip_edit_similarity(self, img_path : str, edit_instruction : str):
        img_features=self.clip_similarity.encode_image(self.load_image_pil(img_path).to(self.device))
        edit_instruction_features=self.clip_similarity.encode_text(edit_instruction)

        return F.cosine_similarity(img_features, edit_instruction_features).item()
