# https://www.kaggle.com/datasets/eyantraiit/semantic-segmentation-datasets-of-indian-roads

import glob
import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:

    # project_name = "Indian roads"
    images_path = (
        "/home/grokhi/rawdata/indian-roads-semantic-segmentation/Indian_road_data/Indian_road_data/Raw_images"
    )
    masks_path = "/home/grokhi/rawdata/indian-roads-semantic-segmentation/Indian_road_data/Indian_road_data/Masks"
    batch_size = 30
    images_ext = ".jpg"
    masks_ext = "_gtFine_polygons.json"


    def create_ann(image_path):
        labels = []

        masks_file_name = get_file_name(image_path).split("_left")[0] + masks_ext
        subfolder_data = masks_file_name[:5]
        masks_file_path = os.path.join(curr_masks_path, subfolder_data, masks_file_name)
        tag_meta = meta.get_tag_meta(subfolder_data)
        tag = sly.Tag(tag_meta)
        if file_exists(masks_file_path):
            ann_data = load_json_file(masks_file_path)

            img_height = ann_data["imageHeight"]
            img_wight = ann_data["imageWidth"]

            for curr_ann_data in ann_data["shapes"]:
                obj_class = meta.get_obj_class(curr_ann_data["label"])
                polygons_coords = curr_ann_data["points"]
                exterior = []
                for coords in polygons_coords:
                    exterior.append([coords[1], coords[0]])
                    if len(exterior) < 3:
                        continue
                poligon = sly.Polygon(exterior)
                label_poly = sly.Label(poligon, obj_class)
                labels.append(label_poly)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[tag])


    obj_class_road = sly.ObjClass("road", sly.Polygon)
    obj_class_footpath = sly.ObjClass("footpath", sly.Polygon)
    obj_class_pothole = sly.ObjClass("pothole", sly.Polygon)
    obj_class_shallow = sly.ObjClass("shallow", sly.Polygon)

    tag_1_005 = sly.TagMeta("1_005", sly.TagValueType.NONE)
    tag_1_012 = sly.TagMeta("1_012", sly.TagValueType.NONE)
    tag_1_016 = sly.TagMeta("1_016", sly.TagValueType.NONE)
    tag_1_018 = sly.TagMeta("1_018", sly.TagValueType.NONE)
    tag_1_007 = sly.TagMeta("1_007", sly.TagValueType.NONE)
    tag_1_017 = sly.TagMeta("1_017", sly.TagValueType.NONE)
    tag_2_006 = sly.TagMeta("2_006", sly.TagValueType.NONE)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class_road, obj_class_footpath, obj_class_pothole, obj_class_shallow],
        tag_metas=[tag_1_005, tag_1_012, tag_1_016, tag_1_018, tag_1_007, tag_1_017, tag_2_006],
    )
    api.project.update_meta(project.id, meta.to_json())


    for ds_name in os.listdir(images_path):
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        curr_images_path = os.path.join(images_path, ds_name)
        curr_masks_path = os.path.join(masks_path, ds_name)
        images_pathes = glob.glob(curr_images_path + "/*/*.jpg")

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            images_names_batch = [get_file_name_with_ext(im_path) for im_path in img_pathes_batch]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(images_names_batch))
    return project


