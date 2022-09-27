import argparse
import csv
import json
import os
import pathlib
from collections import defaultdict

import cv2
import mmcv
import tqdm
from PIL import Image

data_dir = '../Dataset/ROBINv1.1'
classes = ('aeroplane', 'bicycle',
           'boat', 'bus', 'car', 'chair',
           'diningtable', 'motorbike', 'sofa', 'train')


def csv2coco():
    phases = ['train',
              'iid_test',
              'nuisances/context',
              'nuisances/occlusion',
              'nuisances/pose',
              'nuisances/shape',
              'nuisances/texture',
              'nuisances/weather']
    csv_names = ['train.csv',
                 'iid.csv',
                 'context_bias.csv',
                 'occlusion_bias.csv',
                 'pose_bias.csv',
                 'shape_bias.csv',
                 'texture_bias.csv',
                 'weather_bias.csv']
    print('Starting to convert !')
    for phase, csv_name in zip(phases, csv_names):

        output_json_dict = {"images": [],
                            "annotations": [],
                            "categories": []}
        box_id = 1
        annotations = defaultdict(list)
        with open(f'{data_dir}/{phase}/{csv_name}', 'r') as f:
            csv_data = csv.reader(f, delimiter=',')
            next(csv_data)
            for row in csv_data:
                filename = f'{data_dir}/{phase}/' + row[17]
                annotation = (classes.index(row[1]) + 1, [row[11], row[12], row[13], row[14]])
                annotations[filename].append(annotation)
        for filename, annotation in annotations.items():
            image = Image.open(filename)
            width, height = image.size
            img_info = {'file_name': filename,
                        'height': height,
                        'width': width,
                        'id': pathlib.Path(filename).stem}
            output_json_dict['images'].append(img_info)

            for ann in annotation:
                category_id = int(ann[0])
                x_min, y_min, x_max, y_max = list(map(int, ann[1]))

                box_w = x_max - x_min
                box_h = y_max - y_min
                output_json_dict['annotations'].append({'area': box_w * box_h,
                                                        'iscrowd': 0,
                                                        'bbox': [x_min, y_min, box_w, box_h],
                                                        'category_id': category_id,
                                                        'ignore': 0,
                                                        'image_id': pathlib.Path(filename).stem,
                                                        'id': box_id,
                                                        'segmentation': []})
                box_id += 1

        for label_id, label in enumerate(classes):
            category_info = {'supercategory': label, 'id': label_id + 1, 'name': label}
            output_json_dict['categories'].append(category_info)

        with open(f'{data_dir}/{phase}/{csv_name[:-4]}.json', 'w') as f:
            output_json = json.dumps(output_json_dict)
            f.write(output_json)


def image2coco():
    # 1 load image list info
    infos = []
    filenames = [x for x in os.listdir(f'{data_dir}/phase2/images/') if x.endswith('jpg')]
    for filename in tqdm.tqdm(filenames):
        try:
            image = cv2.imread(f'{data_dir}/phase2/images/{filename}')
            height, width = image.shape[:2]
        except cv2.error:
            with open(f'{data_dir}/phase2/images/{filename}', 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')
                image.save(f'{data_dir}/phase2/images/{filename}')
            image = cv2.imread(f'{data_dir}/phase2/images/{filename}')
            height, width = image.shape[:2]
        infos.append({'filename': f'{filename}',
                      'width': width,
                      'height': height})

    # 2 convert to coco format data
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for info in infos:
        file_name = info['filename']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = file_name[:-4]
        image_item['file_name'] = f'{data_dir}/phase2/images/{file_name}'
        image_item['height'] = int(info['height'])
        image_item['width'] = int(info['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

    # 3 dump
    mmcv.dump(coco, '../Dataset/ROBINv1.1/phase2/phase2.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv2coco', action='store_true')
    parser.add_argument('--image2coco', action='store_true')

    args = parser.parse_args()

    if args.csv2coco:
        csv2coco()
    if args.image2coco:
        image2coco()


if __name__ == '__main__':
    main()
