import csv
import json
from collections import defaultdict

from PIL import Image

data_dir = '../Dataset/ROBINv1.1'

start_id = {'train.csv': 10_000,
            'iid.csv': 20_000,
            'context_bias.csv': 30_000,
            'occlusion_bias.csv': 40_000,
            'pose_bias.csv': 50_000,
            'shape_bias.csv': 60_000,
            'texture_bias.csv': 70_000,
            'weather_bias.csv': 80_000}


def csv2coco():
    classes = ('aeroplane',
               'bicycle',
               'boat',
               'bus',
               'car',
               'chair',
               'diningtable',
               'motorbike',
               'sofa',
               'train')
    phases = ['iid_test',
              'nuisances/context',
              'nuisances/occlusion',
              'nuisances/pose',
              'nuisances/shape',
              'nuisances/texture',
              'nuisances/weather']
    csv_names = ['iid.csv',
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
        img_id = start_id[csv_name]
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
                        'id': img_id}
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
                                                        'image_id': img_id,
                                                        'id': box_id,
                                                        'segmentation': []})
                box_id += 1
            img_id += 1

        for label_id, label in enumerate(classes):
            category_info = {'supercategory': label, 'id': label_id + 1, 'name': label}
            output_json_dict['categories'].append(category_info)

        with open(f'{data_dir}/{phase}/{csv_name[:-4]}.json', 'w') as f:
            output_json = json.dumps(output_json_dict)
            f.write(output_json)


def main():
    csv2coco()


if __name__ == '__main__':
    main()
