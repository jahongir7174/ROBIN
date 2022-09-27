[ROBIN](https://github.com/eccv22-ood-workshop/ROBIN-dataset) dataset converter from csv to coco format

### Install

* `pip install tqdm`
* `pip install mmcv`
* `pip install Pillow`

### Steps

* Download the dataset from original [repo](https://github.com/eccv22-ood-workshop/ROBIN-dataset)
* Unzip the file called `ROBINv1.1.zip` into `../Dataset/` folder
* `python main.py --csv2coco` for converting annotations from csv to coco format
* `python main.py --image2coco` for converting annotations from list of images to coco format

### Reference

* https://github.com/eccv22-ood-workshop/ROBIN-dataset