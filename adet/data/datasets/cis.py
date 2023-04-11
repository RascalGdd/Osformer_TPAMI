import os
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog

DATASET_ROOT = '/media/pjl307/data/experiment/datasets/CIS'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'traininstance3040')
TEST_PATH = os.path.join(DATASET_ROOT, 'test2026')
TRAIN_JSON = os.path.join(ANN_ROOT, 'train_instance.json')
TEST_JSON = os.path.join(ANN_ROOT, 'test2026.json')

NC4K_ROOT = '/media/pjl307/data/experiment/datasets/CIS'
NC4K_PATH = os.path.join(NC4K_ROOT, 'testNC4K-4121')
NC4K_JSON = os.path.join(ANN_ROOT, 'nc4k_test.json')

CLASS_NAMES = ["foreground"]

PREDEFINED_SPLITS_DATASET = {
    "my_data_train_coco_cod_style": (TRAIN_PATH, TRAIN_JSON),
    "my_data_test_coco_cod_style": (TEST_PATH, TEST_JSON),
    "my_data_test_coco_nc4k_style": (NC4K_PATH, NC4K_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")