import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from custom_methods import get_parser
from data.filter_annotations import CocoFilter
from data.prepare_thing_sem_from_instance import create_coco_semantic_from_instance


def preprocess(args=None):
    if args is None:
        args = get_parser().parse_args()
    orig_input = args.input
    orig_output = args.output

    # run it twice, once for train and once for val (e.g. -i exc -o filtered_exc -a 1000)
    for s in ["train", "val"]:
        args.input = orig_input.replace("train", "{}".format(s))
        args.output = orig_output.replace("train", "{}".format(s))
        cf = CocoFilter(args)
        cf.main()

    # jank way to set correct paths
    val_json = str(cf.output_json_path)
    train_json = val_json.replace("val", "train")
    image_folder = val_json.replace("val", "images").replace(".json", "")

    # register dataset to get the correct thing_to_contagious_id
    register_coco_instances("car_damage_train", {}, train_json, image_folder)
    register_coco_instances("car_damage_val", {}, val_json, image_folder)
    # apply mapping, otherwise the correct metadata is not set yet
    DatasetCatalog.get("car_damage_train")
    thing_id_to_contiguous_id = MetadataCatalog.get("car_damage_train").as_dict()["thing_dataset_id_to_contiguous_id"]

    for s in [train_json, val_json]:
        create_coco_semantic_from_instance(
            os.path.join("{}".format(s)),
            os.path.join(image_folder, "thing_train"),
            thing_id_to_contiguous_id
        )


if __name__ == "__main__":
    preprocess()
