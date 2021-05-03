import copy
import json
import math
from pathlib import Path
import numpy as np
from pycocotools import mask as maskUtils

from custom_methods import get_parser


class CocoFilter:
    """ Filters the COCO dataset. Based on https://github.com/immersive-limit/coco-manager.
    The following command will filter the input instances json to not include images and annotations for the categories
    person, dog, or cat: python filter.py --input_json path/to/json output_json /path/to/json --categories person dog cat

    Note: This isn't looking for images with all categories in one. It includes images that have at least one of the
    specified categories.
    """

    def __init__(self, args):
        self.input_json_path = Path(args.input)
        self.output_json_path = Path(args.output)
        self.filter_categories = args.categories
        self.combine_categories = args.combine

        self.area = args.area
        # area is calulated from the segmenation, not the bounding box, so is the pixel area
        # APs is 32x32, so this should be close to merging the small annotations
        self.area_threshold = args.merge

        # max distance that can be merged, goal is to merge small close ones
        # we assume squares/rectangles with bbox, so look for things within a little more than the square root
        # if bboxes overlap, this always merges, if not, small margin
        # TODO: a better approximation might exist
        if self.area_threshold is not None:
            self.max_distance = 1.15 * math.sqrt(self.area_threshold)

        # Verify input path exists
        if not self.input_json_path.exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()

        # Verify output path does not already exist
        if self.output_json_path.exists():
            if args.y:
                print("Overwriting old json.")
            else:
                should_continue = input('Output path already exists. Overwrite? (y/n) ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    print('Quitting early.')
                    quit()

        # Load the json
        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.coco = json.load(json_file)
            self.total_segmentations = len(self.coco["annotations"])

            # merge before processing, so we can merge and then filter out
            if self.area_threshold is not None:
                self.coco['annotations'] = self.merge()
                self.total_segmentations_after_merge = len(self.coco['annotations'])

    def _process_info(self):
        self.info = self.coco['info']

    def _process_licenses(self):
        self.licenses = self.coco['licenses']

    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        self.category_set = set()

        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']

            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
                self.category_set.add(category['name'])
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')

            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {cat_id}  # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

    def _process_images(self):
        self.images = dict()
        self.total_images = 0
        for image in self.coco['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
                self.total_images += 1
            # this always gives output and is not informative, uncomment if required
            # else:
            #     print(f'ERROR: Skipping duplicate image id: {image}')

    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def _filter_categories(self):
        """ Find category ids matching args
            Create mapping from original category id to new category id
            Create new collection of categories
        """
        if self.filter_categories is not None:
            missing_categories = set(self.filter_categories) - self.category_set

            # filter out the ones that are given as a filter
            self.filter_categories = set(self.category_set) - set(self.filter_categories)
            if len(missing_categories) > 0:
                print(f'Did not find categories: {missing_categories}')
                should_continue = input('Continue? (y/n) ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    print('Quitting early.')
                    quit()
        else:
            # if no categories are given dont filter on categories
            self.filter_categories = self.category_set

        # original file had new_key, this keeps the intended category ID (yay!)
        self.new_category_map = dict()
        # store to be combined ID's
        self.combine_ids = []
        for key, item in self.categories.items():
            if item['name'] in self.filter_categories:
                # if combined, we dont want those annotations in the final category list
                if self.combine_categories is not None and item['name'] in self.combine_categories[1:]:
                    continue
                else:
                    self.new_category_map[key] = key
            if self.combine_categories is not None and item['name'] in self.combine_categories:
                self.combine_ids.append(key)

        self.new_categories = []
        for original_cat_id, new_id in self.new_category_map.items():
            new_category = dict(self.categories[original_cat_id])
            new_category['id'] = new_id
            self.new_categories.append(new_category)

    def _filter_annotations(self):
        """ Create new collection of annotations matching category ids
            Keep track of image ids matching annotations
        """
        self.new_segmentations = []
        self.new_image_ids = set()
        for image_id, segmentation_list in self.segmentations.items():
            for segmentation in segmentation_list:
                # filter out smaller than certain area bbox, if no area is given don't filter
                if self.area is not None and segmentation['area'] < self.area:
                    continue
                original_seg_cat = segmentation['category_id']

                orig_cat = self.categories.get(original_seg_cat).get('name')
                # combine categories to the first entry of the combine argument
                if self.combine_categories is not None and orig_cat in self.combine_categories:
                    original_seg_cat = self.combine_ids[0]

                if original_seg_cat in self.new_category_map.keys():
                    new_segmentation = dict(segmentation)
                    new_segmentation['category_id'] = self.new_category_map[original_seg_cat]
                    self.new_segmentations.append(new_segmentation)
                    self.new_image_ids.add(image_id)

    def _filter_images(self):
        """ Create new collection of images
        """
        self.new_images = []
        self.total_new_images = 0
        for image_id in self.new_image_ids:
            self.total_new_images += 1
            self.new_images.append(self.images[image_id])

    def distance_between_boxes(self, bbox1, bbox2):
        # bbox: [x,y,width,height]
        # calculate middle of the box
        middle1 = [bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2]
        middle2 = [bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2]

        dx = abs(middle1[0] - middle2[0])
        dy = abs(middle1[1] - middle2[1])

        return math.sqrt(dx ** 2 + dy ** 2)

    def merge(self):
        self.new_annotationlist = []
        # loop over all images and collect annotations for that image
        for image in self.coco['images']:
            annotations = copy.deepcopy({i["id"]: i for i in self.coco['annotations'] if (image["id"] == i["image_id"])})
            for annotation in annotations.values():
                annotation['merged'] = [annotation['id']]

            todo_list = [idx for idx, i in annotations.items() if i["area"] < self.area_threshold]
            # now loop over all annotations for that image and merge if required
            for idx in todo_list:
                annotation = annotations[idx]
                if annotation['area'] > self.area_threshold:
                    continue
                dist_annotations = [
                    (self.distance_between_boxes(annotation["bbox"], i["bbox"]), i)
                    for i in annotations.values()
                    if i["category_id"] == annotation["category_id"]
                       and i['id'] not in annotation['merged']
                ]
                if len(dist_annotations) == 0:
                    continue
                distance, closest_annotation = sorted(dist_annotations, key=lambda x: x[0])[0]
                if distance < self.max_distance:
                    closest_annotation["segmentation"] = maskUtils.merge(
                        [annotation["segmentation"], closest_annotation["segmentation"]])
                    closest_annotation["area"] = int(maskUtils.area(closest_annotation["segmentation"]))
                    closest_annotation["bbox"] = maskUtils.toBbox(closest_annotation["segmentation"])
                    for merge_id in annotation['merged']:
                        closest_annotation['merged'].append(merge_id)

                    annotations.pop(annotation['id'], None)

            self.new_annotationlist += annotations.values()
        return self.new_annotationlist

    def main(self):
        # Process the json
        print('Processing input json...')
        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()

        # Filter to specific categories
        print('Filtering...')
        self._filter_categories()
        self._filter_annotations()
        self._filter_images()
        total_new_segmentations = len(self.new_segmentations)

        for annotation in self.new_segmentations:
            annotation["segmentation"]["counts"] = annotation["segmentation"]["counts"].decode('utf-8') if isinstance(
                annotation["segmentation"]["counts"], bytes) else annotation["segmentation"]["counts"]
            annotation["bbox"] = annotation["bbox"].tolist() if isinstance(annotation["bbox"], np.ndarray) \
                else annotation["bbox"]

        # Build new JSON
        new_master_json = {
            'info': self.info,
            'licenses': self.licenses,
            'images': self.new_images,
            'annotations': self.new_segmentations,
            'categories': self.new_categories
        }

        # Write the JSON to a file
        print('Saving new json file...')
        with open(self.output_json_path, 'w+') as output_file:
            json.dump(new_master_json, output_file, indent=2)

        print('Filtered json saved.')
        print("Total number of images before filtering: " + str(self.total_images))
        print("Total number of images remaining in this set: " + str(self.total_new_images))
        print("Total annotations before filtering: " + str(self.total_segmentations))
        if self.area_threshold is not None:
            print("Total annotations after merging: " + str(self.total_segmentations_after_merge))
        print("Total annotations after filtering: " + str(total_new_segmentations))
        print("Fraction of total annotations left: " + str(total_new_segmentations/self.total_segmentations))


if __name__ == "__main__":
    args = get_parser().parse_args()

    cf = CocoFilter(args)
    cf.main()
