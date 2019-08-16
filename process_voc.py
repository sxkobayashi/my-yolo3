"""
Extract bounding boxes from VOC dataset.
"""

import argparse
from kyolo3.voc import parse_voc_annotation

"""
VOC Annotation format
all_inst [img0, img1, ...]
seen_labels {
    obj[name]: count
}
img {
    object:
        [obj0, obj1, ...]
    filename: str
    width: int
    height: int

}
obj {
    name
    xmin
    xmax
    ymin
    ymax
}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_dir', help='Directory containing VOC annotations.')
    parser.add_argument('image_dir', help='Directory containing VOC images.')
    parser.add_argument('-o', '--output_file', nargs='?', default='./voc_labels.json', help='Output Json file path')

    args = parser.parse_args()
    all_insts, seen_labels = parse_voc_annotation(args.annot_dir, args.image_dir, args.output_file)
    print('{} images are processed.'.format(len(all_insts)))
    print('Labels:')
    for key, val in seen_labels.items():
        print("{} : {}".format(key, val))

