"""
Visualize a few bounding boxes from VOC dataset.
"""
import cv2
import numpy as np
import os
import json
import random

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

def draw_bboxes(image, objs, quiet=True):
    """
    objs: [{name, xmin, ymin, xmax, ymax, score}]
    """
    for obj in objs:
        label_str = obj['name']
        xmin = obj['xmin']
        xmax = obj['xmax']
        ymin = obj['ymin']
        ymax = obj['ymax']
        
        if 'score' in obj:
            score_str = str(round(obj['score'] * 100, 2)) + '%'
        else:
            score_str = ''
        label_str += score_str
        if not quiet: print(label_str)

        text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
        width, height = text_size[0][0], text_size[0][1]
        region = np.array([[xmin-3,       ymin],
                           [xmin-3,       ymin-height-26],
                           [xmin+width+13,ymin-height-26],
                           [xmin+width+13,ymin]], dtype='int32')

        cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0,255,0), thickness=5)
        cv2.fillPoly(img=image, pts=[region], color=(0,255,0))
        cv2.putText(img=image,
                    text=label_str,
                    org=(xmin + 13, ymin - 13), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1.5e-3 * image.shape[0], 
                    color=(0,0,0), 
                    thickness=1)

    return image


def visualize_random_instances(all_insts, num_samples, seed=None):

    if seed is not None:
        random.seed(seed)
    
    sampled_insts = random.sample(all_insts, num_samples)
    for sample in sampled_insts:
        filename = sample['filename']
        objects = sample['object']
        img = cv2.imread(filename)
        img = draw_bboxes(img, objects)
        cv2.imshow('Window', img)
        key = cv2.waitKey(0)
        if key == 27: #if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('label_file', help='Labels of all file in json format.')
    parser.add_argument('-n', '--num_imgs', nargs='?', type=int, default=3, help='Number of images to display.')
    parser.add_argument('--random_seed', nargs='?', type=int, default=1988, help='Random seeds to generate')

    args = parser.parse_args()
    with open(args.label_file, 'r') as handle:
        cache = json.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']

    visualize_random_instances(all_insts, args.num_imgs, args.random_seed)

