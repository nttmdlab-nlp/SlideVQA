import json 
import os
import pytesseract
import cv2
import argparse
import glob
from pytesseract import Output
from tqdm import tqdm 

def extract_ocr(filename, output_filename):
    if os.path.exists(output_filename):
        return

    if '.jpg' not in filename:
        return

    try:
        img = cv2.imread(filename)
    except:
        print('Image Loading Error', filename)
        return

    d = pytesseract.image_to_data(img, output_type=Output.DICT) 
    with open(output_filename, 'w') as f:
        json.dump(d, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--save_dir', type=str, default='ocrs_tesseract')
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    for filename in tqdm(glob.glob(f'{args.image_dir}/{args.split}/*/*')):
        save_dir = os.path.dirname(filename).replace(args.image_dir + '/', args.save_dir + '/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        output_filename = filename.replace(args.image_dir + '/', args.save_dir + '/').replace('.jpg', '.json')
        extract_ocr(filename, output_filename)

