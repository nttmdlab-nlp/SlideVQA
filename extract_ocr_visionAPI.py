import json 
import os
import cv2
import argparse
import glob
from google.cloud import vision
from tqdm import tqdm 

def extract_ocr(filename, output_filename):
    if os.path.exists(output_filename):
        return

    if '.jpg' not in filename:
        return

    try:
        client = vision.ImageAnnotatorClient()
        with io.open(file_name, 'rb') as image_file:
            content = filename.read()
            img = vision.Image(content=content)
            response = client.document_text_detection(image=img)

        with gzip.open(output_filename, 'wt') as fp:
            str_data = vision.AnnotateImageResponse.to_json(response)
            data = json.loads(str_data)
            del data['textAnnotations'], data["faceAnnotations"], data["landmarkAnnotations"], data["logoAnnotations"], data["labelAnnotations"], data["localizedObjectAnnotations"]
            print(json.dumps(data, ensure_ascii=False), file=fp)
    except:
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--save_dir', type=str, default='ocrs_visionAPI')
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    for filename in tqdm(glob.glob(f'{args.image_dir}/{args.split}/*/*')):
        save_dir = os.path.dirname(filename).replace(args.image_dir + '/', args.save_dir + '/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        output_filename = filename.replace(args.image_dir + '/', args.save_dir + '/').replace('.jpg', '.json')
        extract_ocr(filename, output_filename)

