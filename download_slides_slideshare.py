import json
import os
import argparse
import time
from urllib import request
from tqdm import tqdm

def download_deck(args, sample, deck_name):
    ### Download slide images from slideshare ###    
    if not os.path.exists(f'images/{args.split}/{deck_name}'):
        os.makedirs(f'images/{args.split}/{deck_name}')
    
    for url in tqdm(sample['image_urls'], desc=f'Download 20 slides from {deck_name}'):
        save_name = f'images/{args.split}/{deck_name}/{os.path.basename(url)}'
        if not os.path.exists(save_name):
            request.urlretrieve(url, save_name) 
            time.sleep(args.sleep_time)       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, default='annotations/bbox')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--sleep_time', type=float, default=5)
    args = parser.parse_args()

    with open(f'{args.target_dir}/{args.split}.jsonl', 'r') as f:
        samples = f.readlines()

    for sample in tqdm(samples):
        sample = json.loads(sample)
        deck_name = sample['deck_name']
        download_deck(args, sample, deck_name)
