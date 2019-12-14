import glob
import argparse
import requests
import os
import base64
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

def base64tonumpy(buffer):
    binary = base64.b64decode(buffer)

    img = cv2.imdecode(np.fromstring(binary, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepknee_host', default='http://127.0.0.1')
    parser.add_argument('--deepknee_port', type=int, default=5001)
    parser.add_argument('--img_dir', default='')
    parser.add_argument('--save_results', default='/tmp/deepknee')
    args = parser.parse_args()

    url = f'{args.deepknee_host}:{args.deepknee_port}/deepknee/predict/bilateral'
    os.makedirs(args.save_results, exist_ok=True)
    output_csv = []
    flist = glob.glob(os.path.join(args.img_dir, '*'))
    for img_path in tqdm(flist, total=len(flist)):
        # Encoding the DICOM as base64 and sending the request to the server
        with open(img_path, 'rb') as f:
            data_base64 = base64.b64encode(f.read()).decode('ascii')
        response = requests.post(url, json={'dicom': data_base64})
        res = response.json()
        # Parsing the response
        result = {}
        for knee in 'LR':
            # You can also access the localized image, heatmaps and the probability maps
            result[knee] = {'img': base64tonumpy(res[knee]['img']),
                            'hm': base64tonumpy(res[knee]['hm']),
                            'preds_bar': base64tonumpy(res[knee]['preds_bar']),
                            'kl': res[knee]['kl']}
            output_csv.append({'File': img_path, 'Side': knee, 'KL': result[knee]['kl']})

    df = pd.DataFrame(data=output_csv)
    df.to_csv(os.path.join(args.save_results, 'deepknee.csv'), index=None)
