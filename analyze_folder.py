import glob
import argparse
import requests
import os
import base64


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepknee_host', default='http://127.0.0.1')
    parser.add_argument('--deepknee_port', type=int, default=5001)
    parser.add_argument('--img_dir', default='')
    args = parser.parse_args()

    url = f'{args.deepknee_host}:{args.deepknee_port}/deepknee/predict/bilateral'
    for img_path in glob.glob(os.path.join(args.img_dir, '*')):
        with open(img_path, 'rb') as f:
            data_base64 = base64.b64encode(f.read()).decode()
        response = requests.post(url, json={'dicom': data_base64})
        res = response.json()
