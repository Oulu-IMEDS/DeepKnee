"""
    This micro-service takes a DICOM (bilateral X-ray) or PNG image (single knee, localized)
    and makes KL grade predictions with GradCAM.

    (c) Aleksei Tiulpin, University of Oulu, 2019
"""
import argparse
import base64
import glob
import logging
import os

import cv2
from flask import Flask, request
from flask import jsonify, make_response
from gevent.pywsgi import WSGIServer

from ouludeepknee.inference.pipeline import KneeNetEnsemble


def numpy2base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('ascii')


app = Flask(__name__)


def call_pipeline(dicom_raw, landmarks=None):
    # Localization of ROIs and their conversion into 8-bit 140x140mm images
    res_bilateral = net.predict_draw_bilateral(dicom_raw, args.sizemm, args.pad,
                                               kneel_addr=os.environ['KNEEL_ADDR'], landmarks=landmarks)
    if res_bilateral is None:
        return make_response(jsonify({'msg': 'Could not localize the landmarks'}), 400)

    img_l, img_hm_l, preds_bar_l, pred_l, img_r, img_hm_r, preds_bar_r, pred_r = res_bilateral

    response = {'L': {'img': numpy2base64(img_l),
                      'hm': numpy2base64(img_hm_l),
                      'preds_bar': numpy2base64(preds_bar_l),
                      'kl': str(pred_l)},
                'R': {'img': numpy2base64(img_r),
                      'hm': numpy2base64(img_hm_r),
                      'preds_bar': numpy2base64(preds_bar_r),
                      'kl': str(pred_r)}}
    return response


@app.route('/deepknee/predict/bilateral', methods=['POST'])
def analyze_knee():
    logger = logging.getLogger(f'deepknee-backend:app')
    request_json = request.get_json(force=True)
    dicom_base64 = request_json['dicom']
    dicom_binary = base64.b64decode(dicom_base64)
    if 'landmarks' in request_json:
        landmarks = request_json['landmarks']
    else:
        landmarks = None
    logger.log(logging.INFO, f'Received DICOM')

    if os.environ['KNEEL_ADDR'] == '':
        return make_response(jsonify({'msg': 'KNEEL microservice address is not defined'}), 500)

    response = call_pipeline(dicom_binary, landmarks)

    return make_response(response, 200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_path', default='')
    parser.add_argument('--deploy_addr', default='0.0.0.0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--sizemm', type=int, default=140)
    parser.add_argument('--pad', type=int, default=300)
    parser.add_argument('--deploy', type=bool, default=False)
    parser.add_argument('--logs', type=str, default='/tmp/deepknee.log')
    args = parser.parse_args()
    logging.basicConfig(filename=args.logs, filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    logger = logging.getLogger(f'deepknee-backend:app')

    net = KneeNetEnsemble(glob.glob(os.path.join(args.snapshots_path, "*", '*.pth')),
                          mean_std_path=os.path.join(args.snapshots_path, 'mean_std.npy'),
                          device=args.device)

    if args.deploy:
        http_server = WSGIServer((args.deploy_addr, args.port), app, log=logger)
        logger.log(logging.INFO, f'Starting WSGI server')
        http_server.serve_forever()
    else:
        app.run(host=args.deploy_addr, port=args.port, debug=True)
