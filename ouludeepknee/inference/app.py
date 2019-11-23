"""
    This micro-service takes a DICOM (bilateral X-ray) or PNG image (single knee, localized)
    and makes KL grade predictions with GradCAM.

    (c) Aleksei Tiulpin, University of Oulu, 2019
"""
import argparse
import base64
import glob
import os

import cv2
from flask import Flask, request
from flask import jsonify, make_response
from gevent.pywsgi import WSGIServer
import socketio
import eventlet

import logging
from ouludeepknee.inference.pipeline import KneeNetEnsemble

app = Flask(__name__)
# Wrap Flask application with socketio's middleware
sio = socketio.Server(ping_timeout=120, ping_interval=120)


def numpy2base64(img):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return 'data:image/png;base64,' + base64.b64encode(buffer).decode('ascii')


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
                      'kl': str(pred_r)},
                'msg': 'Finished!'}
    return response

# curl -F dicom=@01 -X POST http://127.0.0.1:5001/deepknee/predict/bilateral
@app.route('/deepknee/predict/bilateral', methods=['POST'])
def analyze_knee():
    logger = logging.getLogger(f'deepknee-backend:app')
    dicom_raw = request.files['dicom'].read()
    logger.log(logging.INFO, f'Received DICOM')

    if os.environ['KNEEL_ADDR'] == '':
        return make_response(jsonify({'msg': 'KNEEL microservice is not defined'}), 500)

    response = call_pipeline(dicom_raw)

    return make_response(response, 200)


@sio.on('dicom_submission', namespace='/deepknee/sockets')
def on_dicom_submission(sid, data):
    sio.emit('dicom_received', dict(), room=sid, namespace='/deepknee/sockets')
    logger.info(f'Sent a message back to {sid}')
    sio.sleep(0)

    tmp = data['file_blob'].split(',', 1)[1]
    response = call_pipeline(base64.b64decode(tmp))
    # Send out the results
    sio.emit('dicom_processed', response, room=sid, namespace='/deepknee/sockets')


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

    app = socketio.WSGIApp(sio, app, socketio_path='/deepknee/sockets/socket.io')

    if args.deploy:
        # Deploy as an eventlet WSGI server
        eventlet.wsgi.server(eventlet.listen((args.deploy_addr, args.port)), app, log=logger)
        # http_server = WSGIServer((args.deploy_addr, 5001), app, log=logger)
        # http_server.serve_forever()
    else:
        app.run(host=args.deploy_addr, port=args.port, debug=True)
