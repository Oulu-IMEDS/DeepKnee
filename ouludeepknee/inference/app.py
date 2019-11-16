"""
    This micro-service takes a dicom image in and returns JSON with localized landmark coordinates.
    (c) Aleksei Tiulpin, University of Oulu, 2019
"""
import argparse
from flask import jsonify
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from pydicom import dcmread
from pydicom.filebase import DicomBytesIO
import logging


app = Flask(__name__)


# curl -F dicom=@01 -X POST http://127.0.0.1:5000/predict/bilateral
@app.route('/predict/bilateral', methods=['POST'])
def analyze_knee():
    raise NotImplementedError


@app.route('/predict/single', methods=['POST'])
def analyze_knee():
    """
    Runs prediction for a single cropped right knee X-ray.

    """
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots', default='')
    parser.add_argument('--pad', type=int, default=300)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--mean_std_path', default='')
    parser.add_argument('--deploy', type=bool, default=False)
    args = parser.parse_args()

    loggers = {}

    for logger_level in ['app', 'pipeline', 'single-knee', 'bilateral-knee']:
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(f'kneel-backend:{logger_level}')
        logger.setLevel(logging.DEBUG)

        loggers[f'kneel-backend:{logger_level}'] = logger

    # TODO: Create an inference model

    if args.deploy:
        http_server = WSGIServer(('', 5000), app, log=logger)
        loggers['kneel-backend:app'].log(logging.INFO, 'Production server is running')
        http_server.serve_forever()
    else:
        loggers['kneel-backend:app'].log(logging.INFO, 'Debug server is running')
        app.run(host='', port=5000, debug=True)