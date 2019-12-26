# -*- coding: utf-8 -*-
import argparse
import base64
import logging
import queue
import time

import requests
from pydicom import dcmread
from pydicom.filebase import DicomBytesIO
from pydicom.uid import ImplicitVRLittleEndian
from pydicom.uid import generate_uid
from pynetdicom import AE, VerificationPresentationContexts

queue = queue.Queue()


def ingestion_loop():
    logger.log(logging.INFO, 'Creating application entity...')
    ae = AE(ae_title=b'DEEPKNEE')
    ae.requested_contexts = VerificationPresentationContexts
    ae.add_requested_context('1.2.840.10008.5.1.4.1.1.1.1', transfer_syntax=ImplicitVRLittleEndian)

    current = 0
    base_url = f'{args.orthanc_addr}:{args.orthanc_http_port}'
    response = requests.get(f'{base_url}/changes?since={current}&limit=10', auth=('deepknee', 'deepknee'))
    if response.status_code == 200:
        logger.log(logging.INFO, 'Connection to Orthanc via REST is healthy')

    # Orthanc addr must have http, but DICOM communicates via sockets
    assoc = ae.associate(args.orthanc_addr.split('http://')[1], args.orthanc_dicom_port)
    if assoc.is_established:
        logger.log(logging.INFO, 'Connection to Orthanc via DICOM is healthy')
        assoc.release()

    assoc = ae.associate(args.remote_pacs_addr, args.remote_pacs_port)
    if assoc.is_established:
        logger.log(logging.INFO, 'Connection to Remote PACS via DICOM is healthy')
        assoc.release()

    while True:
        response = requests.get(f'{base_url}/changes?since={current}&limit=10', auth=('deepknee', 'deepknee'))
        response = response.json()
        for change in response['Changes']:
            # We must also filter by the imaged body part in the future
            if change['ChangeType'] == 'NewInstance':
                logger.log(logging.INFO, 'Identified new received instance in Orthanc. '
                                         'Checking if it has been created by DeepKnee...')
                # We should not analyze the instances if they are produced by DeepKnee
                # Checking if it was verified by DeepKnee
                resp_verifier = requests.get(f'{base_url}/instances/{change["ID"]}/content/0040-a027',
                                             auth=('deepknee', 'deepknee'))
                resp_verifier.encoding = 'utf-8'
                resp_content = requests.get(f'{base_url}/instances/{change["ID"]}/content/0070-0080',
                                            auth=('deepknee', 'deepknee'))

                resp_content.encoding = 'utf-8'

                if resp_verifier.text.strip("\x00 ") == 'UniOulu-DeepKnee' and \
                        resp_content.text.strip("\x00 ") == 'DEEPKNEE-XRAY':
                    continue

                # Once we are sure that the instance is new, we need to go ahead with teh analysis
                response = requests.get(f'{base_url}/instances/{change["ID"]}/file', auth=('deepknee', 'deepknee'))

                logger.log(logging.INFO, 'Instance has been retrieved from Orthanc')
                dicom_raw_bytes = response.content
                dcm = dcmread(DicomBytesIO(dicom_raw_bytes))

                dicom_base64 = base64.b64encode(dicom_raw_bytes).decode('ascii')
                logger.log(logging.INFO, 'Sending API request to DeepKnee core')
                url = f'{args.deepknee_addr}:{args.deepknee_port}/deepknee/predict/bilateral'
                response_deepknee = requests.post(url, json={'dicom': dicom_base64})

                if response_deepknee.status_code != 200:
                    logger.log(logging.INFO, 'DeepKnee analysis has failed')
                else:
                    logger.log(logging.INFO, 'Getting rid of the instance in Orthanc')
                    if args.orthanc_addr.split('http://')[1] != args.remote_pacs_addr and \
                            args.orthanc_dicom_port != args.remote_pacs_port:
                        response = requests.delete(f'{base_url}/instances/{change["ID"]}',
                                                   auth=('deepknee', 'deepknee'))
                        if response.status_code == 200:
                            logger.log(logging.INFO, 'Instance has been removed from the Orthanc')
                    else:
                        logger.log(logging.INFO, 'Remote PACS is DeepKnee. The instance will not be removed.')

                    logger.log(logging.INFO, 'DeepKnee has successfully analyzed the image. Routing...')

                    # Report
                    deepknee_json = response_deepknee.json()
                    dcm.add_new([0x40, 0xa160], 'LO', 'KL_right: {}, KL_left: {}'.format(deepknee_json['R']['kl'],
                                                                                         deepknee_json['L']['kl']))
                    # Verifier
                    dcm.add_new([0x40, 0xa027], 'LO', 'UniOulu-DeepKnee')
                    # Content label
                    dcm.add_new([0x70, 0x80], 'CS', 'DEEPKNEE-XRAY')

                    dcm[0x08, 0x8].value = 'DERIVED'
                    # Instance_UUID
                    current_uuid = dcm[0x08, 0x18].value
                    dcm[0x08, 0x18].value = generate_uid(prefix='.'.join(current_uuid.split('.')[:-1])+'.')
                    # Series UUID
                    current_uuid = dcm[0x20, 0x0e].value
                    dcm[0x20, 0x0e].value = generate_uid(prefix='.'.join(current_uuid.split('.')[:-1])+'.')
                    logger.log(logging.INFO, 'Connecting to Orthanc over DICOM')
                    assoc = ae.associate(args.remote_pacs_addr, args.remote_pacs_port)
                    if assoc.is_established:
                        logger.log(logging.INFO, 'Association with Orthanc has been established. Routing..')
                        routing_status = assoc.send_c_store(dcm)
                        logger.log(logging.INFO, f'Routing finished. Status: {routing_status}')
                        assoc.release()

            else:
                # Here there should be a code to remove the change from the pacs
                # Now nothing is done here
                pass
            current += 1
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepknee_addr', default='http://127.0.0.1', help='DeepKnee address')
    parser.add_argument('--deepknee_port', default=5001, help='DeepKnee backend port')

    parser.add_argument('--orthanc_addr', default='http://127.0.0.1', help='The host address that runs Orthanc')
    parser.add_argument('--orthanc_http_port', type=int, default=6001, help='Orthanc REST API port')
    parser.add_argument('--orthanc_dicom_port', type=int, default=6000, help='Orthanc DICOM port')

    parser.add_argument('--remote_pacs_addr', default='http://127.0.0.1', help='Remote PACS IP addr')
    parser.add_argument('--remote_pacs_port', type=int, default=6000, help='Remote PACS port')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(f'dicom-router')

    ingestion_loop()

