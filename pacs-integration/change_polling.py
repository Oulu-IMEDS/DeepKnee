# -*- coding: utf-8 -*-
import queue
import argparse
import requests
import base64
import time


queue = queue.Queue()


def ingestion_loop():
    current = 0
    response = requests.get(f'{args.orthanc_addr}:{args.orthanc_port}/changes?since={current}&limit=10',
                            auth=('deepknee', 'deepknee'))
    if response.status_code == 200:
        print('Connection to orthanc is healthy.')

    while True:
        response = requests.get(f'{args.orthanc_addr}:{args.orthanc_port}/changes?since={current}&limit=10',
                                auth=('deepknee', 'deepknee'))
        response = response.json()
        for change in response['Changes']:
            if change['ChangeType'] == 'NewInstance':
                response = requests.get(f'{args.orthanc_addr}:{args.orthanc_port}/instances/{change["ID"]}/file',
                                        auth=('deepknee', 'deepknee'))
                dicom_base64 = base64.b64encode(response.content).decode('ascii')

                response = requests.post(f'{args.deepknee_addr}:{args.deepknee_port}/deepknee/predict/bilateral',
                                         json={'dicom': dicom_base64})
                if response.status_code != 200:
                    print('Request has failed!')
                else:
                    response_res = response.json()
                    print(response_res['R']['kl'], response_res['L']['kl'])
                response = requests.delete(f'{args.orthanc_addr}:{args.orthanc_port}/instances/{change["ID"]}',
                                           auth=('deepknee', 'deepknee'))
                if response.status_code == 200:
                    print('Instance has been removed from the orthanc router')
            else:
                pass

            current += 1
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepknee_addr', default='http://127.0.0.1', help='DeepKnee address')
    parser.add_argument('--deepknee_port', default=5001, help='DeepKnee backend port')
    parser.add_argument('--orthanc_addr', default='http://127.0.0.1', help='The host address that runs Orthanc')
    parser.add_argument('--orthanc_port', type=int, default=6001, help='Orthanc REST API port')
    parser.add_argument('--remote_pacs_addr', default='http://127.0.0.1', help='Remote PACS IP addr')
    parser.add_argument('--remote_pacs_port', type=int, default=8042, help='Remote PACS IP addr')
    args = parser.parse_args()
    ingestion_loop()

