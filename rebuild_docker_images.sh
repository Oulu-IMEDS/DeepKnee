#!/usr/bin/env bash
# This script must be built on a linux machine

# DeepKnee REST microservice images
docker build -t miptmloulu/deepknee:gpu -f docker/Dockerfile.gpu .
docker build -t miptmloulu/deepknee:cpu -f docker/Dockerfile.cpu .
docker build --build-arg REACT_APP_BROKER_PORT=5002 -t miptmloulu/deepknee:ui -f docker/UIDockerfile deepknee-frontend
docker build -t miptmloulu/deepknee:broker -f docker/BrokerDockerfile deepknee-backend-broker

# Frontend and Backend
docker push miptmloulu/deepknee:cpu && docker push miptmloulu/deepknee:gpu
docker push miptmloulu/deepknee:broker && docker push miptmloulu/deepknee:ui

