#!/usr/bin/env bash
# DeepKnee REST microservice images
docker build -t miptmloulu/deepknee:gpu -f docker/Dockerfile.gpu .
docker build -t miptmloulu/deepknee:cpu -f docker/Dockerfile.cpu .
docker push miptmloulu/deepknee:cpu && docker push miptmloulu/deepknee:gpu

# Frontend and Backend broker