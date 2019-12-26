#!/usr/bin/env bash

if [ ! -d "snapshots_release_kneel" ]; then
    wget http://mipt-ml.oulu.fi/models/KNEEL/snapshots_release.tar.xz
    tar -xvf snapshots_release.tar.xz
    rm snapshots_release.tar.xz
    mv snapshots_release snapshots_release_kneel
fi

if [ ! -d "snapshots_knee_grading" ]; then
    sh ./fetch_snapshots.sh
fi
mkdir -p logs

docker-compose -f ./docker/docker-compose-$1.yml down
docker-compose -f ./docker/docker-compose-$1.yml build
docker-compose -f ./docker/docker-compose-$1.yml up --remove-orphans