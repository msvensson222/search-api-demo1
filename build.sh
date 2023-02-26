#!/bin/bash

container_name="demo1-custom-search-api-hybrid"
docker build -t $container_name .
docker tag $container_name "eu.gcr.io/strange-song-365307/$container_name"
docker push "eu.gcr.io/strange-song-365307/$container_name"