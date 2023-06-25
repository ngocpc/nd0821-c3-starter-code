#!/bin/bash

CONTAINER_NAME="mlops"
CONTAINER_USER="vscode"

touch ~/.gitattributes
echo "*.sh text eol=lf" >> ~/.gitattributes

if [ "$(docker ps -a -q --filter='name=${CONTAINER_NAME}')" ]
then
  echo "ERROR!"
  echo "There is a container running with the current version of "
  echo "the Docker image secarna_ml. Please remove the container"
  echo "before building a new version of the image!"
  exit 1
fi

echo "Building development container"
docker build \
  -t ${CONTAINER_NAME} \
  --build-arg USERNAME=${CONTAINER_USER} \
  --build-arg USER_UID=1000 \
  --build-arg USER_GID=1000 \
  --no-cache \
  .devcontainer/.
