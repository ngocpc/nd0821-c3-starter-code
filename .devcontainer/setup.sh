#!/bin/bash
#
# environment setup (bash), to be added on top of production environment:
# this setup script is used to setup custom dependencies not covered by Conda
# all files that are going to be used need to be either downloaded or copied
# into the Container (Dockerfile) during Docker image building

set -e

echo "Running setup script ..."
