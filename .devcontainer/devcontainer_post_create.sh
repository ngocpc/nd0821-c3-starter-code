#!/bin/bash
#
# post create setup script executed from devcontainer.json definition

set -e

# create copy of .gitconfig for user permissions matching docker user
if [ -f /tmp/.gitconfig ]; then
  cp -f /tmp/.gitconfig ~/.gitconfig
else
  touch ~/.gitconfig
fi

# create copy of .ssh folder for user permissions matching docker user
if [ ! -d ~/.ssh ]; then
  mkdir -p ~/.ssh
fi
if [ "$(ls -A /tmp/.ssh)" ]; then
  install -C -m 600 -o "$(whoami)" -g "$(whoami)" /tmp/.ssh/* ~/.ssh
fi

# obtain the public ssh host key for bitbucket.org for remote git actions
ssh-keyscan -H www.bitbucket.org >> ~/.ssh/known_hosts

# start ssh-agent and add SSH-keys
echo "eval \$(ssh-agent) &>/dev/null" >> ~/.bashrc
echo "ssh-add ~/.ssh/* &>/dev/null" >> ~/.bashrc

# set git directory as safe as devcontainer user differs from system user
git config --global --add safe.directory "$(pwd)"

# install git hook scripts for pre-commit if contained in dev environment
if [ -d .git ] && (grep -q "pre-commit" .devcontainer/environment.yaml); then
  pre-commit install
fi

exit
