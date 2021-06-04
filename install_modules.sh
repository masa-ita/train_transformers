#! /bin/bash
# install gcsfuse
sudo echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install gcsfuse -y

# install python libs
pip install transformers datasets fugashi unidic-lite cloud-tpu-client
