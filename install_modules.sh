#! /bin/bash
# install gcsfuse
sudo sh -c "echo 'deb http://packages.cloud.google.com/apt gcsfuse-bionic main' > /etc/apt/sources.list.d/gcsfuse.list"
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install gcsfuse -y

# install python libs
pip install transformers datasets fugashi unidic-lite cloud-tpu-client

# set shared memory high
# sudo umount /dev/shm/ && sudo mount -t tmpfs -o rw,nosuid,nodev,noexec,relatime,size=50G shm /dev/shm
