#! /bin/bash

usage_exit() {
        echo "Usage: $0 [-b bucket_name] [-m mount_point]" 1>&2
        exit 1
}

while getopts bm:h OPT
do
    case $OPT in
        b)  BUCKET_NAME=$OPTARG
            ;;
        m)  MOUNT_POINT=$OPTARG
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done

mkdir $MOUNT_POINT
gcsfuse --implicit-dirs $BUCKET_NAME $MOUNT_POINT
