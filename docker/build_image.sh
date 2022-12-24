#!/bin/bash

function error_exit {
  echo "$1"
  exit 1
}

# add ssh file to clone some files
cp -r ~/.ssh .ssh

MACHINE=cpu
BRANCH_NAME="master"
BUILD_TYPE="production"
BASE_IMAGE="ubuntu:18.04"
CUSTOM_TAG=false
CUDA_VERSION=""
PUSH_IMAGE=false

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-b, --branch_name=BRANCH_NAME specify a branch_name or tag suffix to use"
          echo "-g, --gpu specify to use gpu"
          echo "-bt, --buildtype specify to created image for codebuild. Possible values: production, dev, codebuild."
          echo "-cv, --cudaversion specify to cuda version to use"
          echo "-t, --tag specify tag name for docker image"
          echo "-s, --suffix specify tag name for docker image"
          echo "-p, --push push image or not"
          exit 0
          ;;
        -g|--gpu)
          MACHINE=gpu
          BASE_IMAGE="nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04"
          CUDA_VERSION="cu102"
          shift
          ;;
        -bt|--buildtype)
          BUILD_TYPE="$2"
          shift
          shift
          ;;
        -t|--tag)
          DOCKER_TAG="$2"
          CUSTOM_TAG=true
          shift
          shift
          ;;
        -s|--suffix)
          USE_SUFFIX=true
          SUFFIX="$2"
          shift
          shift
          ;;
        -cv|--cudaversion)
          CUDA_VERSION="$2"
          if [ $CUDA_VERSION == "cu111" ];
          then
            BASE_IMAGE="nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu113" ];
          then
            BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu102" ];
          then
            BASE_IMAGE="nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu101" ]
          then
            BASE_IMAGE="nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04"
          elif [ $CUDA_VERSION == "cu92" ];
          then
            BASE_IMAGE="nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04"
          else
            echo "CUDA version not supported"
            exit 1
          fi
          shift
          shift
          ;;
        -stage)
          CICD_STAGE="$2"
          shift
          shift
          ;;
        -b|--branch_name)
          BRANCH_NAME="$2"
          shift
          shift
          ;;
        -p)
          PUSH_IMAGE="$2"
          shift
          ;;
    esac
done

echo "BRANCH_NAME:${BRANCH_NAME}"

if [ "${BUILD_TYPE}" == "dev" ] && ! $CUSTOM_TAG ;
then
  DOCKER_TAG="dockerhub.datagrand.com/ysocr/server:dev_${CUDA_VERSION}"
fi

if [ "${BUILD_TYPE}" == "production" ] && $USE_SUFFIX ;
then
  DOCKER_TAG="dockerhub.datagrand.com/ysocr/server:release_${BRANCH_NAME}_${SUFFIX}"
fi

echo "DOCKER_TAG:${DOCKER_TAG}"
echo "MACHINE:${MACHINE}"
echo "CUDA_VERSION:${CUDA_VERSION}"
echo "BASE_IMAGE:${BASE_IMAGE}"

echo "start build image"
if [ $BUILD_TYPE == "production" ]
then
  DOCKER_BUILDKIT=1 docker build --file docker/$BRANCH_NAME.Dockerfile -t $DOCKER_TAG --build-arg BASE_IMAGE=$BASE_IMAGE  \
     --build-arg CUDA_VERSION=$CUDA_VERSION .
else
  DOCKER_BUILDKIT=1 docker build --file docker/dev.Dockerfile -t $DOCKER_TAG --build-arg BASE_IMAGE=$BASE_IMAGE  \
     --build-arg CUDA_VERSION=$CUDA_VERSION .
fi

image_build_status=$?
echo "image_build_status:${image_build_status}"
if [ "${image_build_status}" != "0" ];then
  error_exit "docker build failed"
fi

if ${PUSH_IMAGE}
then
  docker push ${DOCKER_TAG}
fi