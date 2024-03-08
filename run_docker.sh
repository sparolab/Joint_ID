#!/bin/bash


if [ "$#" -eq "0" ]
    then 
        echo ""
        echo "----------- help -----------"
        echo ""
        echo ""
        echo "-- arg0: $0               --- shell script file name"
        echo "-- arg1: $1               --- run/build/up (docker-compose command)"
        echo "-- arg2: $2               --- service name for docker build"
        echo "-- arg2: $3               --- dataset path"
        echo ""
        echo "----------------------------"
        exit
fi 

to_docker_path() {
    local RELATIVE_PATH=$(echo "`dirname docker/docker-compose.yml`")
    echo $(realpath $RELATIVE_PATH)
}

to_workspace_path() {
    local RELATIVE_PATH=$(echo "`dirname $DOCKER_PATH`")
    
    echo $(realpath $RELATIVE_PATH)
}


DOCKER_PATH=$(to_docker_path)                   # docker path define
export WORKSPACE_PATH=$(to_workspace_path)      # workspace path define
export WORKSPACE_NAME=$(echo "`basename $WORKSPACE_PATH`")

COMMAND=$1                                      # docker-compose COMMAND
SERVICE_ARG=$2                                    # service name
SERVICE=build_docker_images_${SERVICE_ARG}            # service name
export DOCKER_ARG=${WORKSPACE_NAME}

export DATASET_PATH=$3                            # dataset path define

##### service name2 #####
export DOCKER_IMAGE=ygm7422/official_joint_id:latest                                                               

xhost +local:docker


if [ "${COMMAND}" = up ]
    then
        docker-compose --file ${DOCKER_PATH}/docker-compose.yml ${COMMAND} -d ${SERVICE}
fi

if [ "${COMMAND}" = down ]
    then
        docker-compose --file ${DOCKER_PATH}/docker-compose.yml ${COMMAND}
fi

if [ "${COMMAND}" = run ]
    then
        docker-compose --file ${DOCKER_PATH}/docker-compose.yml ${COMMAND} ${SERVICE}
fi

if [ "${COMMAND}" = rm ]
    then
        docker-compose --file ${DOCKER_PATH}/docker-compose.yml ${COMMAND} ${SERVICE}
fi

