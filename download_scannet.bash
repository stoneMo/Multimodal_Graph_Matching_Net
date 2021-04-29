#!/bin/bash

# download the scannet_data
# https://drive.google.com/file/d/1IXL5dkMuti9FPaMsMkCVSX16PB99sL2T/view?usp=sharing

fileid="1IXL5dkMuti9FPaMsMkCVSX16PB99sL2T"
filename="scannet_data.tar.gz"
curl -k -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -k -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
