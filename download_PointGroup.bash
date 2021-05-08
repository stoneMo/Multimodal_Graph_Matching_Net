#!/bin/bash

# download the pre-processed data
# https://drive.google.com/file/d/1deN0CrSyxnLHNp8gp9n6ZlsuQNzihVR8/view?usp=sharing

fileid="1deN0CrSyxnLHNp8gp9n6ZlsuQNzihVR8"
filename="PointGroupInst.7z"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

