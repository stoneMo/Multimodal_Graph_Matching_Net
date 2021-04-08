#!/bin/bash

# download the tgnn pretrain
# https://drive.google.com/file/d/1chiXH5C3ex0ZS9mrwBJ6HyB-8VnDLL2U/view?usp=sharing

fileid="1chiXH5C3ex0ZS9mrwBJ6HyB-8VnDLL2U"
filename="TGNN_Pretrained.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

