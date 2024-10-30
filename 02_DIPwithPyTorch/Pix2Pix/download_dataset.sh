FILE=$1
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
mkdir -p $TARGET_DIR

find "${TARGET_DIR}train" -type f -name "*.jpg" |sort -V > ./train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" |sort -V > ./val_list.txt
