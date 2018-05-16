# Download whichever common dataset we're using

#!/bin/bash
# This script will download images from MSCOCO 2017

# Usage: ./build-dataset.sh <path/to/target/folder>
# Global variables

TARGET_PATH="$1"
MYNAME="$(readlink -f "$0")"
MYDIR="$(dirname "${MYNAME}")"

function ensure_cmd_or_install_package_apt() {
  local CMD=$1
  shift
  local PKG=$*
  hash $CMD 2>/dev/null || {
    log warn $CMD not available. Attempting to install $PKG
    (sudo apt-get update -yqq && sudo apt-get install -yqq ${PKG}) || die "Could not find $PKG"
  }
}

function is_sudoer() {
    CAN_RUN_SUDO=$(sudo -n uptime 2>&1|grep "load"|wc -l)
    if [ ${CAN_RUN_SUDO} -gt 0 ]
    then
        echo 1
    else
        echo 0
    fi
}

# Check if we are sudoer or not
if [ $(is_sudoer) -eq 0 ]; then
    die "You must be root or sudo to run this script"
fi

# Eventually installing dependencies
ensure_cmd_or_install_package_apt jq jq
ensure_cmd_or_install_package_apt wget wget

echo "Creating Target folder"
[ -d "${TARGET_PATH}" ] && echo "Target path already started, moving forward" || mkdir -p "${TARGET_PATH}"

# Creating MD5 sum file for large file download validation
cat > /tmp/md5sum.txt << EOF
68baf1a0733e26f5878a1450feeebc20  ${TARGET_PATH}/train2017.zip
a3d79f5ed8d289b7a7554ce06a5782b3  ${TARGET_PATH}/val2017.zip
a3d79f5ed8d289b7a7554ce06a5782b3  ${TARGET_PATH}/test2017.zip
EOF

cd ${TARGET_PATH}
DONE=-1

until [ ${DONE} -eq 0 ]; do

	echo "Downloading files"
	wget -qc images.cocodataset.org/zips/train2017.zip &
	wget -qc images.cocodataset.org/zips/val2017.zip &
  wget -qc images.cocodataset.org/zips/test2017.zip
	echo "Now waiting for all threads to end"
	wait
	echo "Done waiting for threads. Computing MD5SUM"

	DONE=$(md5sum -c /tmp/md5sum.txt | grep 'FAILED' | wc -l)
done

echo "All files downloaded"

# Build the raw JSON file
for file in train2017.zip val2017.zip test2017.zip
do
	echo "Uncompressing ${file}"
	unzip "${file}" && mv "${file}" /tmp/
done

# replace image with problem
# echo "Replacing MS COCO failed image by a fresh and working one"
# wget -qc https://msvocds.blob.core.windows.net/images/262993_z.jpg && \
# mv 262993_z.jpg "${TARGET_PATH}/train2014/COCO_train2014_000000167126.jpg"

echo "OK, all files downloaded and prep'd! You can safely move to training"
