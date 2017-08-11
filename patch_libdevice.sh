#!/bin/bash

if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 DIRECTORY" >&2
  exit 1
fi

LLVMAS=/home/fwinter/toolchain/install/llvm-4.0.1/bin/llvm-as
LLVMDIS=/home/fwinter/toolchain/install/llvm-4.0.1/bin/llvm-dis

LIBDEVICE_DIR=$1

if [ -d $LIBDEVICE_DIR ]; then

    FIRST="1"
    for libdev in `ls ${LIBDEVICE_DIR}/libdevice*.bc`; do
	echo $libdev

	$LLVMDIS < $libdev | sed -e 's/rsqrt\.approx\.ftz\.f64/rsqrt\.approx\.f64/g' | $LLVMAS > `basename $libdev`

    done
fi
