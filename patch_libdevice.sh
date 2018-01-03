#!/bin/bash

if [ "$#" -ne 2 ] || ! [ -d "$1" ] || ! [ -d "$2" ]; then
  echo "Usage: $0 libdevice_dir llvm_dir" >&2
  exit 1
fi

LLVMAS=$2/bin/llvm-as
LLVMDIS=$2/bin/llvm-dis

if ! [ -x $LLVMAS ] || ! [ -x ${LLVMDIS} ]; then
   echo "either one of $LLVMAS or $LLVMDIS is not executable" >& 2
   exit 1
fi

LIBDEVICE_DIR=$1

if [ -d $LIBDEVICE_DIR ]; then

    FIRST="1"
    for libdev in `ls ${LIBDEVICE_DIR}/libdevice*.bc`; do
	echo $libdev

	$LLVMDIS < $libdev | sed -e 's/rsqrt\.approx\.ftz\.f64/rsqrt\.approx\.f64/g' | $LLVMAS > `basename $libdev`

    done
fi
