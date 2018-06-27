#!/bin/bash

if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 DIRECTORY" >&2
  exit 1
fi


LIBDEVICE_DIR=$1
QDPJIT_DIR=`pwd`
OUT_HEADER="$QDPJIT_DIR/include/qdp_libdevice.h"
OUT_LIB="$QDPJIT_DIR/lib/qdp_libdevice.cc"

if [ -f $OUT_HEADER ]; then
    echo "$OUT_HEADER already exists."
    exit 1
fi

if [ -f $OUT_LIB ]; then
    echo "$OUT_LIB already exists."
    exit 1
fi

if [ ! -f ${LIBDEVICE_DIR}/libdevice.bc ]; then
    echo "${LIBDEVICE_DIR}/libdevice.bc not found."
    exit 1
fi



if [ ! -d include ]; then
    echo "Please call from qdp-jit root dir!"
    exit 1
fi
if [ ! -d lib ]; then
    echo "Please call from qdp-jit root dir!"
    exit 1
fi

cat << 'EOF' > $OUT_HEADER
#ifndef qdp_libdevice
#define qdp_libdevice

namespace QDP {
namespace LIBDEVICE {

EOF


cat << 'EOF' > $OUT_LIB
namespace QDP {
namespace LIBDEVICE {

EOF


if [ -d $LIBDEVICE_DIR ]; then
    cd $LIBDEVICE_DIR
    echo "ok"

    for libdev in `ls libdevice.bc`; do
	echo $libdev
	head=`xxd -i $libdev | head -n 1| sed -e 's/\[\].*$/\[\];/'`
	tail=`xxd -i $libdev | tail -n 1| sed -e 's/_len.*$/_len;/'`
	name=`echo $head | sed -e 's/^.*char //'| sed -e 's/\[\]//' | sed -e 's/;//'`

	# libdevice.compute_20.10.bc    [name] / name_len

	echo "extern $head" >> $OUT_HEADER
	echo "extern $tail" >> $OUT_HEADER
	xxd -i $libdev >> $OUT_LIB
    done
    
fi

cat << 'EOF' >> $OUT_HEADER

} // namespace LIBDEVICE
} // namespace QDP
#endif
EOF

cat << 'EOF' >> $OUT_LIB

} // namespace LIBDEVICE
} // namespace QDP
EOF

cd $QDPJIT_DIR

#unsigned char libdevice_compute_20_10_bc[] = {
