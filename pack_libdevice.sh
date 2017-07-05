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

extern std::map<int, unsigned char* > map_sm_lib;
extern std::map<int, unsigned int > map_sm_len;

EOF


cat << 'EOF' > $OUT_LIB
#include <map>

namespace QDP {
namespace LIBDEVICE {

EOF


if [ -d $LIBDEVICE_DIR ]; then
    cd $LIBDEVICE_DIR
    echo "ok"

    MAP_LIB="std::map<int, unsigned char* > map_sm_lib = {"$'\n'
    MAP_LEN="std::map<int, unsigned int > map_sm_len = {"$'\n'


    FIRST="1"
    for libdev in `ls libdevice*.bc`; do
	if [ "$FIRST" -eq "0" ]; then
	    MAP_LIB=$MAP_LIB$","
	    MAP_LIB=$MAP_LIB$'\n'
	    MAP_LEN=$MAP_LEN$","
	    MAP_LEN=$MAP_LEN$'\n'
	fi
	FIRST="0"
	
	echo $libdev
	head=`xxd -i $libdev | head -n 1| sed -e 's/\[\].*$/\[\];/'`
	tail=`xxd -i $libdev | tail -n 1| sed -e 's/_len.*$/_len;/'`
	name=`echo $head | sed -e 's/^.*char //'| sed -e 's/\[\]//' | sed -e 's/;//'`

	# libdevice.compute_20.10.bc    [name]

	sm=`echo $name | sed -e 's/^.*pute_//'| sed -e 's/_.*$//'`

	MAP_LIB=$MAP_LIB$"{$sm , $name}"

	MAP_LEN=$MAP_LEN$"{$sm , $name"
	MAP_LEN=$MAP_LEN$"_len}"
	
	echo "$sm $name"
	echo "extern $head" >> $OUT_HEADER
	echo "extern $tail" >> $OUT_HEADER
	xxd -i $libdev >> $OUT_LIB
    done
    
    MAP_LIB=$MAP_LIB$"};"$'\n'
    MAP_LEN=$MAP_LEN$"};"$'\n'

    echo $MAP_LIB >> $OUT_LIB
    echo $MAP_LEN >> $OUT_LIB
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
