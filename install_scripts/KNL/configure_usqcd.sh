#!/bin/bash

CXX=mpiicpc
CC=mpiicc

SCIDACPATH=/panfs/users/Xfwinte/git
INSTALLPATH=`pwd`

LLVMDIR=/panfs/users/Xfwinte/install/llvm-3.8-intel

QMPDIR=$SCIDACPATH/qmp
QDPDIR=$SCIDACPATH/qdp-jit.llvm-cpu-inner-loop-no11-qshift
CHROMADIR=$SCIDACPATH/chroma

ARCH=parscalar
PRECISION=single

QMP=no
QDP=no
CHROMA=yes


TYPE="jit-llvm-3.8-intel-$PRECISION"


INSTALL=$INSTALLPATH/install
BUILD=$INSTALLPATH/build


QDPSSEFLAG=""
CHROMASSEFLAG=""
CXXFLAGS="-O3 -qopenmp -std=c++0x -qopt-report"
CFLAGS="-O3 -std=c99"




if [ ! -d $INSTALL ]
then
    mkdir $INSTALL
fi

if [ ! -d $INSTALL/qdp++-$TYPE ]
then
    mkdir $INSTALL/qdp++-$TYPE
fi

if [ ! -d $INSTALL/qmp-$TYPE ]
then
    mkdir $INSTALL/qmp-$TYPE
fi

if [ ! -d $INSTALL/chroma-$TYPE ]
then
    mkdir $INSTALL/chroma-$TYPE
fi

if [ ! -d $BUILD ]
then
    mkdir $BUILD
fi

if [ ! -d $BUILD/qdp++-$TYPE ]
then
    mkdir $BUILD/qdp++-$TYPE
fi

if [ ! -d $BUILD/qmp-$TYPE ]
then
    mkdir $BUILD/qmp-$TYPE
fi

if [ ! -d $BUILD/chroma-$TYPE ]
then
    mkdir $BUILD/chroma-$TYPE
fi


if [ $QMP = "yes" ]
then
    cd $BUILD/qmp-$TYPE
    $QMPDIR/configure \
        --prefix=$INSTALL/qmp-$TYPE  \
        --with-qmp-comms-type=MPI \
        CC=${CC} \
        CFLAGS="$CFLAGS"
    make
    make install
fi

if [ $QDP = "yes" ]
then
    cd $BUILD/qdp++-$TYPE
    $QDPDIR/configure $QDPSSEFLAG \
        --disable-generics \
        --enable-openmp \
        --prefix=$INSTALL/qdp++-$TYPE  \
        --with-qmp=$INSTALL/qmp-$TYPE  \
        --with-llvm=${LLVMDIR} \
        --enable-parallel-arch=$ARCH \
	--enable-precision=$PRECISION \
        CXX=${CXX} \
        CXXFLAGS="$CXXFLAGS" \
        CFLAGS="$CFLAGS"
    make -j 4
    make install
fi


if [ $CHROMA = "yes" ]
then
    cd $BUILD/chroma-$TYPE
    $CHROMADIR/configure $CHROMASSEFLAG \
        --enable-jit-clover \
        --prefix=$INSTALL/chroma-$TYPE \
        --with-qdp=$INSTALL/qdp++-$TYPE \
        CXX=${CXX} \
        CXXFLAGS="${CXXFLAGS}"
fi

