#!/bin/bash                                                                                                                                                                                     

SRC="/panfs/users/Xfwinte/llvm-3.8"

CMAKE="/opt/crtdc/cmake/3.0.2/bin/cmake"

PYTHON="/opt/crtdc/python/2.7.6/python"

CMAKE_BUILD_TYPE="Debug"
CMAKE_INSTALL_PREFIX="/panfs/users/Xfwinte/install/llvm-3.8-intel"
LLVM_TARGETS_TO_BUILD="X86"

CXX="/opt/intel/compiler/2016u2/compilers_and_libraries_2016.2.181/linux/bin/intel64/icpc"
CC="/opt/intel/compiler/2016u2/compilers_and_libraries_2016.2.181/linux/bin/intel64/icc"

$CMAKE -G "Unix Makefiles" \
-DCMAKE_CXX_FLAGS="-cxxlib=/opt/crtdc/gcc/4.8.5-4/ -std=c++11" \
-DCMAKE_C_FLAGS="-cxxlib=/opt/crtdc/gcc/4.8.5-4/" \
-DPYTHON_EXECUTABLE=$PYTHON \
-DCMAKE_CXX_COMPILER=$CXX \
-DLLVM_ENABLE_TERMINFO="OFF" \
-DCMAKE_C_COMPILER=$CC \
-DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
-DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX \
-DLLVM_TARGETS_TO_BUILD=$LLVM_TARGETS_TO_BUILD \
$SRC
