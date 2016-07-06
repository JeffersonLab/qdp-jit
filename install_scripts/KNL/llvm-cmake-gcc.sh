#!/bin/bash

SRC=$HOME/svn/llvm
CMAKE_BUILD_TYPE="Debug"
CMAKE_INSTALL_PREFIX="$HOME/toolchain/install/llvm-mod-shared"
LLVM_TARGETS_TO_BUILD="X86"

CXX="/dist/gcc-4.8.2/bin/g++"
CC="/dist/gcc-4.8.2/bin/gcc"


cmake -G "Unix Makefiles" \                                                                                                                                                                                 
-DCMAKE_CXX_COMPILER=$CXX \                                                                                                                                                                                 
-DCMAKE_C_COMPILER=$CC \                                                                                                                                                                                    
-DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \                                                                                                                                                                      
-DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX \                                                                                                                                                              
-DLLVM_TARGETS_TO_BUILD=$LLVM_TARGETS_TO_BUILD \                                                                                                                                                            
-DLLVM_ENABLE_ZLIB="OFF" \                                                                                                                                                                                  
-DLLVM_ENABLE_TERMINFO="OFF" \                                                                                                                                                                              
-DBUILD_SHARED_LIBS="ON" \                                                                                                                                                                                  
$SRC  


