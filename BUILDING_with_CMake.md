# Basic Example for building with CMake

## ROCM builds

So far I have only attempted to build with ROCm and OpenMPI. 
Here the tricks were
  * Setup ROCm (e.g. via `module use rocm`) -- ensure `ROCM_PATH` is set
  * Setup OpenMPI (e.g. via some module load )
  * Set the OpenMPI wrappers to use HipCC
``` 
     export OMPI_CXX=hipcc
     export OMPI_CC=hipcc
```
 
  * Build QMP (looks like only parscalar arch is currently supported)

  * Configure using CMake as (e.g. for MI100 = gfx908)
```
	# Make a parallel directory to qdp-jit via
	mkdir build_qdpjit ; cd build_qdpjit
	cmake ../qdp-jit \
        -DQDP_PARALLEL_ARCH=parscalar \
        -DQDP_ENABLE_BACKEND=ROCM  \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_C_COMPILER=mpicc \
        -DGPU_TARGETS=gfx908 \
        -DQDP_ENABLE_ROCM_STATS=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DQMP_DIR=${HOME}/QUDA-Hipify/build_attempt/install/qmp/lib/cmake/QMP
```

  * `cmake --build . -j `  (or simply `make -j ` if you are using UNIX makefiles)
  * `cmake --install . --prefix <INSTALL PREFIX>`  (if you haven't set a `-DCMAKE_INSTALL_PREFIX=<location>` in which case just `make install` will work)

### Building the examples using an installed cmake
```
	# MAke a build dir parallel to the qdp-jit source dir
	mkdir build_exammples ; cd examples
	cmake ../qdp-jit/examples -DQMP_DIR=<qmp-install-prefix>/lib/cmake/QMP \
				  -DQDPXX_DIR=<qdp-jit-install-prefix>/lib/cmake/QDPXX \
				  -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc
        make -j 32
```

## CUDA Based builds

Here the main steps were:
 * obtain an LLVM build of LLVM-12 supporting NVPTX
 * ensure CUDA itself is set up
 * make a parallel build dir with `qdp-jit` and enter it:
```mkdir build_qdpjit_nv ; cd build_qdpjit_nv```
 * configure with
```
   cmake ../qdp-jit \
        -DQDP_PARALLEL_ARCH=parscalar \
        -DQDP_ENABLE_BACKEND=CUDA \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_C_COMPILER=mpicc \
        -DBUILD_SHARED_LIBS=ON \
        -DLLVM_DIR=<llvm-prefix>/lib/cmake/llvm \
        -DQMP_DIR=<qmp-prefix>/lib/cmake/QMP
```
  unlike for ROCM we do not need to provide the SM -- it will be queried at runtime.
  if CUDA is not found one can also add `-DCUDAToolkit_ROOT_DIR=<cuda prefix>`
  * build an install
```
    cmake --build . -j 32
    cmake --install . --prefix <qdp_install_prefix>
```
  or if GNU makefiles are used for the build (CMake default), add a 
  `-DCMAKE_INSTALL_PREFIX=<qdp_install_prefix>` option to the cmake configure and
``` make -j 32
    make install
``` 
### Building against installed QDP-JIT
To check that the `QDPXXConfig.cmake` is sane, one can build the examples against an installed QDP-JIT
  * make a parallel directory to the `qdp-jit` and go to it:
```
	mkdir build_examples_nv ; cd build_examples_nv
```
  * configure the build of the examples
```
cmake ../qdp-jit/examples -DQDPXX_DIR=<qdp-jit-CUDA-install-prefix>/lib/cmake/QDPXX \
                          -DQMP_DIR=<qmp-install-prefix>/lib/cmake/QMP \
			  -DLLVM_DIR=<llvm-install-prefix>/lib/cmake/llvm \
                          -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc

``` 
  * make with `cmake` as:
``` 
	cmake --build . -j 32
```
   or if CMake generated UNIX Makefiles (default) with e.g.: 
```
         make -j 32
```
