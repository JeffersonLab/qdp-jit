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

