#include "qdp.h"

namespace QDP {

  kernel_geom_t getGeom(int numSites , int threadsPerBlock)
  {
    kernel_geom_t geom_host;

    geom_host.threads_per_block = threadsPerBlock;
    geom_host.Nblock_x = min( DeviceParams::Instance().getMaxGridX() , (size_t)std::ceil( (double)numSites / (double)geom_host.threads_per_block ) );
    geom_host.Nblock_y = (int)std::ceil(  (double)numSites / (double)(geom_host.Nblock_x * threadsPerBlock) );

    return geom_host;
  }



  size_t DeviceParams::roundDown2pow(size_t x) {
    size_t s=1;
    while (s<=x) s <<= 1;
    s >>= 1;
    return s;
  }


  void DeviceParams::autoDetect() {
    unifiedAddressing = CudaGetConfig(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) == 1;
    asyncTransfers = CudaGetConfig(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP) == 1;
    smem = CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK );
    smem_default = 0;
    max_gridx = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X ) );
    max_gridy = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y ) );
    max_gridz = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z ) );
    max_blockx = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X ) );
    max_blocky = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y ) );
    max_blockz = roundDown2pow( CudaGetConfig( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z ) );

    int ma,mi;
    if (!boolNoReadSM)
      CudaGetSM(&ma,&mi);
    major = ma;
    minor = mi;
    divRnd = major >= 2;

    QDP_info_primary("Compute capability (major)              = %d",major);
    QDP_info_primary("Compute capability (minor)              = %d",minor);
    QDP_info_primary("Divide with IEEE 754 compliant rounding = %d",divRnd);
    QDP_info_primary("Sqrt with IEEE 754 compliant rounding   = %d",divRnd);
    QDP_info_primary("unified addr                            = %d",unifiedAddressing ? 1 : 0);
    QDP_info_primary("asyncTransfers                          = %d",asyncTransfers ? 1 : 0);
    QDP_info_primary("smem                                    = %d",smem);
    QDP_info_primary("max_gridx                               = %d",max_gridx);
    QDP_info_primary("max_gridy                               = %d",max_gridy);
    QDP_info_primary("max_gridz                               = %d",max_gridz);
    QDP_info_primary("max_blockx                              = %d",max_blockx);
    QDP_info_primary("max_blocky                              = %d",max_blocky);
    QDP_info_primary("max_blockz                              = %d",max_blockz);
  }

  void DeviceParams::setSM(int sm) {
    QDP_info_primary("Compiling LLVM IR to PTX for compute capability sm_%d (instead of autodetect)",sm);
    major = sm / 10;
    minor = sm % 10;
    boolNoReadSM = true;
  }

}
