#include "qdp.h"

namespace QDP {

  int DeviceParams::roundDown2pow(int x) {
    int s=1;
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
    QDP_info_primary("unified addr   = %d",unifiedAddressing ? 1 : 0);
    QDP_info_primary("asyncTransfers = %d",asyncTransfers ? 1 : 0);
    QDP_info_primary("smem           = %d",smem);
    QDP_info_primary("max_gridx      = %d",max_gridx);
    QDP_info_primary("max_gridy      = %d",max_gridy);
    QDP_info_primary("max_gridz      = %d",max_gridz);
    QDP_info_primary("max_blockx     = %d",max_blockx);
    QDP_info_primary("max_blocky     = %d",max_blocky);
    QDP_info_primary("max_blockz     = %d",max_blockz);
  }

  void DeviceParams::setCC(int sm) {
    switch(sm) {
    case 12:
      asyncTransfers = false;
      smem = 16*1024;
      smem_default = 0;
      max_gridx  = max_gridy = 32768; // We need a power of 2 here!
      max_gridz = 1;
      max_blockx = max_blocky = 512;
      max_blockz = 64;
      break;
    case 20:
    case 21:
      asyncTransfers = true;
      smem = 48*1024;
      smem_default = 0;
      max_gridx  = max_gridy = max_gridz = 32768; // We need a power of 2 here!
      max_blockx = max_blocky = 1024;
      max_blockz = 64;
      break;
    case 30:
      asyncTransfers = true;
      smem = 48*1024;
      smem_default = 0;
      max_gridx  = max_gridy = max_gridz = 512 * 1024; // Its 2^31-1, but this value is large enough
      max_blockx = max_blocky = 1024;
      max_blockz = 64;
      break;
    default:
      QDP_error_exit("DeviceParams::setCC compute capability %d not known!",sm);
    }
    if (Layout::primaryNode())
      QDP_info_primary("CUDA device compute capability set to sm_%d",sm);
  }

}
