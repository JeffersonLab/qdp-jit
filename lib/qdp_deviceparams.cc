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
    max_gridx = roundDown2pow( HipGetConfig( hipDeviceAttributeMaxGridDimX ) );
    max_gridy = roundDown2pow( HipGetConfig( hipDeviceAttributeMaxGridDimY ) );
    max_gridz = roundDown2pow( HipGetConfig( hipDeviceAttributeMaxGridDimZ ) );
    max_blockx = roundDown2pow( HipGetConfig( hipDeviceAttributeMaxBlockDimX ) );
    max_blocky = roundDown2pow( HipGetConfig( hipDeviceAttributeMaxBlockDimY ) );
    max_blockz = roundDown2pow( HipGetConfig( hipDeviceAttributeMaxBlockDimZ ) );

    major = HipGetConfig( hipDeviceAttributeComputeCapabilityMajor );
    minor = HipGetConfig( hipDeviceAttributeComputeCapabilityMinor );

    smem = HipGetConfig( hipDeviceAttributeMaxSharedMemoryPerMultiprocessor );

    sm_count = HipGetConfig( hipDeviceAttributeMultiprocessorCount );

    QDPIO::cout << "Trying GPU Direct !!!\n";
    GPUDirect = true;

    QDP_info_primary("Compute capability (major)              = %d",major);
    QDP_info_primary("Compute capability (minor)              = %d",minor);
    QDP_info_primary("sm count                                = %d",sm_count);
    QDP_info_primary("max_shared mem                          = %d",smem);
    QDP_info_primary("max_gridx                               = %d",max_gridx);
    QDP_info_primary("max_gridy                               = %d",max_gridy);
    QDP_info_primary("max_gridz                               = %d",max_gridz);
    QDP_info_primary("max_blockx                              = %d",max_blockx);
    QDP_info_primary("max_blocky                              = %d",max_blocky);
    QDP_info_primary("max_blockz                              = %d",max_blockz);
  }

  void DeviceParams::setDefaultGPU(int ngpu) {
    defaultGPU = ngpu;
  }


}
