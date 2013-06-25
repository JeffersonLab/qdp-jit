// -*- c++ -*-

#ifndef QDP_DEVPARAMS_H
#define QDP_DEVPARAMS_H


#include <list>


namespace QDP {

  struct kernel_geom_t {
    int threads_per_block;
    int Nblock_x;
    int Nblock_y;
  };

  kernel_geom_t getGeom(int numSites , int threadsPerBlock);


  class DeviceParams {
  public:
    static DeviceParams& Instance()
    {
      static DeviceParams singleton;
      return singleton;
    }

    void setSM(int sm);

    size_t getMaxGridX() const {return max_gridx;}
    size_t getMaxGridY() const {return max_gridy;}
    size_t getMaxGridZ() const {return max_gridz;}

    size_t getMaxBlockX() const {return max_blockx;}
    size_t getMaxBlockY() const {return max_blocky;}
    size_t getMaxBlockZ() const {return max_blockz;}

    int getMaxSMem() const {return smem;}
    int getDefaultSMem() const {return smem_default;}

    bool getDivRnd() { return divRnd; }
    bool getSyncDevice() { return syncDevice; }
    bool getGPUDirect() { return GPUDirect; }
    void setENVVAR(const char * envvar_) {
      envvar = envvar_;
    } 
    const char* getENVVAR() {
      return envvar.c_str();
    }
    void setSyncDevice(bool sync) { 
      QDP_info_primary("Setting device sync = %d",(int)sync);
      syncDevice = sync;
    };
    void setGPUDirect(bool direct) { 
      QDP_info_primary("Setting GPU Direct = %d",(int)direct);
      GPUDirect = direct;
    };

    int& getMaxKernelArg() { return maxKernelArg; }
    int getMajor() { return major; }
    int getMinor() { return minor; }

    bool getAsyncTransfers() { return asyncTransfers; }

    void autoDetect();

  private:
    DeviceParams(): GPUDirect(false), syncDevice(false), maxKernelArg(512), boolNoReadSM(false) {}; // Private constructor
    DeviceParams(const DeviceParams&);                                           // Prevent copy-construction
    DeviceParams& operator=(const DeviceParams&);
    size_t roundDown2pow(size_t x);

  private:
    bool boolNoReadSM;
    int device;
    std::string envvar;
    bool GPUDirect;
    bool syncDevice;
    bool asyncTransfers;
    bool unifiedAddressing;
    bool divRnd;
    int maxKernelArg;

    int smem;
    int smem_default;

    int max_gridx;
    int max_gridy;
    int max_gridz;

    int max_blockx;
    int max_blocky;
    int max_blockz;

    int major;
    int minor;

  };


}




#endif
