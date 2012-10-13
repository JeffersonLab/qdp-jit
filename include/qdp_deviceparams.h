// -*- c++ -*-

#ifndef QDP_DEVPARAMS_H
#define QDP_DEVPARAMS_H


#include <list>


namespace QDP {


  class DeviceParams {
  public:
    static DeviceParams& Instance()
    {
      static DeviceParams singleton;
      return singleton;
    }

    void setCC(int sm);

    int getMaxGridX() const {return max_gridx;}
    int getMaxGridY() const {return max_gridy;}
    int getMaxGridZ() const {return max_gridz;}

    int getMaxBlockX() const {return max_blockx;}
    int getMaxBlockY() const {return max_blocky;}
    int getMaxBlockZ() const {return max_blockz;}

    int getMaxSMem() const {return smem;}
    int getDefaultSMem() const {return smem_default;}

    bool getSyncDevice() { return syncDevice; }
    void setSyncDevice(bool sync) { 
      QDP_info_primary("Setting device sync = %u",sync);
      syncDevice = sync;
    };

    int& getMaxKernelArg() { return maxKernelArg; }

    bool getAsyncTransfers() { return asyncTransfers; }

    void autoDetect();

  private:
    DeviceParams(): syncDevice(false), maxKernelArg(512) {};   // Private constructor
    DeviceParams(const DeviceParams&);                            // Prevent copy-construction
    DeviceParams& operator=(const DeviceParams&);
    int roundDown2pow(int x);

  private:
    int device;
    bool syncDevice;
    bool asyncTransfers;
    bool unifiedAddressing;
    int maxKernelArg;

    int smem;
    int smem_default;

    int max_gridx;
    int max_gridy;
    int max_gridz;

    int max_blockx;
    int max_blocky;
    int max_blockz;
  };


}




#endif
