// -*- C++ -*-

/*! @file
 * @brief Outer lattice routines specific to a parallel platform with scalar layout
 */

#ifndef QDP_SCALAR_INTERNAL_H
#define QDP_SCALAR_INTERNAL_H


namespace QDP {
  namespace QDPInternal
  {
    //! Wrapper to get a functional unsigned global sum
    inline void globalSumArray(unsigned int *dest, int len)
    {
    }

    //! Low level hook to QMP_global_sum
    inline void globalSumArray(int *dest, int len)
    {
    }

    //! Low level hook to QMP_global_sum
    inline void globalSumArray(float *dest, int len)
    {
    }

    //! Low level hook to QMP_global_sum
    inline void globalSumArray(double *dest, int len)
    {
    }

    //! Sum across all nodes
    template<class T>
    inline void globalSum(T& dest)
    {
    }



    template<class T>
    inline void globalSum(OScalar<T>& dest)
    {
      // the only side-effect we have here is that the call makes sure, the OScalar is in host memory.
      // So, make sure of that.
      dest.getF();
    }

    template<>
    inline void globalSum(double& dest)
    {
    }


    //! Low level hook to QMP_max_double
    inline void globalMaxValue(float* dest)
    {
    }

    //! Low level hook to QMP_max_double
    inline void globalMaxValue(double* dest)
    {
    }

    //! Global max across all nodes
    template<class T>
    inline void globalMax(T& dest)
    {
      // the only side-effect we have here is that the call makes sure, the OScalar is in host memory.
      // So, make sure of that.
      dest.getF();
    }


    //! Low level hook to QMP_min_float
    inline void globalMinValue(float* dest)
    {
    }

    //! Low level hook to QMP_min_double
    inline void globalMinValue(double* dest)
    {
    }

    //! Global min across all nodes
    template<class T>
    inline void globalMin(T& dest)
    {
      // the only side-effect we have here is that the call makes sure, the OScalar is in host memory.
      // So, make sure of that.
      dest.getF();
    }


    //! Broadcast from primary node to all other nodes
    template<class T>
    inline void broadcast(T& dest)
    {
    }

    //! Broadcast from primary node to all other nodes
    inline void broadcast(void* dest, size_t nbytes)
    {
    }

    //! Wrapper to get a functional global And
    inline void globalAnd(bool& dest)
    {
    }

  } // QDPInternal
  
} // QDP

#endif
