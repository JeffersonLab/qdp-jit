// -*- C++ -*-

/*! @file
 * @brief Outer lattice routines specific to a parallel platform with scalar layout
 */

#ifndef QDP_PARSCALAR_INTERNAL_H
#define QDP_PARSCALAR_INTERNAL_H

#include "qmp.h"

namespace QDP {
  namespace QDPInternal
  {
    //! Unsigned accumulate
    inline void sumAnUnsigned(void* inout, void* in)
    {
      *(unsigned int*)inout += *(unsigned int*)in;
    }

    //! Global And
    inline void globalCheckAnd(void* inout, void* in)
    {
      *(unsigned int*)inout = *(unsigned int*)inout & *(unsigned int*)in;
    }


    //! Wrapper to get a functional unsigned global sum
    inline void globalSumArray(unsigned int *dest, int len)
    {
      for(int i=0; i < len; i++, dest++)
	QMP_binary_reduction(dest, sizeof(unsigned int), sumAnUnsigned);
    }

    //! Low level hook to QMP_global_sum
    inline void globalSumArray(int *dest, int len)
    {
      for(int i=0; i < len; i++, dest++)
	QMP_sum_int(dest);
    }

    //! Low level hook to QMP_global_sum
    inline void globalSumArray(float *dest, int len)
    {
      QMP_sum_float_array(dest, len);
    }

    //! Low level hook to QMP_global_sum
    inline void globalSumArray(double *dest, int len)
    {
      QMP_sum_double_array(dest, len);
    }

    //! Sum across all nodes
    template<class T>
    inline void globalSum(T& dest)
    {
      // The implementation here is relying on the structure being packed
      // tightly in memory - no padding
      typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 0 
      QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
      QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
      QDPIO::cout << "Calling global sum array with length " << sizeof(T)/sizeof(W) << endl;
#endif
      globalSumArray((W *)&dest, int(sizeof(T)/sizeof(W))); // call appropriate hook
    }



    template<class T>
    inline void globalSum(OScalar<T>& dest)
    {
      // The implementation here is relying on the structure being packed
      // tightly in memory - no padding
      typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 0 
      QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
      QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
      QDPIO::cout << "Calling global sum array with length " << sizeof(T)/sizeof(W) << endl;
#endif
      if (QMP_get_number_of_nodes() > 1) {
	globalSumArray((W *)dest.getF(), int(sizeof(T)/sizeof(W))); // call appropriate hook
      } else {
#ifdef GPU_DEBUG
	QDP_debug("global sum: no MPI reduction");
#endif      
      }
    }

    template<>
    inline void globalSum(double& dest)
    {
#if 0 
      QDPIO::cout << "Using simple sum_double" << endl;
#endif
      QMP_sum_double(&dest);
    }



    //! Low level hook to QMP_max_double
    inline void globalMaxValue(float* dest)
    {
      QMP_max_float(dest);
    }

    //! Low level hook to QMP_max_double
    inline void globalMaxValue(double* dest)
    {
      QMP_max_double(dest);
    }


    //! Global max across all nodes
    template<class T>
    inline void globalMax(T& dest)
    {
      typedef typename WordType<T>::Type_t  W;   // find the machine word type

      if (QMP_get_number_of_nodes() > 1) {
	globalMaxValue((W *)dest.getF());
      } else {
#ifdef GPU_DEBUG    
	QDP_debug("global max: no MPI reduction");
#endif      
      }
    }




    //! Low level hook to QMP_min_float
    inline void globalMinValue(float* dest)
    {
      QMP_min_float(dest);
    }

    //! Low level hook to QMP_min_double
    inline void globalMinValue(double* dest)
    {
      QMP_min_double(dest);
    }

    //! Global min across all nodes
    template<class T>
    inline void globalMin(T& dest)
    {
      typedef typename WordType<T>::Type_t  W;   // find the machine word type

      if (QMP_get_number_of_nodes() > 1) {
	globalMinValue((W *)dest.getF());
      } else {
#ifdef GPU_DEBUG    
	QDP_debug("global min: no MPI reduction");
#endif      
      }
    }


    //! Broadcast from primary node to all other nodes
    template<class T>
    inline void broadcast(T& dest)
    {
      QMP_broadcast((void *)&dest, sizeof(T));
    }

    //! Broadcast from primary node to all other nodes
    inline void broadcast(void* dest, size_t nbytes)
    {
      QMP_broadcast(dest, nbytes);
    }


    //! Wrapper to get a functional global And
    inline void globalAnd(bool& dest)
    {
      QMP_binary_reduction(&dest, sizeof(bool), globalCheckAnd);
    }

    inline void barrier()
    {
      QMP_barrier();
    }
  } // QDPInternal
  
} // QDP

#endif
