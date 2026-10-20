#ifndef QDP_PARSCALAR_INTERNAL_H
#define QDP_PARSCALAR_INTERNAL_H


#include "qmp.h"

namespace QDP {
  namespace QDPInternal
  {
    inline void sumAnUnsigned(void* inout, void* in)
    {
      *(unsigned int*)inout += *(unsigned int*)in;
    }

    inline void globalCheckAnd(void* inout, void* in)
    {
      *(unsigned int*)inout = *(unsigned int*)inout & *(unsigned int*)in;
    }

    inline void globalSumArray(unsigned int *dest, int len)
    {
      for(int i=0; i < len; i++, dest++)
	QMP_binary_reduction(dest, sizeof(unsigned int), sumAnUnsigned);
    }

    inline void globalSumArray(int *dest, int len)
    {
      for(int i=0; i < len; i++, dest++)
	QMP_sum_int(dest);
    }

    inline void globalSumArray(float *dest, int len)
    {
      QMP_sum_float_array(dest, len);
    }

    inline void globalSumArray(double *dest, int len)
    {
      QMP_sum_double_array(dest, len);
    }

    template<Arithmetic T>
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
	//globalSumArray((W *)dest.get_ptr(), int(sizeof(T)/sizeof(W))); // call appropriate hook
	globalSumArray((W *)dest.get_ptr(), int( jit_sizeof(dest)/sizeof(W))); // call appropriate hook
      } else {
#ifdef GPU_DEBUG
	QDP_debug("global sum: no MPI reduction");
#endif      
      }
    }

    template<>
    inline void globalSum(double& dest)
    {
      QMP_sum_double(&dest);
    }


    inline void globalMaxValue(float* dest)
    {
      QMP_max_float(dest);
    }


    inline void globalMaxValue(double* dest)
    {
      QMP_max_double(dest);
    }


    template<class T>
    inline void globalMax(T& dest)
    {
      typedef typename WordType<T>::Type_t  W;   // find the machine word type

      if (QMP_get_number_of_nodes() > 1) {
	globalMaxValue((W *)dest.get_ptr());
      } else {
#ifdef GPU_DEBUG    
	QDP_debug("global max: no MPI reduction");
#endif      
      }
    }




    template<class T>
    struct Broadcaster
    {
      static void send(const T& dest)
      {
	static_assert(std::is_trivially_copyable_v<T> && !requires(const T& t) { t.get_ptr(); },
		      "Broadcaster: only plain data is broadcast by value");
	QMP_broadcast((void *)&dest, sizeof(T));
      }
    };


    template<class T>
    struct Broadcaster< OScalar<T> >
    {
      static void send(const OScalar<T>& dest)
      {
	QMP_broadcast((void *)dest.get_ptr(), jit_sizeof(dest));
      }
    };



    template<class T>
    inline void broadcast(T& dest)
    {
      Broadcaster<T>::send(dest);
    }

    
    inline void broadcast(void* dest, size_t nbytes)
    {
      QMP_broadcast(dest, nbytes);
    }


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
