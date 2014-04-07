#ifndef QDP_INTERNAL
#define QDP_INTERNAL

namespace QDP
{

namespace QDPInternal
{
  //! Route to another node (blocking)
  void route(void *send_buf, int srce_node, int dest_node, int count);

  //! Wait on send-receive
  void wait(int dir);

  //! Send to another node (wait)
  /*! All nodes participate */
  void sendToWait(void *send_buf, int dest_node, int count);

  //! Receive from another node (wait)
  void recvFromWait(void *recv_buf, int srce_node, int count);

  //! Via some mechanism, get the dest to node 0
  /*! Ultimately, I do not want to use point-to-point */
  template<class T>
  void sendToPrimaryNode(T& dest, int srcnode)
  {
    if (srcnode != 0)
    {
      if (Layout::primaryNode())
	recvFromWait((void *)&dest, srcnode, sizeof(T));

      if (Layout::nodeNumber() == srcnode)
	sendToWait((void *)&dest, 0, sizeof(T));
    }
  }

  //! Unsigned accumulate
  inline void sumAnUnsigned(void* inout, void* in)
  {
    *(unsigned int*)inout += *(unsigned int*)in;
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

  ///////////////////////////////////
#if 0
  //! Global sum on a multi1d
  template<class T>
  inline void globalSumArray(multi1d<T>& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 1
    QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling multi1d global sum array with length " << dest.size()*sizeof(T)/sizeof(W) << endl;
#endif
    globalSumArray((W *)dest.slice(), dest.size()*sizeof(T)/sizeof(W)); // call appropriate hook
  }
#endif
  //////////////////////////////////

  //! Global sum on a multi1d
  template<class T>
  inline void globalSumArray(multi1d<T>& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type
    typedef typename T::SubType_t         P;   // Primitive type

#if 0
    QDPIO::cout << "sizeof(P) = " << sizeof(P) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling multi1d global sum array with length " << sizeof(P)/sizeof(W) << endl;
#endif

    for (int i = 0 ; i < dest.size() ; ++i )
      globalSumArray( (W *)dest[i].getF(), sizeof(P)/sizeof(W)); // call appropriate hook

    //globalSumArray((W *)dest.slice(), dest.size()*sizeof(T)/sizeof(W)); // call appropriate hook
  }

  //! Global sum on a multi2d
  template<class T>
  inline void globalSumArray(multi2d<T>& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 0
    QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling multi2d global sum array with length " << dest.size1()*dest.size2()*sizeof(T)/sizeof(W) << endl;
#endif
    // call appropriate hook
    globalSumArray((W *)dest.slice(0), dest.size1()*dest.size2()*sizeof(T)/sizeof(W));
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
      QDP_debug("global sum: no MPI reduction");
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
      QDP_debug("global max: no MPI reduction");
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
      QDP_debug("global min: no MPI reduction");
    }
  }


  //! Broadcast from primary node to all other nodes
  template<class T>
  inline void broadcast(T& dest)
  {
    QMP_broadcast((void *)&dest, sizeof(T));
  }

  //! Broadcast from primary node to all other nodes
  void broadcast_str(std::string& dest);

  //! Broadcast from primary node to all other nodes
  inline void broadcast(void* dest, size_t nbytes)
  {
    QMP_broadcast(dest, nbytes);
  }

  //! Broadcast a string from primary node to all other nodes
  template<>
  inline void broadcast(std::string& dest)
  {
    broadcast_str(dest);
  }

  //! Global And
  inline void globalCheckAnd(void* inout, void* in)
  {
    *(unsigned int*)inout = *(unsigned int*)inout & *(unsigned int*)in;
  }

  //! Wrapper to get a functional global And
  inline void globalAnd(bool& dest)
  {
    QMP_binary_reduction(&dest, sizeof(bool), globalCheckAnd);
  }



  //! Global Or
  inline void globalCheckOr(void* inout, void* in)
  {
    *(unsigned int*)inout = *(unsigned int*)inout | *(unsigned int*)in;
  }

  //! Wrapper to get a functional global Or
  inline void globalOr(bool& dest)
  {
    QMP_binary_reduction(&dest, sizeof(bool), globalCheckOr);
  }


}

}

#endif
