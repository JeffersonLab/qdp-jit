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
  QDPIO::cout << "Calling " << dest.size() << "x global sum array with length " << sizeof(P)/sizeof(W) << endl;
#endif

#if 0
  for (int i = 0 ; i < dest.size() ; ++i )
    globalSumArray( (W *)dest[i].getF(), sizeof(P)/sizeof(W)); // call appropriate hook
#else
    globalSumArray((W *)dest.slice(), dest.size()*sizeof(P)/sizeof(W)); // call appropriate hook
#endif  
}


    //! Global sum on a multi1d
  template<class T1>
  inline void globalSumArray(multi1d< OScalar<T1> >& dest)
  {
    //QDPIO::cout << "globalSumArray special " << endl;
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T1>::Type_t  W;   // find the machine word type
    typedef          T1                    P;   // Primitive type

#if 0
    QDPIO::cout << "sizeof(P) = " << sizeof(P) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling multi1d global sum array with length " << dest.size()*sizeof(P)/sizeof(W) << endl;
#endif

    globalSumArray((W *)dest.slice(), dest.size()*sizeof(P)/sizeof(W)); // call appropriate hook
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




  //! Broadcast from primary node to all other nodes
  void broadcast_str(std::string& dest);

  template<class T>
  inline void broadcast(T& dest);

  //! Broadcast a string from primary node to all other nodes
  template<>
  inline void broadcast(std::string& dest)
  {
    broadcast_str(dest);
  }

  //! Call a barrier
  void barrier();

} // QDPInternal

} // QDP

#endif
