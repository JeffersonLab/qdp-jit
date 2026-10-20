#ifndef QDP_INTERNAL_H
#define QDP_INTERNAL_H

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
	recvFromWait((void *)dest.get_ptr(), srcnode, jit_sizeof(dest));

      if (Layout::nodeNumber() == srcnode)
	sendToWait((void *)dest.get_ptr(), 0, jit_sizeof(dest));
    }
  }



  template<class T1>
  inline void globalSumArray(multi1d< OScalar<T1> >& dest)
  {
    typedef typename WordType<T1>::Type_t  W;   // find the machine word type

    if (dest.size() < 1)
      {
	QDPIO::cout << "globalSumArray(multi1d<OScalar>& dest) with zero-size argument\n";
	QDP_abort(1);
      }

    for (int i = 0 ; i < dest.size() ; ++i )
      globalSumArray( (W *)dest[i].get_ptr(), jit_sizeof(dest[i])/sizeof(W)); // call appropriate hook
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
