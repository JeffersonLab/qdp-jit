// -*- C++ -*-

/*! @file
 * @brief Outer lattice routines specific to a parallel platform with scalar layout
 */

#ifndef QDP_PARSCALAR_MAPRESOURCE_H
#define QDP_PARSCALAR_MAPRESOURCE_H

#include "qmp.h"

namespace QDP {


  // The MPI resources class for an FnMap.
  // An instance for each (dest/src node,msg_size) combination
  // exists so they can be reused over the whole program lifetime.
  // Can't allocate resources in constructor, since I use ::operator new
  // to allocate a whole array of them. This is necessary since if a 
  // size of 2 occurs in the logical machine grid, then forward/backward
  // points to the same MPI node.

struct FnMapRsrc
{
private:
  FnMapRsrc(const FnMapRsrc&);
  int send_buf_id = -1;
  int recv_buf_id = -1;
public:
  FnMapRsrc():bSet(false) {};

  int getSendBufId() const { assert(send_buf_id>=0); return send_buf_id; }
  int getRecvBufId() const { assert(recv_buf_id>=0); return recv_buf_id; }
  
  void setup(int _destNode,int _srcNode,int _sendMsgSize,int _rcvMsgSize);
  void cleanup();

  ~FnMapRsrc() {
    //QDPIO::cout << "~FnMapRsrc()\n";
  }

  void qmp_wait() const;
  void send_receive() const;

  void * getSendBufDevPtr() const { return send_buf_dev; }
  void * getRecvBufDevPtr() const { return recv_buf_dev; }

  bool bSet;
  mutable void * send_buf;
  mutable void * recv_buf;
  void * send_buf_dev;
  void * recv_buf_dev;

  int srcnum, dstnum;
  QMP_msgmem_t msg[2];
  QMP_msghandle_t mh_a[2], mh;
  QMP_mem_t *send_buf_mem;
  QMP_mem_t *recv_buf_mem;
};

  

} // namespace QDP
#endif
