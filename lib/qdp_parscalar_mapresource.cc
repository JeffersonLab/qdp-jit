/*! @file
 * @brief Parscalar specific routines
 * 
 * Routines for parscalar implementation
 */


#include "qdp.h"
#include "qdp_util.h"
#include "qmp.h"


namespace QDP {

  void FnMapRsrc::setup(int _destNode,int _srcNode,int _sendMsgSize,int _rcvMsgSize) {

    bSet=true;

    srcnum=_rcvMsgSize;
    dstnum=_sendMsgSize;

    int srcnode = _srcNode;
    int dstnode = _destNode;

    if (!DeviceParams::Instance().getGPUDirect()) {
      CudaHostAlloc(&send_buf,dstnum,0);
      CudaHostAlloc(&recv_buf,srcnum,0);
    }

    //QDPIO::cout << "Allocating receive buffer on device: " << srcnum << " bytes\n";
    recv_buf_id = QDP_get_global_cache().addDeviceStatic( &recv_buf_dev , srcnum);

    //QDPIO::cout << "Allocating send buffer on device: " << dstnum << " bytes\n";
    send_buf_id = QDP_get_global_cache().addDeviceStatic( &send_buf_dev , dstnum);

    if (!DeviceParams::Instance().getGPUDirect()) {
      msg[0] = QMP_declare_msgmem( recv_buf , srcnum );
    } else {
      msg[0] = QMP_declare_msgmem( recv_buf_dev , srcnum );
    }

    if( msg[0] == (QMP_msgmem_t)NULL ) { 
      QDP_error_exit("QMP_declare_msgmem for msg[0] failed in Map::operator()\n");
    }

    if (!DeviceParams::Instance().getGPUDirect()) {
      msg[1] = QMP_declare_msgmem( send_buf , dstnum );
    } else {
      msg[1] = QMP_declare_msgmem( send_buf_dev , dstnum );
    }


    if( msg[1] == (QMP_msgmem_t)NULL ) {
      QDP_error_exit("QMP_declare_msgmem for msg[1] failed in Map::operator()\n");
    }

    mh_a[0] = QMP_declare_receive_from(msg[0], srcnode, 0);
    if( mh_a[0] == (QMP_msghandle_t)NULL ) { 
      QDP_error_exit("QMP_declare_receive_from for mh_a[0] failed in Map::operator()\n");
    }

    mh_a[1] = QMP_declare_send_to(msg[1], dstnode , 0);
    if( mh_a[1] == (QMP_msghandle_t)NULL ) {
      QDP_error_exit("QMP_declare_send_to for mh_a[1] failed in Map::operator()\n");
    }

    mh = QMP_declare_multiple(mh_a, 2);
    if( mh == (QMP_msghandle_t)NULL ) { 
      QDP_error_exit("QMP_declare_multiple for mh failed in Map::operator()\n");
    }

  }

  void FnMapRsrc::cleanup() {
    if (bSet) {
      QMP_free_msghandle(mh);
      // QMP_free_msghandle(mh_a[1]);
      // QMP_free_msghandle(mh_a[0]);
      QMP_free_msgmem(msg[1]);
      QMP_free_msgmem(msg[0]);
#if 0
      QMP_free_memory(recv_buf_mem);
      QMP_free_memory(send_buf_mem);
#endif
      QDP_get_global_cache().signoff( send_buf_id );
      QDP_get_global_cache().signoff( recv_buf_id );
      CudaHostFree(send_buf);
      CudaHostFree(recv_buf);
    }
  }


  void FnMapRsrc::qmp_wait() const {
    QMP_status_t err;
    if ((err = QMP_wait(mh)) != QMP_SUCCESS)
      QDP_error_exit(QMP_error_string(err));

    if (!DeviceParams::Instance().getGPUDirect()) {
      CudaMemcpyH2D( recv_buf_dev , recv_buf , srcnum );
    }

#if QDP_DEBUG >= 3
    QDP_info("Map: calling free msgs");
#endif
  }


  void FnMapRsrc::send_receive() const {

    QMP_status_t err;
#if QDP_DEBUG >= 3
    QDP_info("Map: send = 0x%x  recv = 0x%x",send_buf,recv_buf);
    QDP_info("Map: establish send=%d recv=%d",destnodes[0],srcenodes[0]);
    {
      const multi1d<int>& me = Layout::nodeCoord();
      multi1d<int> scrd = Layout::getLogicalCoordFrom(destnodes[0]);
      multi1d<int> rcrd = Layout::getLogicalCoordFrom(srcenodes[0]);

      QDP_info("Map: establish-info   my_crds=[%d,%d,%d,%d]",me[0],me[1],me[2],me[3]);
      QDP_info("Map: establish-info send_crds=[%d,%d,%d,%d]",scrd[0],scrd[1],scrd[2],scrd[3]);
      QDP_info("Map: establish-info recv_crds=[%d,%d,%d,%d]",rcrd[0],rcrd[1],rcrd[2],rcrd[3]);
    }
#endif

#if QDP_DEBUG >= 3
    QDP_info("Map: calling start send=%d recv=%d",destnodes[0],srcenodes[0]);
#endif

#ifdef GPU_DEBUG_DEEP
    QDP_info("D2H %d bytes receive buffer",dstnum);
#endif

    if (!DeviceParams::Instance().getGPUDirect()) {
      CudaMemcpyD2H( send_buf , send_buf_dev , dstnum );
    }

    // Launch the faces
    if ((err = QMP_start(mh)) != QMP_SUCCESS)
      QDP_error_exit(QMP_error_string(err));
  }




} // namespace QDP;
