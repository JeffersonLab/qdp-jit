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

    if (!jit_config_get_gpu_direct()) {
      gpu_host_alloc(&send_buf,dstnum);
      gpu_host_alloc(&recv_buf,srcnum);
    }

    if (jit_config_get_gpu_direct()) {
      recv_buf_id = QDP_get_global_cache().addDeviceStaticDirect( &recv_buf_dev , srcnum);
      send_buf_id = QDP_get_global_cache().addDeviceStaticDirect( &send_buf_dev , dstnum);
    } else {
      recv_buf_id = QDP_get_global_cache().addDeviceStatic( &recv_buf_dev , srcnum);
      send_buf_id = QDP_get_global_cache().addDeviceStatic( &send_buf_dev , dstnum);
    }

    if (!jit_config_get_gpu_direct()) {
      msg[0] = QMP_declare_msgmem( recv_buf , srcnum );
    } else {
      msg[0] = QMP_declare_msgmem( recv_buf_dev , srcnum );
    }

    if( msg[0] == (QMP_msgmem_t)NULL ) { 
      QDP_error_exit("QMP_declare_msgmem for msg[0] failed in Map::operator()\n");
    }

    if (!jit_config_get_gpu_direct()) {
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
      QDP_get_global_cache().signoff( send_buf_id );
      QDP_get_global_cache().signoff( recv_buf_id );
     
      if (!jit_config_get_gpu_direct()) {
        gpu_host_free(send_buf);
        gpu_host_free(recv_buf);
      }

      bSet=false;
    }
  }


  void FnMapRsrc::qmp_wait() const {
    QMP_status_t err;
    if ((err = QMP_wait(mh)) != QMP_SUCCESS)
      QDP_error_exit(QMP_error_string(err));

    if (!jit_config_get_gpu_direct()) {
      gpu_memcpy_h2d( recv_buf_dev , recv_buf , srcnum );
    }
  }


  void FnMapRsrc::send_receive() const {
    QMP_status_t err;

    if (!jit_config_get_gpu_direct()) {
      gpu_memcpy_d2h( send_buf , send_buf_dev , dstnum );
    }

    if ((err = QMP_start(mh)) != QMP_SUCCESS)
      QDP_error_exit(QMP_error_string(err));
  }




} // namespace QDP;
