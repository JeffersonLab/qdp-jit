/*! @file
 * @brief Parscalar specific routines
 * 
 * Routines for parscalar implementation
 */


#include "qdp.h"
//#include "qdp_util.h"
//#include "qmp.h"


namespace QDP {

  void FnMapRsrc::setup(int _destNode,int _srcNode,int _sendMsgSize,int _rcvMsgSize)
  {
    assert(!"I shouldn't be here.");
  }


  void FnMapRsrc::cleanup()
  {
    if (bSet) {
      assert(!"I shouldn't be here.");
    }
  }


  void FnMapRsrc::qmp_wait() const
  {
    assert(!"I shouldn't be here.");
  }


  void FnMapRsrc::send_receive() const
  {
    assert(!"I shouldn't be here.");
  }

} // QDP
