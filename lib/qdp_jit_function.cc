#include "qdp.h"

namespace QDP {

  namespace {
    std::map< JitFunction::Func_t , int > call_counter;
    bool record_stats = false;
    std::vector<JitFunction*> vec_functions;
  }

  
  void gpu_set_record_stats()
  {
    record_stats = true;
  }

  
  bool gpu_get_record_stats()
  {
    return record_stats;
  }

  std::vector<JitFunction*>& gpu_get_functions()
  {
    return vec_functions;
  }


  JitFunction::JitFunction(): isEmpty(true), called(0) {
    vec_functions.push_back( this );
  }


  void JitFunction::check_empty() {
    if (isEmpty)
      {
	QDPIO::cerr << "internal error. jit function not set.\n";
	QDP_abort(1);
      }
    }


  void JitFunction::inc_call_counter() {
    check_empty();
    call_counter[ function ]++;
  }
  
  int JitFunction::get_call_counter() {
    check_empty();
    return call_counter[ function ];
  }

  

} // QDP
