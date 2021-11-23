#ifndef QDP_FILE_DEEP_LOG
#define QDP_FILE_DEEP_LOG

namespace QDP {

#ifdef QDP_DEEP_LOG
  void gpu_deep_logger_close();
  void gpu_deep_logger        ( const void* host_ptr , std::string type_W , size_t field_size , std::string pretty , bool isLat );
#endif
  
}
#endif
