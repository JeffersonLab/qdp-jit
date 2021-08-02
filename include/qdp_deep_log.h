#ifndef QDP_FILE_DEEP_LOG
#define QDP_FILE_DEEP_LOG

namespace QDP {

#ifdef QDP_DEEP_LOG
  void gpu_deep_logger_close();
  void gpu_deep_logger_create ( void* host_ptr , std::string type_W , int size_T , int start , int count , std::string pretty );
  void gpu_deep_logger_compare( void* host_ptr , std::string type_W , int size_T , int start , int count , std::string pretty );
  void gpu_deep_logger        ( void* host_ptr , std::string type_W , int size_T , int start , int count , std::string pretty );
#endif
  
}
#endif
