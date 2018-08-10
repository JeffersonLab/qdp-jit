#ifndef QDP_POOLBISECT_H
#define QDP_POOLBISECT_H

namespace QDP {

  void qdp_pool_bisect();

  bool   qdp_cache_get_pool_bisect();
  size_t qdp_cache_get_pool_bisect_max();
  
  void qdp_cache_set_pool_bisect(bool b);
  void qdp_cache_set_pool_bisect_max(size_t val);



} // QDP
#endif

