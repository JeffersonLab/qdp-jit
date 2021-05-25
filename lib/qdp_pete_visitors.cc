#include "qdp.h"

namespace QDP {

  template<>
  void AddressLeaf::setLit<float>( float f ) const
  {
    ids.push_back( QDPCache::ArgKey(QDP_get_global_cache().addJitParamFloat(f)) );
    ids_signoff.push_back( ids.back() );
  }

  
  template<>
  void AddressLeaf::setLit<double>( double d ) const
  {
    ids.push_back( QDPCache::ArgKey(QDP_get_global_cache().addJitParamDouble(d)) );
    ids_signoff.push_back( ids.back() );
  }

  
  template<>
  void AddressLeaf::setLit<int>( int i ) const
  {
    ids.push_back( QDPCache::ArgKey(QDP_get_global_cache().addJitParamInt(i)) );
    ids_signoff.push_back( ids.back() );
  }

  
  template<>
  void AddressLeaf::setLit<int64_t>( int64_t i ) const
  {
    ids.push_back( QDPCache::ArgKey(QDP_get_global_cache().addJitParamInt64(i)) );
    ids_signoff.push_back( ids.back() );
  }

  
  template<>
  void AddressLeaf::setLit<bool>( bool b ) const
  {
    ids.push_back( QDPCache::ArgKey(QDP_get_global_cache().addJitParamBool(b)) );
    ids_signoff.push_back( ids.back() );
  }

}
