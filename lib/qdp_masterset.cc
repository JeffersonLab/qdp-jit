#include "qdp.h"
#include "qdp_util.h"

namespace QDP {

  MasterSet& MasterSet::Instance()
  {
    static MasterSet singleton;
    return singleton;
  }


  void MasterSet::add(Set& set)
  {
    for ( int i = 0 ; i < set.numSubsets() ; ++i )
      {
	set[i].setId( generations.size() );
	generations.push_back( 0 );
      }
  }

  void MasterSet::remade(const Set& set)
  {
    for ( int i = 0 ; i < set.numSubsets() ; ++i )
      ++generations.at( set[i].getId() );
  }

  int MasterSet::numSubsets() const {
    return generations.size();
  }


} // namespace QDP


