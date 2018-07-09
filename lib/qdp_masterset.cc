// -*- C++ -*-

#include "qdp.h"
#include "qdp_util.h"

namespace QDP {


  MasterSet& MasterSet::Instance()
  {
    static MasterSet singleton;
    return singleton;
  }


  void MasterSet::registrate(Set& set)
  {
    //QDPIO::cerr << "masterset: register set with " << set.numSubsets() << " subsets\n";

    for ( int i = 0 ; i < set.numSubsets() ; ++i ) {
      //QDPIO::cerr << "subset no. " << vecSubset.size() << "\n";
      set[i].setId( vecSubset.size() );
      vecSubset.push_back( &set[i] );
    }
  }

  const Subset& MasterSet::getSubset(int id)
  {
    assert( id >= 0 && (unsigned)id < vecSubset.size() && "MasterSet::getSubset out of range");
    return *vecSubset[id];
  }

  int MasterSet::numSubsets() const {
    return vecSubset.size();
  }


} // namespace QDP


