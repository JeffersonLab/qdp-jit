#include "qdp.h"

namespace QDP {


  void FnMapRsrcMatrix::cleanup()
  {
    for (auto& i : m2d)
      {
	for (auto& r: i.second.second )
	  {
	    delete r;
	  }
      }
    m2d.clear();
  }


  std::pair< int , std::vector<FnMapRsrc*> >& FnMapRsrcMatrix::get(int _destNode,int _srcNode,
								  int _sendMsgSize,int _rcvMsgSize)
  {
    auto key = std::make_pair( _sendMsgSize , _destNode );
    
    if ( ! m2d.count(key) )
      {
	m2d[key].first = 0;
      }
    
    std::pair< int , std::vector<FnMapRsrc*> >& pos = m2d[key];

    if ( pos.second.size() ==  (unsigned)pos.first ) {
      pos.second.push_back( new FnMapRsrc() );
      pos.second.at(pos.first)->setup( _destNode, _srcNode, _sendMsgSize, _rcvMsgSize );
    }

    return m2d[key];
  }



} // namespace QDP
