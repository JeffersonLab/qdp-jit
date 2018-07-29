#include "qdp.h"

namespace QDP {


  void FnMapRsrcMatrix::cleanup() {
    //QDPIO::cout << "FnMapRsrcMatrix cleanup\n";
    for(unsigned int i=0;i<numSendMsgSize;i++) {
      for(unsigned int q=0;q<numDestNode;q++) {
	//QDPIO::cout << "cleanup m2d(" << i << "," << q << ")\n";
	for (std::vector<FnMapRsrc*>::iterator v = (*m2d(i,q)).second.begin() ; v != (*m2d(i,q)).second.end() ; ++v )
	  delete *v;
	//std::for_each( (*m2d(i,q)).second.begin() , (*m2d(i,q)).second.end() , this->*del );
	delete m2d(i,q);
      }
    }
  }


  std::pair< int , std::vector<FnMapRsrc*> >* FnMapRsrcMatrix::get(int _destNode,int _srcNode,
								   int _sendMsgSize,int _rcvMsgSize) {
    bool found = false;
    unsigned int xDestNode=0;
    for(; xDestNode < destNode.size(); ++xDestNode)
      if (destNode[xDestNode] == _destNode)
	{
	  found = true;
	  break;
	}
    if (! found) {
      if (destNode.size() == numDestNode) {
	QDP_error_exit("FnMapRsrcMatrix not enough space in destNode");
      } else {
	destNode.push_back(_destNode);
	xDestNode=destNode.size()-1;
      }
    }
    //QDPIO::cout << "using node place = " << xDestNode << "\n";


    found = false;
    unsigned int xSendmsgsize=0;
    for(; xSendmsgsize < sendMsgSize.size(); ++xSendmsgsize)
      if (sendMsgSize[xSendmsgsize] == _sendMsgSize)
	{
	  found = true;
	  break;
	}
    if (! found) {
      if (sendMsgSize.size() == numSendMsgSize) {
	QDP_error_exit("FnMapRsrcMatrix not enough space in sendmsgsize");
      } else {
	sendMsgSize.push_back(_sendMsgSize);
	xSendmsgsize=sendMsgSize.size()-1;
      }
    }
    //QDPIO::cout << "using msg_size place = " << xSendmsgsize << "\n";

    std::pair< int , std::vector<FnMapRsrc*> >& pos = *m2d(xSendmsgsize,xDestNode);

#if QDP_DEBUG >= 3
    // SANITY
    if ( pos.second.size() <  pos.first )
      QDP_error_exit(" pos.second.size()=%d  pos.first=%d",pos.second.size(), pos.first);
#endif

    // Vector's size large enough ?
    if ( pos.second.size() ==  (unsigned)pos.first ) {
      //QDPIO::cout << "allocate and setup new rsrc-obj (destnode=" << _destNode << ",sndmsgsize=" << _sendMsgSize << ")\n";
      pos.second.push_back( new FnMapRsrc() );
      pos.second.at(pos.first)->setup( _destNode, _srcNode, _sendMsgSize, _rcvMsgSize );
    }

    //QDPIO::cout << "returning rsrc-obj " << pos.first << "\n";

    return &pos;
  }



} // namespace QDP
