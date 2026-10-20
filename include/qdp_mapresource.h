#ifndef QDP_MAPRESOURCE_H
#define QDP_MAPRESOURCE_H




namespace QDP {


class FnMapRsrcMatrix
{
  std::map< std::pair<int,int> , std::pair< int , std::vector<FnMapRsrc*> > > m2d; // [send msg size , dest node] --> (count , vector of resources)

public:

  void cleanup();

  std::pair< int , std::vector<FnMapRsrc*> >& get(int _destNode,int _srcNode,
						  int _sendMsgSize,int _rcvMsgSize);
  
  static FnMapRsrcMatrix& Instance() {
    static FnMapRsrcMatrix singleton;
    return singleton;
  }

};



class RsrcWrapper
{
  int destnode;
  int srcenode;
  
  std::pair< int , std::vector<FnMapRsrc*> >* pPair;
  const FnMapRsrc* cached;
  bool rAlloc;
public:
  ~RsrcWrapper() {
    if (rAlloc) {
      pPair->first--;
#if QDP_DEBUG >= 3
      if (pPair->first < 0)
	QDP_error_exit("<0");
#endif
    }
  }
  
  RsrcWrapper( int destnode_ , int srcenode_): destnode(destnode_) , srcenode(srcenode_) , cached(nullptr) , rAlloc(false)
  {
  }

  const FnMapRsrc& getResource(int srcnum_, int dstnum_) {
#if QDP_DEBUG >= 3
    if ( !srcenodes.size() || !destnodes.size() )
      QDP_error_exit("FnMapRsrc& getResource srcnode_size=%d destnode_size=%d", srcenodes.size() , destnodes.size() );
#endif
    pPair = &FnMapRsrcMatrix::Instance().get( destnode , srcenode , dstnum_ , srcnum_ );
    cached = pPair->second.at(pPair->first++);
    rAlloc=true;
    return *cached;
  }

  const FnMapRsrc& get() const {
#if QDP_DEBUG >= 3
    if (!rAlloc)
      QDP_error_exit("RsrcWrapper::get() internal error");
#endif
    assert(cached);
    return *cached;
  }

};








} // namespace QDP

#endif
