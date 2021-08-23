#ifndef QDP_DYN_FUNCTIONS_H
#define QDP_DYN_FUNCTIONS_H



namespace QDP {




  template<class T1, class Op, class RHS>
  DynKey get_dyn_key(const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
  {
    DynKey ret;

    ret.add( s.hasOrderedRep() ? 1 : 0 );

    DynKeyTag tag(ret);
    bool offnode = forEach( rhs , tag , OrCombine() );

    ret.set_offnode_comms( offnode );
    
    //QDPIO::cout << "dyn_get_key offnode = " << offnode << std::endl;
    
    return ret;
  }

  

} //
#endif
