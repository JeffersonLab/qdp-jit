// -*- C++ -*-

#ifndef QDP_DATALAYOUT_H
#define QDP_DATALAYOUT_H


namespace QDP {

  typedef std::pair< int , llvm::Value * > IndexDomain;
  typedef std::vector< IndexDomain >     IndexDomainVector;


  llvm::Value * datalayout( JitDeviceLayout lay , IndexDomainVector a );
  llvm::Value * datalayout_stack(IndexDomainVector a);

} // namespace QDP

#endif
