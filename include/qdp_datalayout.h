// -*- C++ -*-

#ifndef QDP_DATALAYOUT_H
#define QDP_DATALAYOUT_H


namespace QDP {

  typedef std::pair< int , jit_value_t > IndexDomain;
  typedef std::vector< IndexDomain >     IndexDomainVector;


  jit_value_t datalayout( JitDeviceLayout lay , IndexDomainVector a );
  jit_value_t datalayout_stack(IndexDomainVector a);

} // namespace QDP

#endif
