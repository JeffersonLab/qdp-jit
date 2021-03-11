#ifndef QDP_VIEWLEAF
#define QDP_VIEWLEAF

namespace QDP {

template<class T>
struct LeafFunctor<QDPTypeJIT<T,OLatticeJIT<T> >, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const QDPTypeJIT<T,OLatticeJIT<T> > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup( s.elem( v.getLayout() , v.getIndex() ) );
    return reg;
  }
};


template<class T>
struct LeafFunctor<QDPTypeJIT<T,OScalarJIT<T> >, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const QDPTypeJIT<T,OScalarJIT<T> > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup( s.elem() );
    return reg;
  }
};


template<class T>
struct LeafFunctor<OScalarJIT<T>, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT<T> & s, const ViewLeaf& v)
  {

    std::cout << __PRETTY_FUNCTION__ << std::endl;
    
    Type_t reg;
    reg.setup( s.elem() );
    return reg;
  }
};


#if 0
template<class T>
struct LeafFunctor<OLiteralJIT<T>, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OLiteralJIT<T> & s, const ViewLeaf& v)
  {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    
    Type_t reg;
    //reg.setup( s.elem() );
    return reg;
  }
};
#endif
  

#if 1
template<>
struct LeafFunctor<OLiteralJIT< PScalarJIT< PScalarJIT < RScalarJIT < WordJIT <float> > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT < WordJIT <float> > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OLiteralJIT< PScalarJIT< PScalarJIT < RScalarJIT < WordJIT <float> > > > > & s, const ViewLeaf& v)
  {
    std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;

    WordREG<float> a0( s.get_val() );
    RScalarREG< WordREG<float> > a1(a0);
    PScalarREG< RScalarREG< WordREG<float> > > a2(a1);
    //PScalarREG< PScalarREG< RScalarREG< WordREG<float> > > > a3(a2);
    
    Type_t reg( a2 );

    //reg.setup( s.elem() );
    return reg;
  }
};
#endif
  
}

#endif
