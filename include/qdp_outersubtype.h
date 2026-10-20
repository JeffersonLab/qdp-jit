#ifndef QDP_OUTERSUBTYPE_H
#define QDP_OUTERSUBTYPE_H

namespace QDP {

template<class T> 
class OSubLattice: public QDPSubType<T, OLattice<T> >
{
  OLattice<T>& orig;
  
  Subset* s;

public:
  OSubLattice(OLattice<T>& a, const Subset& ss): orig(a), s(&(const_cast<Subset&>(ss)))  {}

  OLattice<T>& me()
  {
    return orig;
  }
  
  OSubLattice(const OSubLattice& a): QDPSubType<T, OLattice<T> >(), orig(a.orig), s(a.s)
  {
  }


  OSubLattice() = delete;


  inline
  void operator=(const typename WordType<T>::Type_t& rhs)
  {
    this->assign(rhs);
  }

  inline
  void operator=(const Zero& rhs)
  {
    this->assign(rhs);
  }

  template<class T1,class C1>
  inline
  void operator=(const QDPType<T1,C1>& rhs)
  {
    this->assign(rhs);
  }

  template<class T1,class C1>
  inline
  void operator=(const QDPExpr<T1,C1>& rhs)
  {
    this->assign(rhs);
  }


  inline
  void operator=(const OSubLattice& rhs)
  {
    this->assign(rhs);
  }

  const Subset& subset() const {return *s;}

  int getId() const {
    return orig.getId();
  }

  
  LatticeLayout layout() const
  {
    return orig.layout();
  }


  Shape shape() const
  {
    return orig.shape();
  }

  
  JitDataFormat data_format() const
  {
    return orig.data_format();
  }



  
};


template<class T>
struct QDPSubTypeTrait<OScalar<T> > 
{
  typedef OSubScalar<T>  Type_t;
};


template<class T>
struct QDPSubTypeTrait<OLattice<T> > 
{
  typedef OSubLattice<T>  Type_t;
};



template<class T>
struct WordType<OSubScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


template<class T>
struct WordType<OSubLattice<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


template<class T>
struct SinglePrecType<OSubScalar<T> > 
{
  typedef OSubScalar<typename SinglePrecType<T>::Type_t>  Type_t;
};


template<class T>
struct SinglePrecType<OSubLattice<T> > 
{
  typedef OSubLattice<typename SinglePrecType<T>::Type_t>  Type_t;
};


template<class T>
struct DoublePrecType<OSubScalar<T> > 
{
  typedef OSubScalar<typename DoublePrecType<T>::Type_t>  Type_t;
};



template<class T>
struct DoublePrecType<OSubLattice<T> > 
{
  typedef OSubLattice<typename DoublePrecType<T>::Type_t>  Type_t;
};



template<class T> 
struct JITType<OSubLattice<T> >
{
  typedef OSubLatticeJIT<typename JITType<T>::Type_t>  Type_t;
};



template<class T>
struct LeafFunctor<OSubLattice<T>, DummyJitLeaf>
{
  typedef typename JITType< OSubLattice<T> >::Type_t  Type_t;
  inline static
  Type_t apply(const OSubLattice<T>& do_not_use, DummyJitLeaf& p)
  {
    return Type_t();
  }
};
  


template<class T> 
void zero_rep(OScalar<T>& dest, const Subset& s) 
{
  zero_rep(dest.field().elem());
}



template<class T>
void zero_rep(OSubScalar<T> dest) 
{
  zero_rep(dest.field().elem());
}



template<class T>
void random(OSubScalar<T> d);



template<class T>
void gaussian(OSubScalar<T> dd)
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();

  OScalar<T>  r1, r2;

  random(r1(s));
  random(r2(s));

  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}

} // namespace QDP

#endif
