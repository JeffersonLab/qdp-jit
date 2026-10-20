#ifndef QDP_SCALARSITE_DEFS2_H
#define QDP_SCALARSITE_DEFS2_H

namespace QDP {
  template<class F>
  Shape special_copy_shape(F my_shape, const Shape& src)
  {
    Shape a = my_shape(Shape(1, 1));
    Shape b = my_shape(Shape(8, 8));
    return Shape(a.ns() == b.ns() ? a.ns() : src.ns(),
                 a.nc() == b.nc() ? a.nc() : src.nc());
  }
}

#define SPECIAL( name , prims , ns , primc , nc , precision )		\
  template<class T>						\
  class OLattice##name: public OLattice<T>			\
  {								\
    static Shape myShape(const Shape& s = Shape::current()) { return Shape( (ns) , (nc) ); }		\
  public:							\
  using OLattice<T>::operator=;					\
  using OLattice<T>::OLattice;						\
  OLattice##name(): OLattice<T>( myShape() ) {}				\
  OLattice##name(const Zero& rhs): OLattice<T>( myShape() , rhs ) {}	\
  OLattice##name( typename WordType<T>::Type_t rhs): OLattice<T>( myShape() , rhs ) {} \
  OLattice##name( const OLattice<T>& rhs): OLattice<T>( special_copy_shape( &myShape , rhs.shape() ) , rhs ) {} \
  };									\
  template<class T>							\
  class OScalar##name: public OScalar<T>					\
  {									\
    static Shape myShape(const Shape& s = Shape::current()) { return Shape( (ns) , (nc) ); }			\
  public:								\
  using OScalar<T>::operator=;						\
  using OScalar<T>::OScalar;						\
  OScalar##name(): OScalar<T>( myShape() ) {}				\
  OScalar##name(const Zero& rhs): OScalar<T>( myShape() , rhs ) {}		\
  OScalar##name( typename WordType<T>::Type_t rhs): OScalar<T>( myShape() , rhs ) {} \
  OScalar##name( const OScalar<T>& rhs): OScalar<T>( special_copy_shape( &myShape , rhs.shape() ) , rhs ) {} \
  };									\
  typedef OLattice##name< P##prims< P##primc< RComplex<Word<precision> > > > > Lattice##name; \
  typedef OScalar##name< P##prims< P##primc< RComplex<Word<precision> > > > > name;

#define SPECIAL_FD( name , prims , ns , primc , nc )		\
  template<class T>						\
  class OLattice##name: public OLattice<T>			\
  {								\
    static Shape myShape(const Shape& s = Shape::current()) { return Shape( (ns) , (nc) ); }		\
  public:							\
  using OLattice<T>::operator=;					\
  using OLattice<T>::OLattice;						\
  OLattice##name(): OLattice<T>( myShape() ) {}				\
  OLattice##name(const Zero& rhs): OLattice<T>( myShape() , rhs ) {}	\
  OLattice##name( typename WordType<T>::Type_t rhs): OLattice<T>( myShape() , rhs ) {} \
  OLattice##name( const OLattice<T>& rhs): OLattice<T>( special_copy_shape( &myShape , rhs.shape() ) , rhs ) {} \
  };									\
  template<class T>							\
  class OScalar##name: public OScalar<T>					\
  {									\
    static Shape myShape(const Shape& s = Shape::current()) { return Shape( (ns) , (nc) ); }			\
  public:								\
  using OScalar<T>::operator=;						\
  using OScalar<T>::OScalar;						\
  OScalar##name(): OScalar<T>( myShape() ) {}				\
  OScalar##name(const Zero& rhs): OScalar<T>( myShape() , rhs ) {}		\
  OScalar##name( typename WordType<T>::Type_t rhs): OScalar<T>( myShape() , rhs ) {} \
  OScalar##name( const OScalar<T>& rhs): OScalar<T>( special_copy_shape( &myShape , rhs.shape() ) , rhs ) {} \
  };									\
  typedef OScalar##name< P##prims< P##primc< RComplex<Word<REAL> > > > > name; \
  typedef OScalar##name< P##prims< P##primc< RComplex<Word<REAL64> > > > > name##D;\
  typedef OScalar##name< P##prims< P##primc< RComplex<Word<REAL32> > > > > name##F;\
  typedef OLattice##name< P##prims< P##primc< RComplex<Word<REAL> > > > > Lattice##name; \
  typedef OLattice##name< P##prims< P##primc< RComplex<Word<REAL64> > > > > Lattice##name##D; \
  typedef OLattice##name< P##prims< P##primc< RComplex<Word<REAL32> > > > > Lattice##name##F;


#define SPECIAL_SCALAR( name , prims , ns , primc , nc , precision )	\
  template<class T>							\
  class OScalar##name: public OScalar<T>				\
  {									\
    static Shape myShape(const Shape& s = Shape::current()) { return Shape( (ns) , (nc) ); }			\
  public:								\
  using OScalar<T>::operator=;						\
  using OScalar<T>::OScalar;						\
  OScalar##name(): OScalar<T>( myShape() ) {}				\
  OScalar##name(const Zero& rhs): OScalar<T>( myShape() , rhs ) {}		\
  OScalar##name( typename WordType<T>::Type_t rhs): OScalar<T>( myShape() , rhs ) {} \
  OScalar##name( const OScalar<T>& rhs): OScalar<T>( special_copy_shape( &myShape , rhs.shape() ) , rhs ) {} \
  };									\
  typedef OScalar##name< P##prims< P##primc< RComplex<Word<precision> > > > > name;


#define SPECIAL_LATTICE( name , prims , ns , primc , nc , precision )	\
  template<class T>						\
  class OLattice##name: public OLattice<T>			\
  {								\
    static Shape myShape(const Shape& s = Shape::current()) { return Shape( (ns) , (nc) ); }		\
  public:							\
  using OLattice<T>::operator=;					\
  using OLattice<T>::OLattice;						\
  OLattice##name(): OLattice<T>( myShape() ) {}				\
  OLattice##name(const Zero& rhs): OLattice<T>( myShape() , rhs ) {}	\
  OLattice##name( typename WordType<T>::Type_t rhs): OLattice<T>( myShape() , rhs ) {} \
  OLattice##name( const OLattice<T>& rhs): OLattice<T>( special_copy_shape( &myShape , rhs.shape() ) , rhs ) {} \
  };									\
  typedef OLattice##name< P##prims< P##primc< RComplex<Word<precision> > > > > Lattice##name;

#define SPECIAL_LATTICE_FD( name , prims , ns , primc , nc )	\
  template<class T>						\
  class OLattice##name: public OLattice<T>			\
  {								\
    static Shape myShape(const Shape& s = Shape::current()) { return Shape( (ns) , (nc) ); }		\
  public:							\
  using OLattice<T>::operator=;					\
  using OLattice<T>::OLattice;						\
  OLattice##name(): OLattice<T>( myShape() ) {}				\
  OLattice##name(const Zero& rhs): OLattice<T>( myShape() , rhs ) {}	\
  OLattice##name( typename WordType<T>::Type_t rhs): OLattice<T>( myShape() , rhs ) {} \
  OLattice##name( const OLattice<T>& rhs): OLattice<T>( special_copy_shape( &myShape , rhs.shape() ) , rhs ) {} \
  };									\
  typedef OLattice##name< P##prims< P##primc< RComplex<Word<REAL>   > > > > Lattice##name; \
  typedef OLattice##name< P##prims< P##primc< RComplex<Word<REAL64> > > > > Lattice##name##D; \
  typedef OLattice##name< P##prims< P##primc< RComplex<Word<REAL32> > > > > Lattice##name##F;


namespace QDP {
  SPECIAL(HalfFermion,SpinVector,s.ns()>>1,ColorVector,s.nc(),REAL);
  //SPECIAL(StaggeredFermion,SpinVector,1,ColorVector,s.nc(),REAL);
  //SPECIAL(StaggeredPropagator,SpinMatrix,1,ColorMatrix,s.nc(),REAL);
  SPECIAL(DiracFermion,SpinVector,4,ColorVector,s.nc(),REAL);

  //SPECIAL_SCALAR(DStaggeredPropagator,SpinMatrix,1,ColorMatrix,s.nc(),DOUBLE);
  //SPECIAL_SCALAR(StaggeredPropagatorF,SpinMatrix,1,ColorMatrix,s.nc(),REAL32);

  SPECIAL_LATTICE(DiracPropagator,SpinMatrix,4,ColorMatrix,s.nc(),REAL);
  SPECIAL_LATTICE(DiracFermionF,SpinVector,4,ColorVector,s.nc(),REAL32);
  SPECIAL_LATTICE(DiracFermionF3,SpinVector,4,ColorVector,3,REAL32);
  SPECIAL_LATTICE(HalfFermionF,SpinVector,s.ns()>>1,ColorVector,s.nc(),REAL32);
  SPECIAL_LATTICE(FermionF3,SpinVector,s.ns(),ColorVector,3,REAL32);
  SPECIAL_LATTICE(ColorMatrixF3,Scalar,s.ns(),ColorMatrix,3,REAL32);
  SPECIAL_LATTICE(DiracFermionD,SpinVector,4,ColorVector,s.nc(),REAL64);
  SPECIAL_LATTICE(DiracFermionD3,SpinVector,4,ColorVector,3,REAL64);

  SPECIAL_LATTICE(FermionD3,SpinVector,s.ns(),ColorVector,3,REAL64);
  SPECIAL_LATTICE(HalfFermionD,SpinVector,s.ns()>>1,ColorVector,s.nc(),REAL64);
  SPECIAL_LATTICE(ColorMatrixD3,Scalar,s.ns(),ColorMatrix,3,REAL64);

  SPECIAL_FD(StaggeredFermion,SpinVector,1,ColorVector,s.nc());
  SPECIAL_FD(StaggeredPropagator,SpinMatrix,1,ColorMatrix,s.nc());
  //SPECIAL_LATTICE_FD(StaggeredFermion,SpinVector,1,ColorVector,s.nc());


  
}

#endif
