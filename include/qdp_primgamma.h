#ifndef QDP_PRIMGAMMA_H
#define QDP_PRIMGAMMA_H

namespace QDP {


class GammaType
{
  const int m;

public:
  GammaType(int mm) : m(mm) {}

  int elem() const {return m;}

  Shape shape() const { return Shape{4,1}; }
  
private:
  GammaType() = delete;
};



template <>
struct LeafFunctor<GammaType, MGParamLeaf>
{
  typedef GammaType Type_t;
  inline static
  Type_t apply(const GammaType& g, MGParamLeaf& p) 
  {
    Type_t ret( p.next( PlanEntry::Kind::Gamma ).gamma );
    return ret;
  }
};

  

template <>
struct LeafFunctor<GammaType, DummyJitLeaf>
{
  typedef GammaType Type_t;
  inline static
  Type_t apply(const GammaType& g, DummyJitLeaf& p) 
  {
    Type_t ret(0);
    return ret;
  }
};

 

template <>
struct LeafFunctor<GammaType, ShiftPhase1>
{
  typedef int Type_t;
  static int apply(const GammaType &s, ShiftPhase1 &f)
  {
    return 0;
  }
};

  
template <>
struct LeafFunctor<GammaType, ShiftPhase2>
{
  typedef int Type_t;
  static int apply(const GammaType &s, ShiftPhase2 &f)
  {
    return 0;
  }
};

 
template <>
struct LeafFunctor<GammaType, MGViewLeaf>
{
  typedef GammaType Type_t;
  inline static
  Type_t apply(const GammaType& g, MGViewLeaf& v)
  {
    Type_t ret(g.elem());
    return ret;
  }
};

  
 
template<class T2>
struct BinaryReturn<GammaType, PScalar<T2>, OpGammaTypeMultiply> {
  typedef PScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


template<class T2>
struct BinaryReturn<PScalar<T2>, GammaType, OpMultiplyGammaType> {
  typedef PScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

 

class GammaTypeDP
{
  const int m;
public:
  GammaTypeDP(int mm) : m(mm) {}

  int elem() const {return m;}  

  Shape shape() const { return Shape{4,1}; }

private:
  GammaTypeDP() = delete;
};



template <>
struct LeafFunctor<GammaTypeDP, MGParamLeaf>
{
  typedef GammaTypeDP Type_t;
  inline static
  Type_t apply(const GammaTypeDP& g, MGParamLeaf& p) 
  {
    Type_t ret( p.next( PlanEntry::Kind::Gamma ).gamma );
    return ret;
  }
};

  

template <>
struct LeafFunctor<GammaTypeDP, DummyJitLeaf>
{
  typedef GammaTypeDP Type_t;
  inline static
  Type_t apply(const GammaTypeDP& g, DummyJitLeaf& p) 
  {
    Type_t ret(0);
    return ret;
  }
};

  
template <>
struct LeafFunctor<GammaTypeDP, ShiftPhase1>
{
  typedef int Type_t;
  static int apply(const GammaTypeDP &s, ShiftPhase1 &f) { return 0; }
};


template <>  
struct LeafFunctor<GammaTypeDP, ShiftPhase2>
{
  typedef int Type_t;
  static int apply(const GammaTypeDP &s, ShiftPhase2 &f) { return 0; }
};




template <>  
struct LeafFunctor<GammaTypeDP, MGViewLeaf>
{
  typedef GammaTypeDP Type_t;
  inline static
  Type_t apply(const GammaTypeDP& g, MGViewLeaf& v)
  {
    Type_t ret(g.elem());
    return ret;
  }
};

  

template<class T2>
struct BinaryReturn<GammaTypeDP, PScalar<T2>, OpGammaTypeMultiply> {
  typedef PScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<PScalar<T2>, GammaTypeDP, OpMultiplyGammaType> {
  typedef PScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



} // namespace QDP

#endif
