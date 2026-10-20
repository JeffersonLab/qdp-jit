#ifndef QDP_FORWARD_JIT_H
#define QDP_FORWARD_JIT_H

namespace QDP
{
  enum class JitDataOrder { SiteFast , SiteSlow };

  enum class JitStorage { Native , Half };

  struct JitDataFormat
  {
    JitDataOrder order   = JitDataOrder::SiteFast;
    JitStorage   storage = JitStorage::Native;

    JitDataFormat() = default;
    JitDataFormat( JitDataOrder o , JitStorage s = JitStorage::Native ): order(o), storage(s) {}

    bool half() const { return storage == JitStorage::Half; }

    bool operator==( const JitDataFormat& o ) const { return order == o.order && storage == o.storage; }
    bool operator!=( const JitDataFormat& o ) const { return !( *this == o ); }
  };

  template<class T> struct ShapeAxes;

  template<class Op, class A>
  struct UnaryShapeTransform;

  template<class Op, class LHS, class RHS>
  struct BinaryShapeTransform;

  template<class Op, class A, class B, class C>
  struct TrinaryShapeTransform;


  
}

#endif

