// -*- C++ -*-
// $Id: qdp_forward.h,v 1.5 2003-10-17 15:56:23 edwards Exp $

/*! @file
 * @brief Forward declarations for QDP
 */

QDP_BEGIN_NAMESPACE(QDP);

// IO
class TextReader;
class TextWriter;
class NmlReader;
class NmlWriter;
class BinaryReader;
class BinaryWriter;


// Forward declarations
namespace RNG 
{
//  float sranf(Seed&, Seed&, const Seed&);
}

  
// Inner
template<class T> class IScalar;
template<class T, int N> class ILattice;

// Reality
template<class T> class RScalar;
template<class T> class RComplex;

// Primitives
template<class T> class PScalar;
template <class T, int N, template<class,int> class C> class PMatrix;
template <class T, int N, template<class,int> class C> class PVector;
template <class T, int N> class PColorVector;
template <class T, int N> class PSpinVector;
template <class T, int N> class PColorMatrix;
template <class T, int N> class PSpinMatrix;
template <class T, int N> class PDWVector;
template <class T> class PSeed;

template<int N> class GammaType;
template<int N, int m> class GammaConst;


// Outer
template<class T> class OScalar;
template<class T> class OLattice;

// Outer types narrowed to a subset
template<class T, class S> class OSubScalar;
template<class T, class S> class OSubLattice;

// Main type
template<class T, class C> class QDPType;

// Main type narrowed to a subset
template<class T, class C, class S> class QDPSubType;

// Simple scalar trait class
template<class T> struct SimpleScalar;
template<class T> struct InternalScalar;
template<class T> struct LatticeScalar;
template<class T> struct PrimitiveScalar;
template<class T> struct RealScalar;
template<class T> struct WordType;

// Empty leaf functor tag
struct ElemLeaf;

// Used for nearest neighbor shift (a map)
class ArrayBiDirectionalMap;


QDP_END_NAMESPACE();

  
