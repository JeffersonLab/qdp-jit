// -*- C++ -*-

/*! \file
 * \brief Primitive Spin Matrix
 */

#ifndef QDP_PRIMSPINMATJIT_H
#define QDP_PRIMSPINMATJIT_H

namespace QDP {


//-------------------------------------------------------------------------------------
/*! \addtogroup primspinmatrix Spin matrix primitive
 * \ingroup primmatrix
 *
 * Primitive type that transforms like a Spin Matrix
 *
 * @{
 */


//! Primitive Spin Matrix class
template <class T, int N> class PSpinMatrixJIT : public PMatrixJIT<T, N, PSpinMatrixJIT>
{
public:
  PSpinMatrixJIT(){}


  template<class T1>
  PSpinMatrixJIT(const PSpinMatrixREG<T1,N>& a)
  {
    for(int i=0; i < N; ++i)
      for(int j=0; j < N; ++j)
	this->elem(i,j) = a.elem(i,j);
  }

  //! PSpinMatrixJIT = PScalarJIT
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  PSpinMatrixJIT& operator=(const PScalarREG<T1>& rhs)
    {
      for(int i=0; i < N; ++i)
  	for(int j=0; j < N; ++j)
	  if (i == j)
	    this->elem(i,j) = rhs.elem();
	  else
	    zero_rep(this->elem(i,j));
      return *this;
    }

  //! PSpinMatrixJIT = PSpinMatrixJIT
  /*! Set equal to another PSpinMatrixJIT */
  template<class T1>
  inline
  PSpinMatrixJIT& operator=(const PSpinMatrixREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
  	for(int j=0; j < N; ++j)
  	  this->elem(i,j) = rhs.elem(i,j);
      return *this;
    }

  template<class T1>
  inline
  PSpinMatrixJIT& operator+=(const PSpinMatrixREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
  	for(int j=0; j < N; ++j)
  	  this->elem(i,j) += rhs.elem(i,j);

      return *this;
    }

  //! PMatrixJIT -= PMatrixJIT
  template<class T1>
  inline
  PSpinMatrixJIT& operator-=(const PSpinMatrixREG<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
  	for(int j=0; j < N; ++j)
  	  this->elem(i,j) -= rhs.elem(i,j);

      return *this;
    }

  template<class T1>
  inline
  PSpinMatrixJIT& operator+=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	this->elem(i,i) += rhs.elem();

      return *this;
    }

  //! PMatrixJIT -= PScalarJIT
  template<class T1>
  inline
  PSpinMatrixJIT& operator-=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	this->elem(i,i) -= rhs.elem();

      return *this;
    }

  //! PMatrixJIT *= PScalarJIT
  template<class T1>
  inline
  PSpinMatrixJIT& operator*=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  this->elem(i,j) *= rhs.elem();

      return *this;
    }

  //! PMatrixJIT /= PScalarJIT
  template<class T1>
  inline
  PSpinMatrixJIT& operator/=(const PScalarREG<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  this->elem(i,j) /= rhs.elem();

      return *this;
    }



};

/*! @} */   // end of group primspinmatrix





//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

template<class T1, int N>
struct ScalarType< PSpinMatrixJIT<T1,N> >
{
  typedef PSpinMatrixJIT< typename ScalarType<T1>::Type_t,N > Type_t;
};

  
template<class T1, int N>
struct REGType<PSpinMatrixJIT<T1,N> > 
{
  typedef PSpinMatrixREG<typename REGType<T1>::Type_t,N>  Type_t;
};

template<class T1, int N>
struct BASEType<PSpinMatrixJIT<T1,N> > 
{
  typedef PSpinMatrix<typename BASEType<T1>::Type_t,N>  Type_t;
};


// Underlying word type
template<class T1, int N>
struct WordType<PSpinMatrixJIT<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

template<class T1, int N>
struct SinglePrecType<PSpinMatrixJIT<T1, N> >
{
  typedef PSpinMatrixJIT< typename SinglePrecType<T1>::Type_t , N > Type_t;
};

template<class T1, int N>
struct DoublePrecType<PSpinMatrixJIT<T1, N> >
{
  typedef PSpinMatrixJIT< typename DoublePrecType<T1>::Type_t , N > Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PSpinMatrixJIT<T,N> > {
  typedef PScalarJIT<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive into a scalar leaving grid alone
template<class T, int N>
struct PrimitiveScalar<PSpinMatrixJIT<T,N> > {
  typedef PScalarJIT<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N>
struct LatticeScalar<PSpinMatrixJIT<T,N> > {
  typedef PSpinMatrixJIT<typename LatticeScalar<T>::Type_t, N>  Type_t;
};




template<class T1, class T2, int N>
inline PSpinMatrixJIT<T1,N>&
pokeSpin(PSpinMatrixJIT<T1,N>& l, const PScalarREG<T2>& r, llvm::Value* row, llvm::Value* col)
{
  l.getJitElem(row,col) = r.elem();
  return l;
}



/*! @} */   // end of group primspinmatrix

} // namespace QDP

#endif
