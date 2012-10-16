// -*- C++ -*-


#ifndef QDP_WORDJIT_H
#define QDP_WORDJIT_H



namespace QDP {


  template<class T>
  class WordJIT
  {
  public:
    //! Size (in number of registers) of the underlying object
    enum {Size_t = 1};


    //! View of an object from global state space
    WordJIT(Jit& j , int r_addr_ , int offset_full_ , int offset_level_ ) : 
      jit(j), 
      r_addr(r_addr_),
      offset_full(offset_full_),
      offset_level(offset_level_) , global_state(true) {
      std::cout << "WordJIT() global view   " << (void*)this << " " << (void*)&j << "\n";
    }

    //! New space 
    WordJIT(Jit& j ) : jit(j), global_state(false) {
      int tmp;
      mapReg.insert( std::make_pair( JitRegType<T>::Val_t , tmp = jit.getRegs( JitRegType<T>::Val_t , 1 ) ) );
      std::cout << "WordJIT(Jit& func_ ) new space   regName = " << jit.getName(tmp) << " " << (void*)this << " " << (void*)&jit <<  "\n";
    }


    template <class T1>
    WordJIT& assign(const WordJIT<T1>& s1) {
      if (global_state) {
	std::cout << " WordJIT& assign global " << s1.mapReg.size() << "\n";
	jit.asm_st( r_addr , offset_level * WordSize<T>::Size , s1.getReg( JitRegType<T>::Val_t ) );
      }
      else {
	std::cout << " +++++++++++++++++++++++++ WordJIT& assign reg \n";
	jit.asm_mov( getReg( JitRegType<T>::Val_t ) , s1.getReg( JitRegType<T>::Val_t ) );
      }
      std::cout << " WordJIT& assign finished \n";
      return *this;
    }

    //---------------------------------------------------------
    template <class T1>
    WordJIT& operator=(const WordJIT<T1>& s1) {
      std::cout << " WordJIT& op= \n";
      return assign(s1);
    }

    WordJIT& operator=(const WordJIT& s1) {
      std::cout << " WordJIT& op=copy \n";
      return assign(s1);
    }

    template<class T1>
    WordJIT& operator+=(const WordJIT<T1>& rhs) {
      *this = *this + rhs;
      return *this;
    }

    template<class T1>
    WordJIT& operator-=(const WordJIT<T1>& rhs) {
      *this = *this - rhs;
      return *this;
    }


    int getReg( Jit::RegType type ) const {
      std::cout << "getReg type=" << type 
      		<< "  mapReg.count(type)=" << mapReg.count(type) 
      		<< "  mapReg.size()=" << mapReg.size() << "\n";
      if (mapReg.count(type) > 0) {
	// We already have the value in a register of the type requested
	std::cout << jit.getName(mapReg.at(type)) << "\n";
	return mapReg.at(type);
      } else {
	if (mapReg.size() > 0) {
	  // SANITY
	  if (mapReg.size() > 1) {
	    std::cout << "getReg: We already have the value in 2 different types. Now a 3rd one ??\n";
	    exit(1);
	  }
	  // We have the value in a register, but not with the requested type 
	  std::cout << "We have the value in a register, but not with the requested type\n";
	  MapRegType::iterator loaded = mapReg.begin();
	  Jit::RegType loadedType = loaded->first;
	  int loadedId = loaded->second;
	  mapReg.insert( std::make_pair( type , jit.getRegs( type , 1 ) ) );
	  jit.asm_cvt( mapReg.at(type) , loadedId );
	  return mapReg.at(type);
	} else {
	  // We don't have the value in a register. Need to load it.
	  std::cout << "We don't have the value in a register. Need to load it " << (void*)this << " " << (void*)&jit << "\n";
	  Jit::RegType myType = JitRegType<T>::Val_t;
	  mapReg.insert( std::make_pair( myType , jit.getRegs( JitRegType<T>::Val_t , 1 ) ) );
	  jit.asm_ld( mapReg.at( myType ) , r_addr , offset_level * WordSize<T>::Size );
	  return getReg(type);
	}
      }
    }
#if 0
    WordJIT(const WordJIT& a):
      global_state(a.global_state),
      jit(a.jit),
      r_addr(a.r_addr),
      offset_full(a.offset_full),
      offset_level(a.offset_level)
    {
      mapReg.insert(a.mapReg.begin(), a.mapReg.end());
      std::cout << __PRETTY_FUNCTION__ << " mapReg.size()=" << mapReg.size() << "\n";
    }
#endif

    Jit& func() const {return jit;}

  public:
    typedef std::map< Jit::RegType , int > MapRegType;
    bool global_state;
    Jit&  jit;
    mutable MapRegType mapReg;
    int r_addr;
    int offset_full;
    int offset_level;
  };


  template<>
  template<typename T1>
  WordJIT<bool>& WordJIT<bool>::assign(const WordJIT<T1>& s1) 
  {
    std::cout << "********************** WordJIT SPECIAL ASSIGN\n";
    if (global_state) {
      std::cout << " WordJIT& assign global " << s1.mapReg.size() << "\n";

      int s1_u32 = jit.getRegs( Jit::u32 , 1 );
      jit.asm_pred_to_01( s1_u32 , s1.getReg( JitRegType<bool>::Val_t )  );
      int s1_u8 = jit.getRegs( Jit::u8 , 1 );
      jit.asm_cvt( s1_u8 , s1_u32 );
      
      jit.asm_st( r_addr , offset_level * WordSize<bool>::Size , s1_u8 );
    }
    else {
      std::cout << " +++++++++++++++++++++++++ WordJIT& assign reg \n";
      jit.asm_mov( getReg( JitRegType<bool>::Val_t ) , s1.getReg( JitRegType<bool>::Val_t ) );
    }
    return *this;
  }

  template<>
  int WordJIT<bool>::getReg( Jit::RegType type ) const;




  template<class T> 
  struct WordType<WordJIT<T> >
  {
    typedef T  Type_t;
  };










// template<class T1, class T2>
// struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAnd > {
//   typedef WordJIT<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
// };




  // ---------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------




// Default unary(WordJIT) -> WordJIT
template<class T1, class Op>
struct UnaryReturn<WordJIT<T1>, Op> {
  typedef WordJIT<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


// Default binary(WordJIT,WordJIT) -> WordJIT
template<class T1, class T2, class Op>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, Op> {
  typedef WordJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<WordJIT<T2>, OpCast<T1> > {
  typedef WordJIT<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

// Assignment is different
template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAssign > {
  typedef WordJIT<T1> &Type_t;
};

template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAddAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpSubtractAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpMultiplyAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpDivideAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpModAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseOrAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseAndAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseXorAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLeftShiftAssign > {
  typedef WordJIT<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpRightShiftAssign > {
  typedef WordJIT<T1> &Type_t;
};
 



//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primscalar */
/*! @{ */

// Primitive Scalars

// ! WordJIT
template<class T>
struct UnaryReturn<WordJIT<T>, OpNot > {
  typedef WordJIT<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<WordJIT<T1>, OpNot>::Type_t
operator!(const WordJIT<T1>& l)
{
  typedef typename UnaryReturn<WordJIT<T1>, OpNot>::Type_t  Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());

  tmp.func().asm_not( tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<WT>::Val_t ) );
  return tmp;
}

// + WordJIT
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, OpUnaryPlus>::Type_t
operator+(const WordJIT<T1>& l)
{
  return l;
}

// - WordJIT
  template<class T1>
  inline typename UnaryReturn<WordJIT<T1>, OpUnaryMinus>::Type_t
  operator-(const WordJIT<T1>& l)
  {
    typedef typename UnaryReturn<WordJIT<T1>, OpUnaryMinus>::Type_t  Ret_t;
    typedef typename WordType<Ret_t>::Type_t WT;
    Ret_t tmp(l.func());

    tmp.func().asm_neg( tmp.getReg( JitRegType<WT>::Val_t ) , 
			l.getReg( JitRegType<WT>::Val_t ) );
    return tmp;
  }



// WordJIT + WordJIT
  template<class T1, class T2>
  inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAdd>::Type_t
  operator+(const WordJIT<T1>& l, const WordJIT<T2>& r)
  {
    typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAdd>::Type_t Ret_t;
    typedef typename WordType<Ret_t>::Type_t WT;
    Ret_t tmp(l.func());
    tmp.func().asm_add( tmp.getReg( JitRegType<WT>::Val_t ) , 
			l.getReg( JitRegType<WT>::Val_t ) , 
			r.getReg( JitRegType<WT>::Val_t ) );
    return tmp;
  }


// WordJIT - WordJIT
  template<class T1, class T2>
  inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpSubtract>::Type_t
  operator-(const WordJIT<T1>& l, const WordJIT<T2>& r)
  {
    typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpSubtract>::Type_t Ret_t;
    typedef typename WordType<Ret_t>::Type_t WT;
    Ret_t tmp(l.func());
    tmp.func().asm_sub( tmp.getReg( JitRegType<WT>::Val_t ) , 
			l.getReg( JitRegType<WT>::Val_t ) , 
			r.getReg( JitRegType<WT>::Val_t ) );
    return tmp;
  }


// WordJIT * WordJIT
  template<class T1, class T2>
  inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpMultiply>::Type_t
  operator*(const WordJIT<T1>& l, const WordJIT<T2>& r)
  {
    typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpMultiply>::Type_t Ret_t;
    typedef typename WordType<Ret_t>::Type_t WT;
    Ret_t tmp(l.func());
    tmp.func().asm_mul( tmp.getReg( JitRegType<WT>::Val_t ) , 
			l.getReg( JitRegType<WT>::Val_t ) , 
			r.getReg( JitRegType<WT>::Val_t ) );
    std::cout << " tmp=" << tmp.mapReg.size() << "\n";
    return tmp;
  }

// Optimized  adj(PMatrix)*PMatrix
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAdjMultiply>::Type_t
adjMultiply(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  return l * r;
}

// Optimized  PMatrix*adj(PMatrix)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpMultiplyAdj>::Type_t
multiplyAdj(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  return l * r;
}

// Optimized  adj(PMatrix)*adj(PMatrix)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  return l * r;
}

// WordJIT / WordJIT
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpDivide>::Type_t
operator/(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpDivide>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_div( tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<WT>::Val_t ) , 
		      r.getReg( JitRegType<WT>::Val_t ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
}


// WordJIT << WordJIT
template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLeftShift > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLeftShift>::Type_t
operator<<(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  QDP_error_exit("operator<< not implemented");
}

// WordJIT >> WordJIT
template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpRightShift > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpRightShift>::Type_t
operator>>(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  QDP_error_exit("operator>> not implemented");
}

// WordJIT % WordJIT
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpMod>::Type_t
operator%(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  QDP_error_exit("operator% not implemented");
}

// WordJIT ^ WordJIT
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseXor>::Type_t
operator^(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  QDP_error_exit("operator^ not implemented");
}

// WordJIT & WordJIT
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseAnd>::Type_t
operator&(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  QDP_error_exit("operator& not implemented");
}

// WordJIT | WordJIT
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpBitwiseOr>::Type_t
operator|(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  QDP_error_exit("operator| not implemented");
}


// Comparisons
template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLT > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLT>::Type_t
operator<(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLT>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_cmp( Jit::lt, 
		      tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<T1>::Val_t ) , 
		      r.getReg( JitRegType<T1>::Val_t ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
}


template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLE > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLE>::Type_t
operator<=(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpLE>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_cmp( Jit::le, 
		      tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<T1>::Val_t ) , 
		      r.getReg( JitRegType<T1>::Val_t ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
}


template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGT > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGT>::Type_t
operator>(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGT>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_cmp( Jit::gt, 
		      tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<T1>::Val_t ) , 
		      r.getReg( JitRegType<T1>::Val_t ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
  //  return l.elem() > r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGE > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGE>::Type_t
operator>=(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpGE>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_cmp( Jit::ge, 
		      tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<T1>::Val_t ) , 
		      r.getReg( JitRegType<T1>::Val_t ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
  //  return l.elem() >= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpEQ > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpEQ>::Type_t
operator==(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpEQ>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_cmp( Jit::eq, 
		      tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<T1>::Val_t ) , 
		      r.getReg( JitRegType<T1>::Val_t ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
  //  return l.elem() == r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpNE > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpNE>::Type_t
operator!=(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpNE>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_cmp( Jit::ne, 
		      tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<T1>::Val_t ) , 
		      r.getReg( JitRegType<T1>::Val_t ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
  //  return l.elem() != r.elem();
}


template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAnd > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAnd>::Type_t
operator&&(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpAnd>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_and( tmp.getReg( JitRegType<WT>::Val_t ) , 
		      l.getReg( JitRegType<WT>::Val_t ), r.getReg( JitRegType<WT>::Val_t ) );
  return tmp;
}



template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpOr > {
  typedef WordJIT<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpOr>::Type_t
operator||(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  typedef typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, OpOr>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(l.func());
  tmp.func().asm_or( tmp.getReg( JitRegType<WT>::Val_t ) , 
		     l.getReg( JitRegType<WT>::Val_t ), r.getReg( JitRegType<WT>::Val_t ) );
  return tmp;
  //  return l.elem() || r.elem();
}


//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnAdjoint>::Type_t
adj(const WordJIT<T1>& s1)
{
  return s1;
}


// Conjugate
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnConjugate>::Type_t
conj(const WordJIT<T1>& s1)
{
  return s1;
}


// Transpose
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnTranspose>::Type_t
transpose(const WordJIT<T1>& s1)
{
  return s1;
}


// TRACE
// trace = Trace(source1)
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnTrace>::Type_t
trace(const WordJIT<T1>& s1)
{
  return s1;
}


// trace = Re(Trace(source1))
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnRealTrace>::Type_t
realTrace(const WordJIT<T1>& s1)
{
  return s1;
}


// trace = Im(Trace(source1))
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnImagTrace>::Type_t
imagTrace(const WordJIT<T1>& s1)
{
  return s1;
}


// trace = colorTrace(source1)
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnTraceColor>::Type_t
traceColor(const WordJIT<T1>& s1)
{
  return s1;
}


//! WordJIT = traceSpin(WordJIT)
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnTraceSpin>::Type_t
traceSpin(const WordJIT<T1>& s1)
{
  return s1;
}

//! WordJIT = transposeSpin(WordJIT)
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnTransposeSpin>::Type_t
transposeSpin(const WordJIT<T1>& s1)
{
  return s1;
}

//! WordJIT = trace(WordJIT * WordJIT)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnTraceMultiply>::Type_t
traceMultiply(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  return l * r;
}

//! WordJIT = traceColor(WordJIT * WordJIT)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnTraceColorMultiply>::Type_t
traceColorMultiply(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  return l * r;
}

//! WordJIT = traceSpin(WordJIT * WordJIT)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnTraceSpinMultiply>::Type_t
traceSpinMultiply(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  return l * r;
}

//! WordJIT = traceSpin(outerProduct(WordJIT, WordJIT))
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnTraceSpinOuterProduct>::Type_t
traceSpinOuterProduct(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  return l * r;
}

//! WordJIT = outerProduct(WordJIT, WordJIT)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnOuterProduct>::Type_t
outerProduct(const WordJIT<T1>& l, const WordJIT<T2>& r)
{
  return l * r;
}


//! WordJIT = Re(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnReal>::Type_t
real(const WordJIT<T>& s1)
{
  return s1;
}


// WordJIT = Im(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnImag>::Type_t
imag(const WordJIT<T>& s1)
{
  return s1;
}


// ArcCos
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnArcCos>::Type_t
acos(const WordJIT<T1>& s1)
{
  return acos(s1.elem());
}

// ArcSin
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnArcSin>::Type_t
asin(const WordJIT<T1>& s1)
{
  return asin(s1.elem());
}

// ArcTan
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnArcTan>::Type_t
atan(const WordJIT<T1>& s1)
{
  return atan(s1.elem());
}

// Ceil(ing)
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnCeil>::Type_t
ceil(const WordJIT<T1>& s1)
{
  return ceil(s1.elem());
}

// Cos
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnCos>::Type_t
cos(const WordJIT<T1>& s1)
{
  typedef typename UnaryReturn<WordJIT<T1>, FnCos>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(s1.func());
  tmp.func().asm_cos( tmp.getReg( Jit::f32 ) , 
		      s1.getReg( Jit::f32 ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
}

// Cosh
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnHypCos>::Type_t
cosh(const WordJIT<T1>& s1)
{
  return cosh(s1.elem());
}

// Exp
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnExp>::Type_t
exp(const WordJIT<T1>& s1)
{
  return exp(s1.elem());
}

// Fabs
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnFabs>::Type_t
fabs(const WordJIT<T1>& s1)
{
  return fabs(s1.elem());
}

// Floor
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnFloor>::Type_t
floor(const WordJIT<T1>& s1)
{
  return floor(s1.elem());
}

// Log
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnLog>::Type_t
log(const WordJIT<T1>& s1)
{
  return log(s1.elem());
}

// Log10
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnLog10>::Type_t
log10(const WordJIT<T1>& s1)
{
  return log10(s1.elem());
}

// Sin
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnSin>::Type_t
sin(const WordJIT<T1>& s1)
{
  typedef typename UnaryReturn<WordJIT<T1>, FnSin>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(s1.func());
  tmp.func().asm_sin( tmp.getReg( Jit::f32 ) , 
		      s1.getReg( Jit::f32 ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
  //  return sin(s1.elem());
}

// Sinh
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnHypSin>::Type_t
sinh(const WordJIT<T1>& s1)
{
  return sinh(s1.elem());
}

// Sqrt
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnSqrt>::Type_t
sqrt(const WordJIT<T1>& s1)
{
  typedef typename UnaryReturn<WordJIT<T1>, FnSin>::Type_t Ret_t;
  typedef typename WordType<Ret_t>::Type_t WT;
  Ret_t tmp(s1.func());
  tmp.func().asm_sqrt( tmp.getReg( JitRegType<WT>::Val_t ) , 
		       s1.getReg( JitRegType<WT>::Val_t ) );
  std::cout << " tmp=" << tmp.mapReg.size() << "\n";
  return tmp;
  //return sqrt(s1.elem());
}

// Tan
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnTan>::Type_t
tan(const WordJIT<T1>& s1)
{
  return tan(s1.elem());
}

// Tanh
template<class T1>
inline typename UnaryReturn<WordJIT<T1>, FnHypTan>::Type_t
tanh(const WordJIT<T1>& s1)
{
  return tanh(s1.elem());
}



//! WordJIT<T> = pow(WordJIT<T> , WordJIT<T>)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnPow>::Type_t
pow(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return pow(s1.elem(), s2.elem());
}

//! WordJIT<T> = atan2(WordJIT<T> , WordJIT<T>)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnArcTan2>::Type_t
atan2(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return atan2(s1.elem(), s2.elem());
}


//! WordJIT<T> = (WordJIT<T> , WordJIT<T>)
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnCmplx>::Type_t
cmplx(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return cmplx(s1.elem(), s2.elem());
}



// Global Functions
// WordJIT = i * WordJIT
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnTimesI>::Type_t
timesI(const WordJIT<T>& s1)
{
  return timesI(s1.elem());
}

// WordJIT = -i * WordJIT
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnTimesMinusI>::Type_t
timesMinusI(const WordJIT<T>& s1)
{
  return timesMinusI(s1.elem());
}


//! dest [float type] = source [seed type]
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSeedToFloat>::Type_t
seedToFloat(const WordJIT<T>& s1)
{
  return seedToFloat(s1.elem());
}


//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnGetSite>::Type_t
getSite(const WordJIT<T>& s1, int innersite)
{
  return getSite(s1.elem(), innersite);
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnPeekColorVector>::Type_t
peekColor(const WordJIT<T>& l, int row)
{
  return peekColor(l.elem(),row);
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnPeekColorMatrix>::Type_t
peekColor(const WordJIT<T>& l, int row, int col)
{
  return peekColor(l.elem(),row,col);
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnPeekSpinVector>::Type_t
peekSpin(const WordJIT<T>& l, int row)
{
  return peekSpin(l.elem(),row);
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnPeekSpinMatrix>::Type_t
peekSpin(const WordJIT<T>& l, int row, int col)
{
  return peekSpin(l.elem(),row,col);
}


//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2>
inline WordJIT<T1>&
pokeColor(WordJIT<T1>& l, const WordJIT<T2>& r, int row)
{
  pokeColor(l.elem(),r.elem(),row);
  return l;
}

//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2>
inline WordJIT<T1>&
pokeColor(WordJIT<T1>& l, const WordJIT<T2>& r, int row, int col)
{
  pokeColor(l.elem(),r.elem(),row,col);
  return l;
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2>
inline WordJIT<T1>&
pokeSpin(WordJIT<T1>& l, const WordJIT<T2>& r, int row)
{
  pokeSpin(l.elem(),r.elem(),row);
  return l;
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2>
inline WordJIT<T1>&
pokeSpin(WordJIT<T1>& l, const WordJIT<T2>& r, int row, int col)
{
  pokeSpin(l.elem(),r.elem(),row,col);
  return l;
}


//-----------------------------------------------------------------------------
//! WordJIT = Gamma<N,m> * WordJIT
template<class T2, int N, int m>
inline typename BinaryReturn<GammaConst<N,m>, WordJIT<T2>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<N,m>& l, const WordJIT<T2>& r)
{
  return l * r.elem();
}

//! WordJIT = WordJIT * Gamma<N,m>
template<class T2, int N, int m>
inline typename BinaryReturn<WordJIT<T2>, GammaConst<N,m>, OpGammaConstMultiply>::Type_t
operator*(const WordJIT<T2>& l, const GammaConst<N,m>& r)
{
  return l.elem() * r;
}

//-----------------------------------------------------------------------------
//! WordJIT = SpinProject(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinProjectDir0Minus>::Type_t
spinProjectDir0Minus(const WordJIT<T>& s1)
{
  return spinProjectDir0Minus(s1.elem());
}

//! WordJIT = SpinReconstruct(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinReconstructDir0Minus>::Type_t
spinReconstructDir0Minus(const WordJIT<T>& s1)
{
  return spinReconstructDir0Minus(s1.elem());
}


//! WordJIT = SpinProject(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinProjectDir1Minus>::Type_t
spinProjectDir1Minus(const WordJIT<T>& s1)
{
  return spinProjectDir1Minus(s1.elem());
}

//! WordJIT = SpinReconstruct(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinReconstructDir1Minus>::Type_t
spinReconstructDir1Minus(const WordJIT<T>& s1)
{
  return spinReconstructDir1Minus(s1.elem());
}


//! WordJIT = SpinProject(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinProjectDir2Minus>::Type_t
spinProjectDir2Minus(const WordJIT<T>& s1)
{
  return spinProjectDir2Minus(s1.elem());
}

//! WordJIT = SpinReconstruct(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinReconstructDir2Minus>::Type_t
spinReconstructDir2Minus(const WordJIT<T>& s1)
{
  return spinReconstructDir2Minus(s1.elem());
}


//! WordJIT = SpinProject(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinProjectDir3Minus>::Type_t
spinProjectDir3Minus(const WordJIT<T>& s1)
{
  return spinProjectDir3Minus(s1.elem());
}

//! WordJIT = SpinReconstruct(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinReconstructDir3Minus>::Type_t
spinReconstructDir3Minus(const WordJIT<T>& s1)
{
  return spinReconstructDir3Minus(s1.elem());
}


//! WordJIT = SpinProject(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinProjectDir0Plus>::Type_t
spinProjectDir0Plus(const WordJIT<T>& s1)
{
  return spinProjectDir0Plus(s1.elem());
}

//! WordJIT = SpinReconstruct(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinReconstructDir0Plus>::Type_t
spinReconstructDir0Plus(const WordJIT<T>& s1)
{
  return spinReconstructDir0Plus(s1.elem());
}


//! WordJIT = SpinProject(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinProjectDir1Plus>::Type_t
spinProjectDir1Plus(const WordJIT<T>& s1)
{
  return spinProjectDir1Plus(s1.elem());
}

//! WordJIT = SpinReconstruct(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinReconstructDir1Plus>::Type_t
spinReconstructDir1Plus(const WordJIT<T>& s1)
{
  return spinReconstructDir1Plus(s1.elem());
}


//! WordJIT = SpinProject(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinProjectDir2Plus>::Type_t
spinProjectDir2Plus(const WordJIT<T>& s1)
{
  return spinProjectDir2Plus(s1.elem());
}

//! WordJIT = SpinReconstruct(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinReconstructDir2Plus>::Type_t
spinReconstructDir2Plus(const WordJIT<T>& s1)
{
  return spinReconstructDir2Plus(s1.elem());
}


//! WordJIT = SpinProject(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinProjectDir3Plus>::Type_t
spinProjectDir3Plus(const WordJIT<T>& s1)
{
  return spinProjectDir3Plus(s1.elem());
}

//! WordJIT = SpinReconstruct(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSpinReconstructDir3Plus>::Type_t
spinReconstructDir3Plus(const WordJIT<T>& s1)
{
  return spinReconstructDir3Plus(s1.elem());
}

//-----------------------------------------------------------------------------
//! WordJIT = chiralProjectPlus(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(const WordJIT<T>& s1)
{
  return chiralProjectPlus(s1.elem());
}

//! WordJIT = chiralProjectMinus(WordJIT)
template<class T>
inline typename UnaryReturn<WordJIT<T>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(const WordJIT<T>& s1)
{
  return chiralProjectMinus(s1.elem());
}


//-----------------------------------------------------------------------------
// quark propagator contraction
template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnQuarkContract13>::Type_t
quarkContract13(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return quarkContract13(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnQuarkContract14>::Type_t
quarkContract14(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return quarkContract14(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnQuarkContract23>::Type_t
quarkContract23(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return quarkContract23(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnQuarkContract24>::Type_t
quarkContract24(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return quarkContract24(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnQuarkContract12>::Type_t
quarkContract12(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return quarkContract12(s1.elem(), s2.elem());
}

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnQuarkContract34>::Type_t
quarkContract34(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return quarkContract34(s1.elem(), s2.elem());
}


//-----------------------------------------------------------------------------
// Contraction for color matrices
// colorContract 
//! dest  = colorContract(Qprop1,Qprop2,Qprop3)
/*!
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<WordJIT<T1>, WordJIT<T2>, WordJIT<T3>, FnColorContract> {
  typedef WordJIT<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<WordJIT<T1>, WordJIT<T2>, WordJIT<T3>, FnColorContract>::Type_t
colorContract(const WordJIT<T1>& s1, const WordJIT<T2>& s2, const WordJIT<T3>& s3)
{
  return colorContract(s1.elem(), s2.elem(), s3.elem());
}


//-----------------------------------------------------------------------------
// Contraction of two colorvectors
//! dest  = colorVectorContract(Qvec1,Qvec2)
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnColorVectorContract> {
  typedef WordJIT<typename BinaryReturn<T1, T2, FnColorVectorContract>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnColorVectorContract>::Type_t
colorVectorContract(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return colorVectorContract(s1.elem(), s2.elem());
}



//-----------------------------------------------------------------------------
// Cross product for color vectors
//! dest  = colorCrossProduct(Qvec1,Qvec2)
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnColorCrossProduct> {
  typedef WordJIT<typename BinaryReturn<T1, T2, FnColorCrossProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnColorCrossProduct>::Type_t
colorCrossProduct(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return colorCrossProduct(s1.elem(), s2.elem());
}



//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline void 
copymask(WordJIT<T>& d, const WordJIT<T1>& mask, const WordJIT<T>& s1) 
{
  copymask(d.elem(),mask.elem(),s1.elem());
}

//! dest  = random  
template<class T, class T1, class T2>
inline void
fill_random(WordJIT<T>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  fill_random(d.elem(), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian  
template<class T>
inline void
fill_gaussian(WordJIT<T>& d, WordJIT<T>& r1, WordJIT<T>& r2)
{
  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}


#if 1
// Global sum over site indices only
template<class T>
struct UnaryReturn<WordJIT<T>, FnSum > {
  typedef WordJIT<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<WordJIT<T>, FnSum>::Type_t
sum(const WordJIT<T>& s1)
{
  return sum(s1.elem());
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T>
struct UnaryReturn<WordJIT<T>, FnNorm2 > {
  typedef WordJIT<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<WordJIT<T>, FnLocalNorm2 > {
  typedef WordJIT<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<WordJIT<T>, FnLocalNorm2>::Type_t
localNorm2(const WordJIT<T>& s1)
{
  return localNorm2(s1.elem());
}

// Global max
template<class T>
struct UnaryReturn<WordJIT<T>, FnGlobalMax> {
  typedef WordJIT<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<WordJIT<T>, FnGlobalMax>::Type_t
globalMax(const WordJIT<T>& s1)
{
  return globalMax(s1.elem());
}


// Global min
template<class T>
struct UnaryReturn<WordJIT<T>, FnGlobalMin> {
  typedef WordJIT<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<WordJIT<T>, FnGlobalMin>::Type_t
globalMin(const WordJIT<T>& s1)
{
  return globalMin(s1.elem());
}


//! WordJIT<T> = InnerProduct(adj(WordJIT<T1>)*WordJIT<T2>)
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnInnerProduct > {
  typedef WordJIT<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnLocalInnerProduct > {
  typedef WordJIT<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! WordJIT<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnInnerProductReal > {
  typedef WordJIT<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnLocalInnerProductReal > {
  typedef WordJIT<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<WordJIT<T1>, WordJIT<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const WordJIT<T1>& s1, const WordJIT<T2>& s2)
{
  return localInnerProductReal(s1.elem(), s2.elem());
}


//! WordJIT<T> = where(WordJIT, WordJIT, WordJIT)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<WordJIT<T1>, WordJIT<T2>, WordJIT<T3>, FnWhere> {
  typedef WordJIT<typename TrinaryReturn<T1, T2, T3, FnWhere>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<WordJIT<T1>, WordJIT<T2>, WordJIT<T3>, FnWhere>::Type_t
where(const WordJIT<T1>& a, const WordJIT<T2>& b, const WordJIT<T3>& c)
{
  return where(a.elem(), b.elem(), c.elem());
}


//-----------------------------------------------------------------------------
//! QDP Int to int primitive in conversion routine
template<class T> 
inline int 
toInt(const WordJIT<T>& s) 
{
  return toInt(s.elem());
}

//! QDP Real to float primitive in conversion routine
template<class T> 
inline float
toFloat(const WordJIT<T>& s) 
{
  return toFloat(s.elem());
}

//! QDP Double to double primitive in conversion routine
template<class T> 
inline double
toDouble(const WordJIT<T>& s) 
{
  return toDouble(s.elem());
}

//! QDP Boolean to bool primitive in conversion routine
template<class T> 
inline bool
toBool(const WordJIT<T>& s) 
{
  return toBool(s.elem());
}

//! QDP Wordtype to primitive wordtype
template<class T> 
inline typename WordType< WordJIT<T> >::Type_t
toWordType(const WordJIT<T>& s) 
{
  return toWordType(s.elem());
}


//-----------------------------------------------------------------------------
// Other operations
//! dest = 0
template<class T> 
inline void 
zero_rep(WordJIT<T>& dest) 
{
  zero_rep(dest.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
cast_rep(T& d, const WordJIT<T1>& s1)
{
  cast_rep(d, s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
cast_rep(WordJIT<T>& d, const WordJIT<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
copy_site(WordJIT<T>& d, int isite, const WordJIT<T1>& s1)
{
  copy_site(d.elem(), isite, s1.elem());
}

//! gather several inner sites together
template<class T, class T1>
inline void 
gather_sites(WordJIT<T>& d, 
	     const WordJIT<T1>& s0, int i0, 
	     const WordJIT<T1>& s1, int i1,
	     const WordJIT<T1>& s2, int i2,
	     const WordJIT<T1>& s3, int i3)
{
  gather_sites(d.elem(), s0.elem(), i0, s1.elem(), i1, s2.elem(), i2, s3.elem(), i3);
}



} // namespace QDP

#endif
