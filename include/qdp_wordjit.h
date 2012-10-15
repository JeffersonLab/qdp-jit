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
	std::cout << " WordJIT& assign reg \n";
	jit.asm_mov( getReg( JitRegType<T>::Val_t ) , s1.getReg( JitRegType<T>::Val_t ) );
      }
      std::cout << " WordJIT& assign finished \n";
      return *this;
    }

    //---------------------------------------------------------
    template <class T1>
    WordJIT& operator=(const WordJIT<T1>& s1) {
      // std::cout << __PRETTY_FUNCTION__ << ": instructions needed?\n" << (void*)this << " " << (void*)&s1 << "\n";
      // Value can be in register file (AddAssign for example)
      // jit.asm_st( r_addr , offset_level * WordSize<T>::Size , s1.getReg( JitRegType<T>::Val_t ) );
      // return *this;
      std::cout << " WordJIT& op= \n";
      return assign(s1);
    }


    WordJIT& operator=(const WordJIT& s1) {
      // std::cout << __PRETTY_FUNCTION__ << ": (assignment op) instructions needed?\n" << (void*)this << " " << (void*)&s1 << "\n";
      // jit.asm_st( r_addr , offset_level * WordSize<T>::Size , s1.getReg( JitRegType<T>::Val_t ) );
      // return *this;
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


    //! Do shallow copies here
    // Not sure?? Maybe deep
    //WordJIT(const WordJIT& a): jit(a.jit), addr(a.addr), off(a.off) {}

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


template<class T> 
struct WordType<WordJIT<T> >
{
  typedef T  Type_t;
};


  // Default binary(WordJIT,WordJIT) -> WordJIT
  template<class T1, class T2, class Op>
  struct BinaryReturn<WordJIT<T1>, WordJIT<T2>, Op> {
    typedef WordJIT<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
  };


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






} // namespace QDP

#endif
