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
    WordJIT(Jit& func_ , int r_addr_ , LayoutFunc lf_ ) : function(func_), lf(lf_), r_addr(r_addr_) {
      std::cout << "WordJIT(Jit& func_ , int r_addr_ , const LayoutFunc& lf_ ) global view\n";
    }

    //! New space 
    WordJIT(Jit& func_ ) : function(func_) {
      int tmp;
      mapReg.insert( std::make_pair( JitRegType<T>::Val_t , tmp = function.getRegs( JitRegType<T>::Val_t , 1 ) ) );
      std::cout << "WordJIT(Jit& func_ ) new space   regName = " << function.getName(tmp) << "\n";
    }

    //! Destructor
    ~WordJIT() {}

    //---------------------------------------------------------
    template <class T1>
    WordJIT& operator=(const WordJIT<T1>& s1) {
      std::cout << __PRETTY_FUNCTION__ << ": instructions needed?\n" << (void*)this << " " << (void*)&s1 << "\n";

      // Value can be in register file (AddAssign for example)
      function.asm_st( r_addr , lf.getOffset()*WordSize<T>::Size , s1.getReg( JitRegType<T>::Val_t ) );
      return *this;
    }

    WordJIT& operator=(const WordJIT& s1) {
      std::cout << __PRETTY_FUNCTION__ << ": (assignment op) instructions needed?\n" << (void*)this << " " << (void*)&s1 << "\n";

      // Value can be in register file (AddAssign for example)
      function.asm_st( r_addr , lf.getOffset()*WordSize<T>::Size , s1.getReg( JitRegType<T>::Val_t ) );
      return *this;
    }

    template<class T1>
    WordJIT& operator+=(const WordJIT<T1>& rhs) {
      addRep( *this , *this , rhs );
      assign(*this);
    }

    template<class T1>
    WordJIT& operator-=(const WordJIT<T1>& rhs) {
      subRep( *this , *this , rhs );
      assign(*this);
    }


    //! Do shallow copies here
    // Not sure?? Maybe deep
    //WordJIT(const WordJIT& a): function(a.function), addr(a.addr), off(a.off) {}

    int getReg( Jit::RegType type ) const {
      std::cout << "getReg type=" << type 
		<< "  mapReg.count(type)=" << mapReg.count(type) 
		<< "  mapReg.size()=" << mapReg.size() << "\n";
      if (mapReg.count(type) > 0) {
	// We already have the value in a register of the type requested
	return mapReg.at(type);
      } else {
	if (mapReg.size() > 0) {
	  // SANITY
	  if (mapReg.size() > 1) {
	    std::cout << "getReg: We already have the value in 2 different types. Now a 3rd one ??\n";
	    exit(1);
	  }
	  // We have the value in a register, but not with the requested type 
	  MapRegType::iterator loaded = mapReg.begin();
	  Jit::RegType loadedType = loaded->first;
	  int loadedId = loaded->second;
	  mapReg.insert( std::make_pair( type , function.getRegs( type , 1 ) ) );
	  function.asm_cvt( mapReg.at(type) , loadedId );
	  return mapReg.at(type);
	} else {
	  // We don't have the value in a register. Need to load it.
	  Jit::RegType myType = JitRegType<T>::Val_t;
	  mapReg.insert( std::make_pair( myType , function.getRegs( JitRegType<T>::Val_t , 1 ) ) );
	  function.asm_ld( mapReg.at( myType ) , r_addr , lf.getOffset()*WordSize<T>::Size );
	  return getReg(type);
	}
      }
    }

    WordJIT(const WordJIT& a): function(a.function), mapReg(a.mapReg), r_addr(a.r_addr), lf(a.lf) {
      std::cout << "WordJIT copy c-tor\n";
    }


    Jit& getFunc() const {return function;}


  private:
    typedef std::map< Jit::RegType , int > MapRegType;
    Jit&  function;
    mutable MapRegType mapReg;
    mutable int r_addr;
    LayoutFunc lf;
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

  // Add
  //template<>
  template<class T0,class T1, class T2>
  inline void
  addRep(const WordJIT<T0>& dest, const WordJIT<T1>& l, const WordJIT<T2>& r)
  {
    dest.getFunc().asm_add( dest.getReg( JitRegType<T0>::Val_t ) , 
			    l.getReg( JitRegType<T0>::Val_t ) , 
			    r.getReg( JitRegType<T0>::Val_t ) );
  }

  // Sub
  //template<>
  template<class T0,class T1, class T2>
  inline void
  subRep(const WordJIT<T0>& dest, const WordJIT<T1>& l, const WordJIT<T2>& r)
  {
    dest.getFunc().asm_sub( dest.getReg( JitRegType<T0>::Val_t ) , 
			    l.getReg( JitRegType<T0>::Val_t ) , 
			    r.getReg( JitRegType<T0>::Val_t ) );
  }


  // Multiply
  // I can't use BinaryReturn since this might be mixed precision words
  template<class T0,class T1, class T2>
  inline void
  mulRep(const WordJIT<T0>& dest,const WordJIT<T1>& l, const WordJIT<T2>& r)
  {
    dest.getFunc().asm_mul( dest.getReg( JitRegType<T0>::Val_t ) , 
			    l.getReg( JitRegType<T0>::Val_t ) , 
			    r.getReg( JitRegType<T0>::Val_t ) );
  }


  // FMA
  template<class T0,class T1,class T2,class T3>
  inline void
  fmaRep( const WordJIT<T0>& dest, const WordJIT<T1>& l, const WordJIT<T2>& r, const WordJIT<T3>& add )
  {
    dest.getFunc().asm_fma( dest.getReg( JitRegType<T0>::Val_t ) , 
			    l.getReg( JitRegType<T0>::Val_t ) , 
			    r.getReg( JitRegType<T0>::Val_t ) , 
			    add.getReg( JitRegType<T0>::Val_t ) );
  }

  // neg
  template<class T0,class T1>
  inline void
  negRep( const WordJIT<T0>& dest, const WordJIT<T1>& src )
  {
    dest.getFunc().asm_neg( dest.getReg( JitRegType<T0>::Val_t ) , 
			    src.getReg( JitRegType<T0>::Val_t ) );
  }

  // identity
  template<class T0,class T1>
  inline void
  idRep( const WordJIT<T0>& dest, const WordJIT<T1>& src )
  {
    dest.getFunc().asm_mov( dest.getReg( JitRegType<T0>::Val_t ) , 
			    src.getReg( JitRegType<T0>::Val_t ) );
  }




} // namespace QDP

#endif
