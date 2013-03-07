// -*- C++ -*-
//
// QDP data parallel interface
//


#ifndef QDP_WORDJIT_H
#define QDP_WORDJIT_H



namespace QDP {


  template<class T>
  class WordJIT 
  {
  public:
    enum {Size_t = 1};

    // Default constructing should be possible
    // then there is no need for MPL index when
    // construction a PMatrix<T,N>
    WordJIT() {}

    void setup(jit_function_t func_, jit_value_t r_base_, int full_, int level_ ) {
      func          = func_;
      r_base        = r_base_;
      offset_full   = full_;
      offset_level  = level_;
      setup_m = true;
    }


    template<class T1>
    void operator=(const WordREG<T1>& s1) {
      assert(setup_m);
      std::cout << __PRETTY_FUNCTION__ << "\n";
      jit_ins_store( r_base , offset_level * WordSize<T>::Size , jit_type<T>::value , s1.get_val() );
    }


    jit_function_t get_func() const { assert(setup_m); return func;}
    jit_value_t getBaseReg() const { assert(setup_m); return r_base; }
    int getFull() const { assert(setup_m); return offset_full; }
    int getLevel() const { assert(setup_m); return offset_level; }

  private:
    template<class T1>
    void operator=(const WordJIT<T1>& s1);
    void operator=(const WordJIT& s1);

    jit_function_t func;
    jit_value_t    r_base;
    int offset_full;
    int offset_level;
    bool setup_m=false;
  };


  template<> struct WordSize< WordJIT<float> > { enum { Size = sizeof(float) }; };
  template<> struct WordSize< WordJIT<double> > { enum { Size = sizeof(double) }; };
  template<> struct WordSize< WordJIT<int> > { enum { Size = sizeof(int) }; };
  template<> struct WordSize< WordJIT<bool> > { enum { Size = sizeof(bool) }; };



  template<class T>
  struct REGType< WordJIT<T> >
  {
    typedef WordREG<typename REGType<T>::Type_t>  Type_t;
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


} // namespace QDP

#endif
