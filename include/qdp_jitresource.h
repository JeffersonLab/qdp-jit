#ifndef QDP_JITR_H
#define QDP_JITR_H

#include<array>

namespace QDP {


  struct curry_t {
    curry_t(Jit& j,int r,int fu,int le): jit(j),r_addr(r),ful(fu),lev(le) {}
    Jit& jit;
    int r_addr;
    int ful,lev;
  };

  struct newspace_t {
    newspace_t(Jit& j): jit(j) {}
    Jit& jit;
  };



template <int... Is>
struct indices {};

template <int N, int... Is>
struct build_indices
  : build_indices<N-1, N-1, Is...> {};

template <int... Is>
struct build_indices<0, Is...> : indices<Is...> {};








  template<class T,int N>
class JV {
  public:

    enum { ThisSize = N };                 // Size in T's
    enum { Size_t = ThisSize * T::Size_t}; // Size in registers

    ~JV() {}

    JV(newspace_t n) : JV(n, build_indices<N>{}) {}
    template<int... Is>
    JV(newspace_t n, indices<Is...>) : 
      jit(n.jit), F{{(void(Is),n)...}} {}


    JV(newspace_t n,const JV<T,N>* ptr) : JV(n,const_cast<JV<T,N>*>(ptr), build_indices<N>{}) {}
    template<int... Is>
    JV(newspace_t n,JV<T,N>* ptr, indices<Is...>) : 
      jit(n.jit), off_full(ptr->off_full), off_level(ptr->off_level), r_addr(ptr->r_addr), F{{ T( n , &ptr->F[Is] )... }} {}


#if 1
    JV(Jit& j,const typename WordType<T>::Type_t& w) : jit(j), F{{T(j,w)}} 
    {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
    }
    JV(Jit& j,
       const typename WordType<T>::Type_t& re,
       const typename WordType<T>::Type_t& im) : jit(j), F{{T(j,re),T(j,im)}} 
    {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
    }

    // JV(const T& t0,const T& t1) : jit(t0.func()), F{{t0,t1}} {
    //   //std::cout << __PRETTY_FUNCTION__ << "\n";
    // }
    // JV(const T& t0) : jit(t0.func()), F{{t0}} {
    //   //std::cout << __PRETTY_FUNCTION__ << "\n";
    // }
#endif


    JV(curry_t c): JV(c,build_indices<N>{}) {}
    template<int... Indices>
    JV(curry_t c, indices<Indices...>)
      : jit(c.jit), 
	r_addr(c.r_addr), 
	off_full(c.ful), 
	off_level(c.lev),
	F { { {curry_t( c.jit , c.r_addr , c.ful * N , c.lev + c.ful * Indices )}... } }
    {}



    Jit& func() const {
      //std::cout << "JV::func() " << (void*)this << " returns=" << (void*)&jit << "\n";
      return jit;
    }

    T getRegElem( int r_idx ) const {
      //std::cout << __PRETTY_FUNCTION__ << "\n";

      // The following calculates with the jitter
      //
      // r_base = r_addr  +  r_idx * off_full * 4 
      //

      int r_base = r_addr;
  
      int r_wordsize = jit.getRegs( Jit::s32 , 1 );
      jit.asm_mov_literal( r_wordsize , static_cast<int>(sizeof(WordType<T>::Type_t)) );

      int r_full = jit.getRegs( Jit::s32 , 1 );
      jit.asm_mov_literal( r_full , (int)off_full );
  
      jit.asm_mul( r_full , r_wordsize , r_full );
      jit.asm_mul( r_full , r_full , r_idx );
      int r_full_u64 = jit.getRegs( Jit::u64 , 1 );
      jit.asm_cvt( r_full_u64 , r_full );
      jit.asm_add( r_base , r_base , r_full_u64 );

      // only level because ful*Index alreay in r_base
      T ret( curry_t( jit , r_base , off_full * N , off_level ) );  
      return ret;
    }

    const std::array<T,N>& getF() const { return F; }
    std::array<T,N>& getF() { return F; }

    int getLevel() const { return off_level; };
    int getFull() const { return off_full; };
    int getRegAddr() const { return r_addr; };
    
    //JV<T,N>* getOrig() const { return ptr_orig; }
 
  private:
    //JV<T,N>* ptr_orig;
    Jit& jit;
    int off_full;
    int off_level;
    int r_addr;
    std::array<T,N> F;
  };




}

#endif
