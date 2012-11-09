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


    JV(newspace_t n) : JV(n, build_indices<N>{}) {}
    template<int... Is>
    JV(newspace_t n, indices<Is...>) : 
      jit(n.jit),  F{{(void(Is),n)...}} {}


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

    const std::array<T,N>& getF() const { return F; }
    std::array<T,N>& getF() { return F; }

    int getLevel() const { return off_level; };
    int getFull() const { return off_full; };
    int getRegAddr() const { return r_addr; };
 
  private:
    Jit& jit;
    int off_full;
    int off_level;
    int r_addr;
    std::array<T,N> F;
  };




}

#endif
