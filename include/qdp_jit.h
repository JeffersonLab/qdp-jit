#ifndef QDP_JIT_
#define QDP_JIT_

#include<iostream>
#include<sstream>
#include<fstream>
#include<map>
#include<vector>
#include<string>
#include<stdlib.h>

namespace QDP {

  class Jit {
  public:
    enum { RegTypeShift = 24 };
    enum RegType { f32=0,f64=1,u16=2,u32=3,u64=4,s16=5,s32=6,s64=7 };

    Jit(const std::string& _filename , const std::string& _funcname );

    void asm_mov(int dest,int src);
    void asm_st(int base,int offset,int src);
    void asm_ld(int dest,int base,int offset);
    void asm_add(int dest,int lhs,int rhs);
    void asm_sub(int dest,int lhs,int rhs);
    void asm_mul(int dest,int lhs,int rhs);
    void asm_fma(int dest,int lhs,int rhs,int add);
    void asm_neg(int dest,int src);
    void asm_cvt(int dest,int src);

    std::string getName(int id) const;
    int getRegs(RegType type,int count);
    void dumpVarDefType(RegType type);
    void dumpVarDef();
    void dumpParam();
    int addParam(RegType type);
    int addParamLatticeBaseAddr(int wordSize);
    int addParamScalarBaseAddr();
    int getThreadIdMultiplied(int wordSize);
    void write();

  private:
    int getRegId(int id) const;
    RegType getRegType(int id) const;

    //
    mutable std::string filename,funcname;
    mutable int r_threadId_u32;
    mutable std::map<int,int> threadIdMultiplied;
    mutable int nparam;
    mutable std::map<RegType,int> nreg;
    mutable std::map<RegType,int> regsize;
    mutable std::map<RegType,std::string> regprefix;
    mutable std::map<RegType,std::string> regptx;
    mutable std::vector<std::string> param;
    mutable std::vector<RegType> paramtype;
    mutable std::ostringstream oss_prg;
    mutable std::ostringstream oss_tidcalc;
    mutable std::ostringstream oss_tidmulti;
    mutable std::ostringstream oss_baseaddr;
    mutable std::ostringstream oss_vardef;
    mutable std::ostringstream oss_param;
    mutable std::string param_prefix;
    mutable std::map< int , std::map< int , std::string > > mapCVT;
  };


  template <class T> struct JitRegType {};

  template <> struct JitRegType<float>  { static const Jit::RegType Val_t = Jit::f32; };
  template <> struct JitRegType<double> { static const Jit::RegType Val_t = Jit::f64; };
  template <> struct JitRegType<int>    { static const Jit::RegType Val_t = Jit::s32; };

}


#endif
