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
    enum RegType { f32=0,f64=1,u16=2,u32=3,u64=4,s16=5,s32=6,s64=7,u8=8,b32=9,pred=10 };
    enum CmpOp { eq, ne, lt, le, gt, ge, lo, ls, hi, hs , equ, neu, ltu, leu, gtu, geu, num, nan };

    // Specify the PTX register type to use when doing logical operations.
    // C++ bool is 1 bytes. PTX can't operate on 8 bits types. Must cast each access.
    //typedef unsigned int PTX_Bool_C_equiv;
    //static const Jit::RegType PTX_Bool = u32;
    //static const std::string PTX_Bool_OP;//{"b32"};   // and.b32  PTX can only do logical on bittypes

    Jit(const std::string& _filename , const std::string& _funcname );

    void asm_mov(int dest,int src);
    void asm_st(int base,int offset,int src);
    void asm_ld(int dest,int base,int offset);
    void asm_add(int dest,int lhs,int rhs);
    void asm_and(int dest,int lhs,int rhs);
    void asm_or(int dest,int lhs,int rhs);
    void asm_sub(int dest,int lhs,int rhs);
    void asm_mul(int dest,int lhs,int rhs);
    void asm_div(int dest,int lhs,int rhs);
    void asm_fma(int dest,int lhs,int rhs,int add);
    void asm_neg(int dest,int src);
    void asm_abs(int dest,int src);
    void asm_not(int dest,int src);
    void asm_cvt(int dest,int src);
    void asm_pred_to_01(int dest,int pred);
    void asm_01_to_pred(int pred,int src);
    void asm_cmp(CmpOp op,int dest,int lhs,int rhs);
    void asm_cos(int dest,int src);
    void asm_sin(int dest,int src);
    void asm_exp(int dest,int src);
    void asm_log(int dest,int src);
    void asm_sqrt(int dest,int src);

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
    mutable std::map< RegType , std::string > mapDivRnd;
    mutable std::map< RegType , std::string > mapSqrtRnd;
    mutable std::map< CmpOp , std::string > mapCmpOp;
  };


  template <class T> struct JitRegType {};

  template <> struct JitRegType<float>        { static const Jit::RegType Val_t = Jit::f32; };
  template <> struct JitRegType<double>       { static const Jit::RegType Val_t = Jit::f64; };
  template <> struct JitRegType<int>          { static const Jit::RegType Val_t = Jit::s32; };
  template <> struct JitRegType<unsigned int> { static const Jit::RegType Val_t = Jit::u32; };
  template <> struct JitRegType<bool>         { static const Jit::RegType Val_t = Jit::pred; };

#if 0
  template<int BoolSize> struct BoolReg {};
  template<> struct BoolReg<1> { static const Jit::RegType Val_t = Jit::u8; };
  template<> struct BoolReg<2> { static const Jit::RegType Val_t = Jit::u16; };
  template<> struct BoolReg<4> { static const Jit::RegType Val_t = Jit::u32; };
  template <> struct JitRegType<bool>         { static const Jit::RegType Val_t = BoolReg< sizeof(bool) >::Val_t; }; // 
#endif

}


#endif
