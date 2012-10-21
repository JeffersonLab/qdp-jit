#include "qdp.h"

namespace QDP {

  const Jit::RegType JitRegType<float>::Val_t;
  const Jit::RegType JitRegType<double>::Val_t;
  const Jit::RegType JitRegType<int>::Val_t;
  const Jit::RegType JitRegType<unsigned int>::Val_t;
  const Jit::RegType JitRegType<bool>::Val_t;

#if 0
  const Jit::RegType BoolReg<1>::Val_t;
  const Jit::RegType BoolReg<2>::Val_t;
  const Jit::RegType BoolReg<4>::Val_t;
#endif


  //  const Jit::RegType Jit::PTX_Bool;
  //  const std::string Jit::PTX_Bool_OP = "b32";

  //    static const std::string PTX_Bool_OP;{"b32"};   // and.b32  PTX can only do logical on bittypes

  Jit::Jit(const std::string& _filename , const std::string& _funcname ) :
    filename(_filename) , funcname(_funcname)
  {
    nparam=0;
    nreg[f32]=0;
    nreg[f64]=0;
    nreg[u16]=0;
    nreg[u32]=0;
    nreg[u64]=0;
    nreg[s16]=0;
    nreg[s32]=0;
    nreg[s64]=0;
    nreg[u8]=0;
    nreg[b16]=0;
    nreg[b32]=0;
    nreg[b64]=0;
    nreg[pred]=0;
    regprefix[f32]="f";
    regprefix[f64]="d";
    regprefix[u16]="h";
    regprefix[u32]="u";
    regprefix[u64]="w";
    regprefix[s16]="q";
    regprefix[s32]="i";
    regprefix[s64]="l";
    regprefix[u8]="s";
    regprefix[b16]="x";
    regprefix[b32]="y";
    regprefix[b64]="z";
    regprefix[pred]="p";
    regptx[f32]="f32";
    regptx[f64]="f64";
    regptx[u16]="u16";
    regptx[u32]="u32";
    regptx[u64]="u64";
    regptx[s16]="s16";
    regptx[s32]="s32";
    regptx[s64]="s64";
    regptx[u8]="u8";
    regptx[b16]="b16";
    regptx[b32]="b32";
    regptx[b64]="b64";
    regptx[pred]="pred";

    int r_ctaid = getRegs( u16 , 1 );
    int r_ntid = getRegs( u16 , 1 );
    int r_mul = getRegs( u32 , 2 );
    int r_threadId_u32 = getRegs( u32 , 1 );
    r_threadId_s32 = getRegs( s32 , 1 );

    oss_tidcalc <<  "mov.u16 " << getName(r_ctaid) << ",%ctaid.x;\n";
    oss_tidcalc <<  "mov.u16 " << getName(r_ntid) << ",%ntid.x;\n";
    oss_tidcalc <<  "mul.wide.u16 " << getName(r_mul) << "," << getName(r_ntid) << "," << getName(r_ctaid) << ";\n";
    oss_tidcalc <<  "cvt.u32.u16 " << getName(r_mul + 1) << ",%tid.x;\n";
    oss_tidcalc <<  "add.u32 " << getName(r_threadId_u32) << "," << getName(r_mul + 1)  << "," << getName(r_mul) << ";\n";

    int r_lo = addParamImmediate(oss_tidcalc,Jit::s32);
    int r_hi = addParamImmediate(oss_tidcalc,Jit::s32);

    oss_tidcalc <<  "cvt.s32.u32 " << getName(r_threadId_s32) << "," << getName(r_threadId_u32) << ";\n";
    oss_tidcalc <<  "add.s32 " << getName(r_threadId_s32) << "," << getName(r_threadId_s32) << "," << getName(r_lo) << ";\n";

    int r_out_of_range = getRegs( Jit::pred , 1 );
    oss_tidcalc <<  "setp.ge.s32 " << getName(r_out_of_range) << "," << getName(r_threadId_s32) << "," << getName(r_hi) << ";\n";
    oss_tidcalc <<  "@" << getName(r_out_of_range) << " exit;\n";

    // mapCVT[to][from]
    mapCVT[f32][f64]="rz.";

    mapCmpOp[Jit::eq]="eq";
    mapCmpOp[Jit::ne]="ne";
    mapCmpOp[Jit::lt]="lt";
    mapCmpOp[Jit::le]="le";
    mapCmpOp[Jit::gt]="gt";
    mapCmpOp[Jit::ge]="ge";
    mapCmpOp[Jit::lo]="lo";
    mapCmpOp[Jit::ls]="ls";
    mapCmpOp[Jit::hi]="hi";
    mapCmpOp[Jit::hs ]="hs";
    mapCmpOp[Jit::equ]="equ";
    mapCmpOp[Jit::neu]="neu";
    mapCmpOp[Jit::ltu]="ltu";
    mapCmpOp[Jit::leu]="leu";
    mapCmpOp[Jit::gtu]="gtu";
    mapCmpOp[Jit::geu]="geu";
    mapCmpOp[Jit::num]="num";
    mapCmpOp[Jit::nan]="nan";

    mapIntMul[Jit::s32]="lo.";

    
    mapBitType[f32]=b32;
    mapBitType[f64]=b64;
    mapBitType[u16]=b16;
    mapBitType[u32]=b32;
    mapBitType[u64]=b64;
    mapBitType[s16]=b16;
    mapBitType[s32]=b32;
    mapBitType[s64]=b64;
    mapBitType[b32]=b32;


    if (DeviceParams::Instance().getDivRnd()) {
      mapDivRnd[f32]="rn."; // rn
      mapDivRnd[f64]="rn.";
      mapSqrtRnd[f32]="rn."; // rn
      mapSqrtRnd[f64]="rn.";
    } else {
      mapDivRnd[f32]="approx."; // rn
      mapDivRnd[f64]="approx.";
      mapSqrtRnd[f32]="approx."; // rn
      mapSqrtRnd[f64]="approx.";
    }
  }

  int Jit::addParamImmediate(std::ostream& oss,RegType type){
    if (paramtype.size() != nparam) {
      std::cout << "error paramtype.size() != nparam\n";
      exit(1);
    }
    paramtype.push_back(type);
    std::ostringstream tmp;
    tmp << ".param ." << regptx[type] << " param" << nparam;
    param.push_back(tmp.str());
    int r_ret = getRegs( type , 1 );
    oss << "ld.param." << regptx[type] << " " << getName(r_ret) << ",[param" << nparam << "];\n";
    nparam++;
    return r_ret;
  }


  int Jit::getRegId(int id) const {
    return id & ((1 << RegTypeShift)-1);
  }

  Jit::RegType Jit::getRegType(int id) const {
    return (RegType)(id >> RegTypeShift);
  }


  void Jit::asm_mov(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) ) {
      std::cout << "JIT::asm_mov trying to mov types " << getRegType(dest) << " " << getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "mov." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }

  void Jit::asm_st(int base,int offset,int src)
  {
    oss_prg << "st.global." << regptx[getRegType(src)] << " [" << getName(base) << "+" << offset << "]," << getName(src) << ";\n";
  }

  void Jit::asm_ld(int dest,int base,int offset)
  {
    oss_prg << "ld.global." << regptx[getRegType(dest)] << " " << getName(dest) << ",[" << getName(base) << "+" << offset << "];\n";
  }

  void Jit::asm_add(int dest,int lhs,int rhs)
  {
    if ( getRegType(dest) != getRegType(rhs) || getRegType(dest) != getRegType(lhs) ) {
      std::cout << "JIT::asm_add: trying to add different types " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
      exit(1);
    }
    oss_prg << "add." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }

  void Jit::asm_pred_to_01(int dest,int pred)
  {
    if ( getRegType(pred) != Jit::pred ) {
      std::cout << "JIT::asm_selp: type mismatch " << getRegType(dest) << " " << getRegType(pred) << "\n";
      exit(1);
    }
    oss_prg << "selp." << regptx[getRegType(dest)] << " " << getName(dest) << ",1,0," << getName(pred) << ";\n";
  }

  void Jit::asm_01_to_pred(int dest,int src)
  {
    if ( getRegType(dest) != Jit::pred ) {
      std::cout << "JIT::asm_01_to_pred: type mismatch " << getRegType(dest) << " " << getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "setp.ne." << regptx[getRegType(src)] << " " << getName(dest) << "," << getName(src) << ",0;\n";
  }

  void Jit::asm_cmp(CmpOp op,int dest,int lhs,int rhs)
  {
    if ( getRegType(dest) != Jit::pred || getRegType(lhs) != getRegType(rhs) ) {
      std::cout << "JIT::asm_lt: type mismatch " << getRegType(dest) << " " << getRegType(lhs) << "\n";
      exit(1);
    }
    oss_prg << "setp." << mapCmpOp.at(op) << "." << regptx[getRegType(lhs)] << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }

  void Jit::asm_and(int dest,int lhs,int rhs)
  {
    if ( getRegType(dest) != getRegType(rhs) || getRegType(dest) != getRegType(lhs) ) {
      std::cout << "JIT::asm_and: trying to add different types " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
      exit(1);
    }
    oss_prg << "and." << regptx[ getRegType(dest) ]  << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }

  void Jit::asm_or(int dest,int lhs,int rhs)
  {
    if ( getRegType(dest) != getRegType(rhs) || getRegType(dest) != getRegType(lhs) ) {
      std::cout << "JIT::asm_or: trying to add different types " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
      exit(1);
    }
    oss_prg << "or." << regptx[ getRegType(dest) ]  << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }

  void Jit::asm_not(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) ) {
      std::cout << "JIT::asm_not: trying to add different types " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "not." << regptx[ getRegType(dest) ] << " " << getName(dest) << "," << getName(src) << ";\n";
  }

  void Jit::asm_sub(int dest,int lhs,int rhs)
  {
    if ( getRegType(dest) != getRegType(rhs) || getRegType(dest) != getRegType(lhs) ) {
      std::cout << "JIT::asm_sub: trying to add different types " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
      exit(1);
    }
    oss_prg << "sub." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }

  void Jit::asm_mul(int dest,int lhs,int rhs)
  {
    if ( getRegType(dest) != getRegType(rhs) || getRegType(dest) != getRegType(lhs) ) {
      std::cout << "JIT::asm_mul: trying to add different types " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
      exit(1);
    }
    oss_prg << "mul." << mapIntMul[getRegType(dest)] << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }

  void Jit::asm_shl(int dest,int src,int bits)
  {
    if ( mapBitType.at(getRegType(dest)) != mapBitType.at(getRegType(src)) || getRegType(bits) != Jit::u32 ) {
      std::cout << "JIT::asm_shl: type mismatch " << getRegType(dest) << " " << getRegType(src) << " " << getRegType(bits) << "\n";
      exit(1);
    }
    oss_prg << "shl." << regptx.at(mapBitType.at(getRegType(src))) << " " << getName(dest) << "," << getName(src) << "," << getName(bits) << ";\n";
  }

  void Jit::asm_shr(int dest,int src,int bits)
  {
    if ( mapBitType.at(getRegType(dest)) != mapBitType.at(getRegType(src)) || getRegType(bits) != Jit::u32 ) {
      std::cout << "JIT::asm_shr: type mismatch " << getRegType(dest) << " " << getRegType(src) << " " << getRegType(bits) << "\n";
      exit(1);
    }
    oss_prg << "shr." << regptx.at(mapBitType.at(getRegType(src))) << " " << getName(dest) << "," << getName(src) << "," << getName(bits) << ";\n";
  }


  void Jit::asm_bitand(int dest,int lhs,int rhs)
  {
    if ( mapBitType.at(getRegType(dest)) != mapBitType.at(getRegType(lhs)) || 
	 mapBitType.at(getRegType(dest)) != mapBitType.at(getRegType(rhs)) ) {
      std::cout << "JIT::asm_bitand: type mismatch " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
      exit(1);
    }
    oss_prg << "and." << regptx.at(mapBitType.at(getRegType(dest))) << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }


  void Jit::asm_bitor(int dest,int lhs,int rhs)
  {
    if ( mapBitType.at(getRegType(dest)) != mapBitType.at(getRegType(lhs)) || 
	 mapBitType.at(getRegType(dest)) != mapBitType.at(getRegType(rhs)) ) {
      std::cout << "JIT::asm_bitor: type mismatch " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
      exit(1);
    }
    oss_prg << "or." << regptx.at(mapBitType.at(getRegType(dest))) << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }


  void Jit::asm_div(int dest,int lhs,int rhs)
  {
    if ( getRegType(dest) != getRegType(rhs) || getRegType(dest) != getRegType(lhs) ) {
      std::cout << "JIT::asm_div: trying to add different types " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
      exit(1);
    }
    oss_prg << "div." << mapDivRnd[getRegType(dest)] << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
  }

  void Jit::asm_fma(int dest,int lhs,int rhs,int add)
  {
    if ( getRegType(dest) != getRegType(rhs) || getRegType(dest) != getRegType(lhs) || getRegType(dest) != getRegType(add) ) {
      std::cout << "JIT::asm_mul: trying to add different types " 
		<< getRegType(dest) << " " 
		<< getRegType(lhs) << " " 
		<< getRegType(rhs) << " "
		<< getRegType(add) << "\n";
      exit(1);
    }
    oss_prg << "fma.rn." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << "," << getName(add) << ";\n";
  }

  void Jit::asm_neg(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) ) {
      std::cout << "JIT::asm_mul: trying to add different types " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "neg." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }

  void Jit::asm_abs(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) ) {
      std::cout << "JIT::asm_abs: trying to add different types " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "abs." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }

  void Jit::asm_cos(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) || getRegType(src) != Jit::f32 ) {
      std::cout << "JIT::asm_cos: type mismatch " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "cos.approx." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }

  void Jit::asm_sin(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) || getRegType(src) != Jit::f32 ) {
      std::cout << "JIT::asm_sin: type mismatch " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "sin.approx." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }

  void Jit::asm_exp(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) || getRegType(src) != Jit::f32 ) {
      std::cout << "JIT::asm_exp: type mismatch " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "ex2.approx." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }

  void Jit::asm_log(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) || getRegType(src) != Jit::f32 ) {
      std::cout << "JIT::asm_log: type mismatch " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "lg2.approx." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }

  void Jit::asm_sqrt(int dest,int src)
  {
    if ( getRegType(dest) != getRegType(src) ) {
      std::cout << "JIT::asm_sqrt: type mismatch " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "sqrt." << mapSqrtRnd[getRegType(dest)] << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }


  void Jit::asm_cvt(int dest,int src)
  {
    if ( getRegType(dest) == getRegType(src) ) {
      std::cout << "JIT::asm_cvt: trying to convert different types " 
		<< getRegType(dest) << " " 
		<< getRegType(src) << "\n";
      exit(1);
    }
    oss_prg << "cvt." << mapCVT[getRegType(dest)][getRegType(src)] << regptx[getRegType(dest)] << "." << regptx[getRegType(src)] << " " << getName(dest) << "," << getName(src) << ";\n";
    //  oss_prg << "cvt." << mapCVT[getRegType(dest)][getRegType(src)] << regptx[getRegType(dest)] << "." << regptx[getRegType(src)] << " " << getName(dest) << "," << getName(src) << ";\n";
  }


  std::string Jit::getName(int id) const 
  {
    //std::cout << "getName:  type=" << type << "  reg=" << reg << "\n";
    RegType type = getRegType(id);
    int reg = getRegId(id);
    std::ostringstream tmp;
    tmp << regprefix[type];
    tmp << reg;
    return tmp.str();
  }

  int Jit::getRegs(RegType type,int count) 
  {
    //std::cout << "register: type=" << type << "  count=" << count << "\n";
    int reg = nreg[type];
    nreg[type] += count;
    return  reg | (type << RegTypeShift);
  }

  void Jit::dumpVarDefType(RegType type) 
  {
    if (nreg[type] > 0)
      oss_vardef << ".reg ." << regptx[type] << " " << regprefix[type] << "<" << nreg[type] << ">;\n";
  }

  void Jit::dumpVarDef() 
  {
    dumpVarDefType(f32);
    dumpVarDefType(f64);
    dumpVarDefType(u16);
    dumpVarDefType(u32);
    dumpVarDefType(u64);
    dumpVarDefType(s16);
    dumpVarDefType(s32);
    dumpVarDefType(s64);
    dumpVarDefType(u8);
    dumpVarDefType(b16);
    dumpVarDefType(b32);
    dumpVarDefType(b64);
    dumpVarDefType(pred);
  }

  void Jit::dumpParam() 
  {
    for (int i = 0 ; i < param.size() ; i++ ) {
      oss_param << param[i];
      if (i<param.size()-1)
	oss_param << ",\n";
      else
	oss_param << "\n";
    }
  }

  int Jit::addParam(RegType type) 
  {
    if (paramtype.size() != nparam) {
      std::cout << "error paramtype.size() != nparam\n";
      exit(1);
    }
    paramtype.push_back(type);
    std::ostringstream tmp;
    tmp << ".param ." << regptx[type] << " param" << nparam;
    param.push_back(tmp.str());
    int r_ret = getRegs( type , 1 );
    oss_baseaddr << "ld.param." << regptx[type] << " " << getName(r_ret) << ",[param" << nparam << "];\n";
    nparam++;
    return r_ret;
  }

// cvt.u32,s32 u3,s2;         // convert because current index type is "s32"
// mul.wide.u32 w2,u2,4;      // (int*) mul with sizeof(int)
// ld.param.u64 w5,[param2];  // w5 = goffset[]
// add.u64 w6,w5,w2;          // w6 = &goffset[idx]
// ld.param.s32 s0,[w6];      // s0 = goffset[idx],   s0 is the new thread idx, i.e. 'u2'

  int Jit::getRegIdx()
  {
    return r_threadId_s32;
  }

  int Jit::addParamIndexField()
  {
    if (paramtype.size() != nparam) {
      std::cout << "error paramtype.size() != nparam\n";
      exit(1);
    }
    int idx_u32 = getRegs( u32 , 1 );
    int idx_u32_mul_4 = getRegs( u64 , 1 );
    oss_idx <<  "cvt.u32.s32 " << getName(idx_u32) << "," << getName(r_threadId_s32) << ";\n";
    oss_idx <<  "mul.wide.u32 " << getName(idx_u32_mul_4) << "," << getName(idx_u32) << ",4;\n";

    paramtype.push_back(u64);
    std::ostringstream tmp;
    tmp << ".param .u64 param" << nparam;
    param.push_back(tmp.str());

    int r_param = getRegs( u64 , 1 );
    oss_idx << "ld.param.u64 " << getName(r_param) << ",[param" << nparam << "];\n";    
    nparam++;

    int r_param_p_idx = getRegs( u64 , 1 );
    oss_idx << "add.u64 " << getName(r_param_p_idx) << "," << getName(r_param) << "," << getName(idx_u32_mul_4) << ";\n";

    r_threadId_s32 = getRegs( s32 , 1 );
    oss_idx << "ld.global.s32 " << getName(r_threadId_s32) << ",[" << getName(r_param_p_idx) << "];\n";

    return r_threadId_s32;
  }


  Jit::IndexRet Jit::addParamIndexFieldRcvBuf()
  {
    if (paramtype.size() != nparam) {
      std::cout << "error paramtype.size() != nparam\n";
      exit(1);
    }
    int idx_u32 = getRegs( u32 , 1 );
    int idx_u32_mul_4 = getRegs( u64 , 1 );
    oss_idx <<  "cvt.u32.s32 " << getName(idx_u32) << "," << getName(r_threadId_s32) << ";\n";
    oss_idx <<  "mul.wide.u32 " << getName(idx_u32_mul_4) << "," << getName(idx_u32) << ",4;\n";

    paramtype.push_back(u64);
    std::ostringstream tmp;
    tmp << ".param .u64 param" << nparam;
    param.push_back(tmp.str());
    int r_param_goff = getRegs( u64 , 1 );
    oss_idx << "ld.param.u64 " << getName(r_param_goff) << ",[param" << nparam << "];\n";    
    nparam++;

    int r_param_goff_p_idx = getRegs( u64 , 1 );
    oss_idx << "add.u64 " << getName(r_param_goff_p_idx) << "," << getName(r_param_goff) << "," << getName(idx_u32_mul_4) << ";\n";

    r_threadId_s32 = getRegs( s32 , 1 );
    oss_idx << "ld.global.s32 " << getName(r_threadId_s32) << ",[" << getName(r_param_goff_p_idx) << "]; // new_idx\n";

    int r_pred_gez = getRegs( pred , 1 );
    oss_idx << "setp.ge.s32 " << getName(r_pred_gez) << "," << getName(r_threadId_s32) << ",0; // on local node ?\n";

    oss_idx << "@" << getName(r_pred_gez) << "  neg.s32 " << getName(r_threadId_s32) << "," << getName(r_threadId_s32) << "; // rcv buf index\n";
    oss_idx << "@" << getName(r_pred_gez) << "  sub.s32 " << getName(r_threadId_s32) << "," << getName(r_threadId_s32) << ",1; // rcv buf index\n";


    paramtype.push_back(u64);
    tmp.str(std::string());
    tmp << ".param .u64 param" << nparam;
    param.push_back(tmp.str());
    int r_param_rcv_buf = getRegs( u64 , 1 );
    oss_idx << "ld.param.u64 " << getName(r_param_rcv_buf) << ",[param" << nparam << "];   // recv buf base addr\n";    
    nparam++;

    


    IndexRet ret;
    ret.r_newidx = r_threadId_s32;
    ret.r_pred_gez = r_pred_gez;
    ret.r_rcvbuf = r_param_rcv_buf;


    return ret;
  }

  int Jit::addParamLatticeBaseAddr(int r_idx,int wordSize) 
  {
    if (paramtype.size() != nparam) {
      std::cout << "error paramtype.size() != nparam\n";
      exit(1);
    }
    paramtype.push_back(u64);
    std::ostringstream tmp;
    tmp << ".param .u64 param" << nparam;
    param.push_back(tmp.str());
    int r_param = getRegs( u64 , 1 );
    int r_ret = getRegs( u64 , 1 );
    oss_baseaddr << "ld.param.u64 " << getName(r_param) << ",[param" << nparam << "];\n";
    oss_baseaddr << "add.u64 " << getName(r_ret) << "," << getName(r_param) << "," << getName( getThreadIdMultiplied(r_idx,wordSize) ) << ";\n";
    nparam++;
    return r_ret;
  }

  int Jit::addParamScalarBaseAddr() 
  {
    if (paramtype.size() != nparam) {
      std::cout << "error paramtype.size() != nparam\n";
      exit(1);
    }
    paramtype.push_back(u64);
    std::ostringstream tmp;
    tmp << ".param .u64 param" << nparam;
    param.push_back(tmp.str());
    int r_param = getRegs( u64 , 1 );
    oss_baseaddr << "ld.param.u64 " << getName(r_param) << ",[param" << nparam << "];\n";
    nparam++;
    return r_param;
  }

  static int branchNum = 0;

  std::string pushTarget() 
  {
    std::ostringstream tmp;
    tmp << "Branch" << branchNum;
    return tmp.str();
  }

  std::string getTarget() 
  {
    std::ostringstream tmp;
    tmp << "Branch" << branchNum;
    return tmp.str();
  }

  void Jit::addCondBranch(IndexRet i)
  {
    oss_prg << "@!" << getName(i.r_pred_gez) << "  bra " << pushTarget() << ";\n";
  }

  void Jit::addCondBranch2()
  {
    oss_prg << getTarget() << ":\n";
  }

  int Jit::getThreadIdMultiplied(int r_idx,int wordSize)
  {
    if (mapRegMul.count(r_idx) < 1) {
      int tmp = getRegs( u64 , 1 );
      int r_idx_u32 = getRegs( u32 , 1 );
      oss_tidmulti << "cvt.u32.s32 " << getName(r_idx_u32) << "," << getName(r_idx) << ";\n";
      oss_tidmulti << "mul.wide.u32 " << getName(tmp) << "," << getName(r_idx_u32) << "," << wordSize << ";\n";
      mapRegMul[r_idx][wordSize]=tmp;
      return tmp;
    } else {
      if (mapRegMul.at(r_idx).count(wordSize) < 1) {
	int tmp = getRegs( u64 , 1 );
	int r_idx_u32 = getRegs( u32 , 1 );
	oss_tidmulti << "cvt.u32.s32 " << getName(r_idx_u32) << "," << getName(r_idx) << ";\n";
	oss_tidmulti << "mul.wide.u32 " << getName(tmp) << "," << getName(r_idx_u32) << "," << wordSize << ";\n";
	mapRegMul.at(r_idx)[wordSize]=tmp;
	return tmp;
      } else {
	return mapRegMul.at(r_idx).at(wordSize);
      }
    }
  }


  void Jit::write() 
  {
    dumpVarDef();
    dumpParam();
    std::ofstream out(filename.c_str());
#if 1
    out << ".version 1.4\n" <<
      ".target sm_12\n" <<
      ".entry " << funcname << " (" <<
      oss_param.str() << ")\n" <<
      "{\n" <<
      oss_vardef.str() <<
      "//\n// Thread ID calculation\n" <<
      oss_tidcalc.str() <<
      "//\n// Index calculation (Map)\n" <<
      oss_idx.str() <<
      "//\n// Thread ID multiplication\n" <<
      oss_tidmulti.str() <<
      "//\n// Base addresses\n" <<
      oss_baseaddr.str() <<
      "//\n// Main body\n" <<
      oss_prg.str() <<
      "}\n";
#else
    out << ".version 2.3\n" <<
      ".target sm_20\n" <<
      ".address_size 64\n" <<
      ".entry " << funcname << " (" <<
      oss_param.str() << ")\n" <<
      "{\n" <<
      oss_vardef.str() <<
      oss_tidcalc.str() <<
      oss_idx.str() <<
      oss_tidmulti.str() <<
      oss_baseaddr.str() <<
      oss_prg.str() <<
      "}\n";
#endif
    out.close();
  } 


}
