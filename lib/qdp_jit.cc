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
  nreg[b32]=0;
  regprefix[f32]="f";
  regprefix[f64]="d";
  regprefix[u16]="h";
  regprefix[u32]="u";
  regprefix[u64]="w";
  regprefix[s16]="q";
  regprefix[s32]="i";
  regprefix[s64]="l";
  regprefix[u8]="s";
  regprefix[b32]="x";
  regptx[f32]="f32";
  regptx[f64]="f64";
  regptx[u16]="u16";
  regptx[u32]="u32";
  regptx[u64]="u64";
  regptx[s16]="s16";
  regptx[s32]="s32";
  regptx[s64]="s64";
  regptx[u8]="u8";
  regptx[b32]="b32";

  int r_ctaid = getRegs( u16 , 1 );
  int r_ntid = getRegs( u16 , 1 );
  int r_mul = getRegs( u32 , 2 );
  r_threadId_u32 = getRegs( u32 , 1 );

  oss_tidcalc <<  "mov.u16 " << getName(r_ctaid) << ",%ctaid.x;\n";
  oss_tidcalc <<  "mov.u16 " << getName(r_ntid) << ",%ntid.x;\n";
  oss_tidcalc <<  "mul.wide.u16 " << getName(r_mul) << "," << getName(r_ntid) << "," << getName(r_ctaid) << ";\n";
  oss_tidcalc <<  "cvt.u32.u16 " << getName(r_mul + 1) << ",%tid.x;\n";
  oss_tidcalc <<  "add.u32 " << getName(r_threadId_u32) << "," << getName(r_mul + 1)  << "," << getName(r_mul) << ";\n";

  // mapCVT[to][from]
  mapCVT[f32][f64]="rz.";

  mapBitType[u32]=b32;
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

void Jit::asm_and(int dest,int lhs,int rhs)
{
  if ( getRegType(dest) != getRegType(rhs) || getRegType(dest) != getRegType(lhs) ) {
    std::cout << "JIT::asm_and: trying to add different types " << getRegType(dest) << " " << getRegType(lhs) << " " << getRegType(rhs) << "\n";
    exit(1);
  }
  oss_prg << "and." << regptx[ mapBitType.at( getRegType(dest) ) ]  << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
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
  oss_prg << "mul." << regptx[getRegType(dest)] << " " << getName(dest) << "," << getName(lhs) << "," << getName(rhs) << ";\n";
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

int Jit::addParamLatticeBaseAddr(int wordSize) 
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
  oss_baseaddr << "add.u64 " << getName(r_ret) << "," << getName(r_param) << "," << getName( getThreadIdMultiplied(wordSize) ) << ";\n";
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

int Jit::getThreadIdMultiplied(int wordSize) 
{
  //std::cout << "getThreadIdMultiplied count = " << threadIdMultiplied.count(wordSize) << "\n";
  if (threadIdMultiplied.count(wordSize) < 1) {
    int tmp = getRegs( u64 , 1 );
    oss_tidmulti << "mul.wide.u32 " << getName(tmp) << "," << getName(r_threadId_u32) << "," << wordSize << ";\n";
    threadIdMultiplied[wordSize]=tmp;
    return tmp;
  } else {
    return threadIdMultiplied[wordSize];
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
    oss_tidcalc.str() <<
    oss_tidmulti.str() <<
    oss_baseaddr.str() <<
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
    oss_tidmulti.str() <<
    oss_baseaddr.str() <<
    oss_prg.str() <<
    "}\n";
#endif
  out.close();
} 


}
