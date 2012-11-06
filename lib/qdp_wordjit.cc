#include "qdp.h"

namespace QDP {

#if 0
  template<>
  void WordJIT<bool>::store() {
    if (needsStoring) {
      if (!global_state)
	QDP_error_exit("WordJIT<bool> store, but no global state");
      QDP_info("WordJIT inserting store asm instruction");

      int u32 = jit.getRegs( Jit::u32 , 1 );
      jit.asm_pred_to_01( u32 , getReg( JitRegType<bool>::Val_t , DoNotLoad ) );
      int u8 = jit.getRegs( Jit::u8 , 1 );
      jit.asm_cvt( u8 , u32 );

      jit.asm_st( r_addr , offset_level * WordSize<bool>::Size , u8 );
      needsStoring = false;
    }
  }
#endif


  template<>
  int WordJIT<bool>::getReg( Jit::RegType type ) const 
  {
    //std::cout << "BOOL SPECIAL getReg type=" << type << "  mapReg.count(type)=" << mapReg.count(type) << "  load = " << load << "  mapReg.size()=" << mapReg.size() << "\n";
    if (mapReg.count(type) > 0) {
      // We already have the value in a register of the type requested
      //std::cout << jit.getName(mapReg.at(type)) << "\n";
      return mapReg.at(type);
    } else {
      if (mapReg.size() > 0) {
	// SANITY
	if (mapReg.size() > 1)
	  QDP_error_exit("getReg: We already have the value in 2 different types. Now a 3rd one ??");
	// We have the value in a register, but not with the requested type 
	//std::cout << "SPECIAL We have the value in a register, but not with the requested type\n";
	MapRegType::iterator loaded = mapReg.begin();
	Jit::RegType loadedType = loaded->first;
	int loadedId = loaded->second;
	mapReg.insert( std::make_pair( type , jit.getRegs( type , 1 ) ) );
	jit.asm_cvt( mapReg.at(type) , loadedId );
	return mapReg.at(type);
      } else {
	// We don't have the value in a register. Need to load it.
	//std::cout << "SPECIAL We don't have the value in a register. Need to load it " << (void*)this << " " << (void*)&jit << "\n";
	Jit::RegType myType = JitRegType<bool>::Val_t;
	mapReg.insert( std::make_pair( myType , jit.getRegs( JitRegType<bool>::Val_t , 1 ) ) );

	//std::cout << "insert u8 ld/cvt instructions\n";
	int load_u8 = jit.getRegs( Jit::u8 , 1 );
	jit.asm_ld( load_u8 , r_addr , offset_level * WordSize<bool>::Size );

	int load_u32 = jit.getRegs( Jit::u32 , 1 );
	jit.asm_cvt( load_u32 , load_u8 );

	jit.asm_01_to_pred( mapReg.at( myType) , load_u32 );

	return getReg(type);
      }
    }
  }

}
