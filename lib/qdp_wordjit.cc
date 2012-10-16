#include "qdp.h"

namespace QDP {

  template<>
  int WordJIT<bool>::getReg( Jit::RegType type ) const 
  {
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
	Jit::RegType myType = JitRegType<bool>::Val_t;
	mapReg.insert( std::make_pair( myType , jit.getRegs( JitRegType<bool>::Val_t , 1 ) ) );

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
