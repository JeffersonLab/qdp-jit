#ifndef QDP_JIT_
#define QDP_JIT_

#include<iostream>
#include<memory>
#include<sstream>
#include<fstream>
#include<map>
#include<vector>
#include<array>
#include<string>
#include<cstdlib>
#include<tuple>
#include<set>

#include "nvvm.h"

namespace QDP {

  enum class jit_llvm_builtin { undef,i1,i32,i64,flt,dbl };
  enum class jit_llvm_ind { undef,yes,no };

  const char * jit_get_llvm_builtin( jit_llvm_builtin type );

  std::ostream& operator<< (std::ostream& stream, const jit_llvm_builtin& type );


  class jit_llvm_type {
    jit_llvm_builtin bi;
    jit_llvm_ind     ind;
  public:
    jit_llvm_type(): bi(jit_llvm_builtin::undef),ind(jit_llvm_ind::undef) {}
    jit_llvm_builtin get_builtin() const { assert( bi  != jit_llvm_builtin::undef); return bi; }
    jit_llvm_ind     get_ind()     const { assert( ind != jit_llvm_ind::undef );    return ind; }

    jit_llvm_type( jit_llvm_builtin bi_, jit_llvm_ind ind_ ) : bi(bi_), ind(ind_) {}
    jit_llvm_type( jit_llvm_builtin bi_ )                    : bi(bi_), ind(jit_llvm_ind::no) {}
    bool operator<  (const jit_llvm_type &rhs) const { return std::tie(bi,ind) <  std::tie(rhs.bi,rhs.ind); }
    bool operator!= (const jit_llvm_type &rhs) const { return std::tie(bi,ind) != std::tie(rhs.bi,rhs.ind); }
    bool operator== (const jit_llvm_type &rhs) const { return std::tie(bi,ind) == std::tie(rhs.bi,rhs.ind); }
    friend std::ostream& operator<< (std::ostream& stream, const jit_llvm_type& type ) {
      stream << type.get_builtin();
      if (type.ind == jit_llvm_ind::yes)
	stream << "*";
      return stream;
    }
  };


  bool jit_type_is_float( jit_llvm_builtin bi );
  bool jit_type_is_float( jit_llvm_type bi );



  // enum class jit_state_space { 
  //   state_default , 
  //     state_global , 
  //     state_local , 
  //     state_shared 
  //     };

  // std::ostream& operator<< (std::ostream& stream, const jit_state_space& space );
    

  //
  // MATCHING C TYPES TO PTX TYPES
  //
  template<class T> struct jit_type {};
  template<> struct jit_type<float>            { static constexpr jit_llvm_builtin value = jit_llvm_builtin::flt; };
  template<> struct jit_type<double>           { static constexpr jit_llvm_builtin value = jit_llvm_builtin::dbl; };
  template<> struct jit_type<int>              { static constexpr jit_llvm_builtin value = jit_llvm_builtin::i32; };
  template<> struct jit_type<bool>             { static constexpr jit_llvm_builtin value = jit_llvm_builtin::i1; };



  template<class T>
    std::shared_ptr<T> get(const llvm::Value* & pA) {
    return std::dynamic_pointer_cast< T >( pA );
  }


  //const char * jit_get_ptx_letter( jit_llvm_type type );
  //const char * jit_get_mul_specifier_lo_str( jit_llvm_type type );
  //const char * jit_get_div_specifier( jit_llvm_type type );
  //const char * jit_get_identifier_local_memory();

  namespace PTX {
    std::map< jit_llvm_builtin , std::array<const char*,1> > create_ptx_type_matrix();
    extern std::map< jit_llvm_builtin , std::array<const char*,1> > ptx_type_matrix;
    extern const std::map< jit_llvm_builtin , std::map<jit_llvm_builtin,jit_llvm_builtin > > map_promote;
    //extern const std::map< jit_llvm_type , jit_llvm_type >               map_wide_promote;
  }


  class jit_function {
    std::ostringstream oss_prg;
    std::ostringstream oss_signature;
    std::ostringstream oss_signature_meta;
    std::ostringstream oss_reg_defs;
    int reg_count;

    //typedef std::vector<std::pair<jit_llvm_type,int> > LocalCountVec;
    //LocalCountVec vec_local_count;
    //int local_count;

    bool m_shared;
    std::vector<bool> m_include_math_ptx_unary;
    std::vector<bool> m_include_math_ptx_binary;
    jit_block_t entry_block;
  public:
    jit_block_t get_entry_block() { return entry_block; }
    std::string get_kernel_as_string();
    void emitShared();
    //int local_alloc( jit_llvm_type type, int count );
    void write_reg_defs();
    jit_function();
    ~jit_function();
    int reg_alloc();
    std::ostringstream& get_prg();
    std::ostringstream& get_signature();
    std::ostringstream& get_signature_meta();
  };

  extern jit_function_t jit_internal_function;

  void jit_start_new_function();
  jit_function_t jit_get_function();
  jit_block_t jit_get_entry_block();
  jit_block_t jit_get_current_block();
  std::string jit_get_kernel_as_string();
  CUfunction jit_get_cufunction(const char* fname);
  

  // class jit_function_singleton
  // {
  //   static jit_function singleton {
  //   return singleton;
  //   };





  class jit_block {
    static int count;
    int count_m;
  public:
    jit_block() {
      count_m = count++;
    }
    friend std::ostream& operator<< (std::ostream& stream, const jit_block& lab) {
      stream << "L" << lab.count_m;
      return stream;
    }
  };


  std::ostream& operator<< (std::ostream& stream, const jit_block_t& block );
  std::ostream& operator<< (std::ostream& stream, jit_block_t& block );

  jit_block_t jit_block_create();


  llvm::Value * llvm_create_value();
  llvm::Value * llvm_create_value(int val);
  llvm::Value * llvm_create_value(size_t val);
  llvm::Value * llvm_create_value(float val);
  llvm::Value * llvm_create_value(double val);
  llvm::Value * llvm_create_value(jit_llvm_type type);


  class llvm::Value* {
    llvm_create_value( const llvm::Value*& rhs );
  public:


    llvm::Value*& operator=( const llvm::Value*& rhs );

    llvm_create_value();
    llvm_create_value( int val );
    llvm_create_value( size_t val );
    llvm_create_value( double val );
    explicit llvm_create_value( jit_llvm_type type_ );


    
    void reg_alloc();
    void assign( const llvm::Value*& rhs );
    ~llvm_create_value() {}

    jit_llvm_type get_type() const;

    // void set_state_space( jit_state_space space );
    // jit_state_space get_state_space() const;
    
    //int get_number() const;
    std::string get_name() const;
    bool get_ever_assigned() const { return ever_assigned; }
    void set_ever_assigned() { ever_assigned = true; }

    friend std::ostream& operator<< (std::ostream& stream, const llvm::Value*& op) {
      if (op.is_constant) {
	if (op.is_int) 
	  stream << op.const_int;
	else {
	  std::ostringstream oss; 
	  oss.setf(ios::scientific);
	  oss.precision(std::numeric_limits<double>::digits10 + 1);
	  oss << op.const_double;
	  stream << oss.str();
	}
      } else {
	stream << "%r" << op.number;
      }
      return stream;
    }
    friend std::ostream& operator<< (std::ostream& stream, const llvm::Value *& op) {
      if (!op)
	QDP_error_exit("You are trying to get the name of a not-initialized jit value");
      stream << *op;
      return stream;
    }
    friend std::ostream& operator<< (std::ostream& stream, llvm::Value *& op) {
      if (!op)
	op = llvm_create_value();
      stream << *op;
      return stream;
    }

  private:
    bool   ever_assigned;
    bool   is_constant;
    bool   is_int;
    int    const_int;
    double const_double;
    jit_llvm_type type;
    int number;
    jit_block_t block;

    //jit_state_space mem_state;
  };


  // const char * get_state_space_str( jit_state_space space );


  struct IndexRet {
    // IndexRet():
    //   r_newidx_local(jit_llvm_type::s32),
    //   r_newidx_buffer(jit_llvm_type::s32),
    //   r_pred_in_buf(jit_llvm_type::pred),
    //   r_rcvbuf(jit_llvm_type::u64)
    // {}
    IndexRet(){}
    llvm::Value * r_newidx_local;
    llvm::Value * r_newidx_buffer;
    llvm::Value * r_pred_in_buf;
    llvm::Value * r_rcvbuf;
  };

  
  




  //jit_function_t jit_get_valid_func( jit_function_t f0 ,jit_function_t f1 );  
  // std::string jit_predicate( const llvm::Value *& pred );
  // jit_llvm_type jit_bit_type(jit_llvm_type type);
  // jit_llvm_type jit_type_wide_promote(jit_llvm_type t0);

  // jit_state_space jit_state_promote( jit_state_space ss0 , jit_state_space ss1 );

  jit_llvm_builtin jit_type_promote(jit_llvm_builtin t0,jit_llvm_builtin t1);
  jit_llvm_type    jit_type_promote(jit_llvm_type t0   ,jit_llvm_type t1);

  void llvm_bar_sync( int a );

  llvm::Value * llvm_add_param( jit_llvm_type type );
  llvm::Value * llvm_alloca( jit_llvm_builtin type , int count );
  llvm::Value * jit_get_shared_mem_ptr();

  //std::string jit_get_reg_name( const llvm::Value *& val );


  // llvm::Value *_reg_t jit_val_create_new( int type );
  // llvm::Value *_reg_t jit_val_create_from_const( int type , int const_val , llvm::Value * pred=llvm::Value *() );
  // llvm::Value *_reg_t jit_val_create_convert( int type , llvm::Value * val , llvm::Value * pred=llvm::Value *() );
  // llvm::Value * jit_val_create_copy( llvm::Value * val , llvm::Value * pred=llvm::Value *() );


  llvm::Value * jit_val_convert( jit_llvm_type type , const llvm::Value *& rhs );


  llvm::Value * jit_geom_get_tidx( );
  llvm::Value * jit_geom_get_ntidx( );
  llvm::Value * jit_geom_get_ctaidx( );

  // Binary operations
  //llvm::Value * llvm_mul_wide( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_mul( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_div( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_add( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_sub( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_shl( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_shr( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_and( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_or ( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_xor( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );
  void llvm_rem( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs  );

  llvm::Value * llvm_mul( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_div( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_add( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_sub( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_shl( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_shr( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_and( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_or ( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_xor( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_rem( const llvm::Value *& lhs , const llvm::Value *& rhs  );

  // ni
  llvm::Value * llvm_mod( llvm::Value *& dest, const llvm::Value *& lhs , const llvm::Value *& rhs );

  // Binary operations returning predicate
  llvm::Value * llvm_lt( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_ne( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_eq( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_ge( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_le( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  llvm::Value * llvm_gt( const llvm::Value *& lhs , const llvm::Value *& rhs  );

  // Native PTX Unary operations
  llvm::Value * llvm_neg( const llvm::Value *& lhs  );
  llvm::Value * llvm_not( const llvm::Value *& lhs  );
  llvm::Value * llvm_fabs( const llvm::Value *& lhs  );
  //llvm::Value * llvm_floor( const llvm::Value *& lhs  );
  //llvm::Value * llvm_sqrt( const llvm::Value *& lhs  );

  // Imported PTX Unary operations single precision
  // llvm::Value * llvm_sin_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_acos_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_asin_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_atan_f32( const llvm::Value *& lhs  );
  // //llvm::Value * llvm_ceil_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_cos_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_cosh_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_exp_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_log_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_log10_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_sinh_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_tan_f32( const llvm::Value *& lhs  );
  // llvm::Value * llvm_tanh_f32( const llvm::Value *& lhs  );

  // Imported PTX Binary operations single precision
  // llvm::Value * llvm_pow_f32( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  // llvm::Value * llvm_atan2_f32( const llvm::Value *& lhs , const llvm::Value *& rhs  );

  // Imported PTX Unary operations double precision
  // llvm::Value * llvm_sin_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_acos_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_asin_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_atan_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_ceil_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_cos_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_cosh_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_exp_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_log_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_log10_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_sinh_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_tan_f64( const llvm::Value *& lhs  );
  // llvm::Value * llvm_tanh_f64( const llvm::Value *& lhs  );

  // Imported PTX Binary operations single precision
  // llvm::Value * llvm_pow_f64( const llvm::Value *& lhs , const llvm::Value *& rhs  );
  // llvm::Value * llvm_atan2_f64( const llvm::Value *& lhs , const llvm::Value *& rhs  );


  // Select
  //llvm::Value * llvm_selp( const llvm::Value *& lhs , const llvm::Value *& rhs , const llvm::Value *& p );

  void llvm_mov( llvm::Value *& dest , const llvm::Value *& src  );
  void llvm_mov( llvm::Value *& dest , const std::string& src  );

  void llvm_branch( jit_block_t& block );
  void llvm_branch( const llvm::Value *& cond,  jit_block_t& block_true , jit_block_t& block_false );

  jit_block_t llvm_start_new_block();
  void llvm_start_block(  jit_block_t& label );
  void llvm_comment(  const char * comment );
  void llvm_exit();
  void llvm_cond_exit( const llvm::Value *& cond );

  llvm::Value * jit_int_array_indirection( const llvm::Value *& idx , jit_llvm_builtin type );

  llvm::Value * llvm_load ( const llvm::Value *& base , 
			     const llvm::Value *& offset , 
			     jit_llvm_type type  );

  void llvm_store( const llvm::Value *& base , 
		      const llvm::Value *& offset , 
		      jit_llvm_type type , 
		      const llvm::Value *& val  );

  llvm::Value * llvm_thread_idx();

  llvm::Value * llvm_phi( const llvm::Value *& v0 , jit_block_t& b0 ,
			   const llvm::Value *& v1 , jit_block_t& b1 );

  class JitOp {
  protected:
    virtual std::ostream& writeToStream( std::ostream& stream ) const = 0;
    jit_llvm_type args_type;
  public:
    JitOp( const llvm::Value *& lhs , const llvm::Value *& rhs ) :
      args_type( jit_type_promote( lhs->get_type() , rhs->get_type() ) ) {}
    jit_llvm_type getArgsType() const { return args_type; }
    virtual jit_llvm_type getDestType() const { return this->getArgsType(); }
    friend std::ostream& operator<< (std::ostream& stream, const JitOp& op) {
      return op.writeToStream(stream);
    }
  };

  class JitOpAdd: public JitOp {
  public:
    JitOpAdd( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fadd" : "add";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpSub: public JitOp {
  public:
    JitOpSub( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fsub" : "sub";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpMul: public JitOp {
  public:
    JitOpMul( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fmul" : "mul";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpDiv: public JitOp {
  public:
    JitOpDiv( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fdiv" : "sdiv";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  // class JitOpMulWide: public JitOp {
  // public:
  //   JitOpMulWide( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
  //   virtual jit_llvm_type getDestType() const { 
  //     return jit_type_wide_promote( getArgsType() );
  //   }
  //   virtual std::ostream& writeToStream( std::ostream& stream ) const {
  //     std::string specifier_wide = 
  // 	getArgsType() != getDestType() ? 
  // 	"wide ":
  // 	jit_get_mul_specifier_lo_str( getDestType() );
  //     stream << "mul " 
  // 	     << specifier_wide
  // 	     << jit_get_ptx_type( getArgsType() );
  //     return stream;
  //   }
  // };

  class JitOpSHL: public JitOp {
  public:
    JitOpSHL( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "shl";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpSHR: public JitOp {
  public:
    JitOpSHR( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "shr";
      stream << op << " " << getDestType();
      return stream;
    }
  };



  class JitOpNE: public JitOp {
  public:
    JitOpNE( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual jit_llvm_type getDestType() const {
      return jit_llvm_builtin::i1;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fcmp one" : "icmp ne";
      stream << op << " " << getArgsType();
      return stream;
    }
  };

  class JitOpEQ: public JitOp {
  public:
    JitOpEQ( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual jit_llvm_type getDestType() const {
      return jit_llvm_builtin::i1;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fcmp oeq" : "icmp eq";
      stream << op << " " << getArgsType();
      return stream;
    }
  };

  class JitOpLE: public JitOp {
  public:
    JitOpLE( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual jit_llvm_type getDestType() const {
      return jit_llvm_builtin::i1;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fcmp ole" : "icmp sle";
      stream << op << " " << getArgsType();
      return stream;
    }
  };


  class JitOpLT: public JitOp {
  public:
    JitOpLT( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual jit_llvm_type getDestType() const {
      return jit_llvm_builtin::i1;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fcmp olt" : "icmp slt";
      stream << op << " " << getArgsType();
      return stream;
    }
  };


  class JitOpGE: public JitOp {
  public:
    JitOpGE( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual jit_llvm_type getDestType() const {
      return jit_llvm_builtin::i1;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fcmp oge" : "icmp sge";
      stream << op << " " << getArgsType();
      return stream;
    }
  };

  class JitOpGT: public JitOp {
  public:
    JitOpGT( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual jit_llvm_type getDestType() const {
      return jit_llvm_builtin::i1;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fcmp ogt" : "icmp sgt";
      stream << op << " " << getArgsType();
      return stream;
    }
  };

  class JitOpAnd: public JitOp {
  public:
    JitOpAnd( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "and";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpOr: public JitOp {
  public:
    JitOpOr( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "or";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpXOr: public JitOp {
  public:
    JitOpXOr( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "xor";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpRem: public JitOp {
  public:
    JitOpRem( const llvm::Value *& lhs , const llvm::Value *& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "rem";
      stream << op << " " << getDestType();
      return stream;
    }
  };






  class JitUnaryOp {
  protected:
    virtual std::ostream& writeToStream( std::ostream& stream ) const = 0;
    jit_llvm_type type;
  public:
    JitUnaryOp( jit_llvm_type type_ ): type(type_) {}
    friend std::ostream& operator<< (std::ostream& stream, const JitUnaryOp& op) {
      return op.writeToStream(stream);
    }
  };

  class JitUnaryOpNeg: public JitUnaryOp {
  public:
    JitUnaryOpNeg( jit_llvm_type type_ ): JitUnaryOp(type_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "neg "
	     << type;
      return stream;
    }
  };

  class JitUnaryOpNot: public JitUnaryOp {
  public:
    JitUnaryOpNot( jit_llvm_type type_ ): JitUnaryOp(type_) {
      assert( type_.get_builtin() == jit_llvm_builtin::i1 );
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "xor "
	     << type << " "
	     << "-1, ";
      return stream;
    }
  };

  class JitUnaryOpAbs: public JitUnaryOp {
  public:
    JitUnaryOpAbs( jit_llvm_type type_ ): JitUnaryOp(type_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "abs "
	     << type;
      return stream;
    }
  };

  // class JitUnaryOpFloor: public JitUnaryOp {
  // public:
  //   JitUnaryOpFloor( jit_llvm_type type_ ): JitUnaryOp(type_) {}
  //   virtual std::ostream& writeToStream( std::ostream& stream ) const {
  //     stream << "cvt.rmi "
  // 	     << type
  // 	     << " "
  // 	     << type;
  //     return stream;
  //   }
  // };

  // class JitUnaryOpSqrt: public JitUnaryOp {
  // public:
  //   JitUnaryOpSqrt( jit_llvm_type type_ ): JitUnaryOp(type_) {}
  //   virtual std::ostream& writeToStream( std::ostream& stream ) const {
  //     stream << "sqrt ";
  //     if ( DeviceParams::Instance().getMajor() >= 2 )
  // 	stream << "rn ";
  //     else
  // 	stream << "approx ";
  //     stream << type;
  //     return stream;
  //   }
  // };

  // class JitUnaryOpCeil: public JitUnaryOp {
  // public:
  //   JitUnaryOpCeil( jit_llvm_type type_ ): JitUnaryOp(type_) {}
  //   virtual std::ostream& writeToStream( std::ostream& stream ) const {
  //     stream << "cvt.rpi "
  // 	     << type
  // 	     << " "
  // 	     << type;
  //     return stream;
  //   }
  // };


}

#endif
