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
    std::shared_ptr<T> get(const jit_value & pA) {
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
    int local_alloc( jit_llvm_type type, int count );
    void write_reg_defs();
    jit_function();
    ~jit_function();
    int reg_alloc();
    std::ostringstream& get_prg();
    std::ostringstream& get_signature();
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


  jit_value_t create_jit_value();
  jit_value_t create_jit_value(int val);
  jit_value_t create_jit_value(size_t val);
  jit_value_t create_jit_value(float val);
  jit_value_t create_jit_value(double val);
  jit_value_t create_jit_value(jit_llvm_type type);


  class jit_value {
    jit_value( const jit_value& rhs );
  public:


    jit_value& operator=( const jit_value& rhs );

    jit_value();
    jit_value( int val );
    jit_value( size_t val );
    jit_value( double val );
    explicit jit_value( jit_llvm_type type_ );


    
    void reg_alloc();
    void assign( const jit_value& rhs );
    ~jit_value() {}

    jit_llvm_type get_type() const;

    // void set_state_space( jit_state_space space );
    // jit_state_space get_state_space() const;
    
    //int get_number() const;
    std::string get_name() const;
    bool get_ever_assigned() const { return ever_assigned; }
    void set_ever_assigned() { ever_assigned = true; }

    friend std::ostream& operator<< (std::ostream& stream, const jit_value& op) {
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
    friend std::ostream& operator<< (std::ostream& stream, const jit_value_t& op) {
      if (!op)
	QDP_error_exit("You are trying to get the name of a not-initialized jit value");
      stream << *op;
      return stream;
    }
    friend std::ostream& operator<< (std::ostream& stream, jit_value_t& op) {
      if (!op)
	op = create_jit_value();
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
    jit_value_t r_newidx_local;
    jit_value_t r_newidx_buffer;
    jit_value_t r_pred_in_buf;
    jit_value_t r_rcvbuf;
  };

  
  




  //jit_function_t jit_get_valid_func( jit_function_t f0 ,jit_function_t f1 );  
  // std::string jit_predicate( const jit_value_t& pred );
  // jit_llvm_type jit_bit_type(jit_llvm_type type);
  // jit_llvm_type jit_type_wide_promote(jit_llvm_type t0);

  // jit_state_space jit_state_promote( jit_state_space ss0 , jit_state_space ss1 );

  jit_llvm_builtin jit_type_promote(jit_llvm_builtin t0,jit_llvm_builtin t1);
  jit_llvm_type    jit_type_promote(jit_llvm_type t0   ,jit_llvm_type t1);

  void jit_ins_bar_sync( int a );

  jit_value_t jit_add_param( jit_llvm_type type );
  jit_value_t jit_allocate_local( jit_llvm_type type , int count );
  jit_value_t jit_get_shared_mem_ptr();

  //std::string jit_get_reg_name( const jit_value_t& val );


  // jit_value_t_reg_t jit_val_create_new( int type );
  // jit_value_t_reg_t jit_val_create_from_const( int type , int const_val , jit_value_t pred=jit_value_t() );
  // jit_value_t_reg_t jit_val_create_convert( int type , jit_value_t val , jit_value_t pred=jit_value_t() );
  // jit_value_t jit_val_create_copy( jit_value_t val , jit_value_t pred=jit_value_t() );


  jit_value_t jit_val_convert( jit_llvm_type type , const jit_value_t& rhs );


  jit_value_t jit_geom_get_tidx( );
  jit_value_t jit_geom_get_ntidx( );
  jit_value_t jit_geom_get_ctaidx( );

  // Binary operations
  //jit_value_t jit_ins_mul_wide( const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_mul( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_div( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_add( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_sub( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_shl( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_shr( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_and( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_or ( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_xor( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );
  void jit_ins_rem( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs  );

  jit_value_t jit_ins_mul( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_div( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_add( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_sub( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_shl( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_shr( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_and( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_or ( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_xor( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_rem( const jit_value_t& lhs , const jit_value_t& rhs  );

  // ni
  jit_value_t jit_ins_mod( jit_value_t& dest, const jit_value_t& lhs , const jit_value_t& rhs );

  // Binary operations returning predicate
  jit_value_t jit_ins_lt( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_ne( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_eq( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_ge( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_le( const jit_value_t& lhs , const jit_value_t& rhs  );
  jit_value_t jit_ins_gt( const jit_value_t& lhs , const jit_value_t& rhs  );

  // Native PTX Unary operations
  jit_value_t jit_ins_neg( const jit_value_t& lhs  );
  jit_value_t jit_ins_not( const jit_value_t& lhs  );
  jit_value_t jit_ins_fabs( const jit_value_t& lhs  );
  //jit_value_t jit_ins_floor( const jit_value_t& lhs  );
  //jit_value_t jit_ins_sqrt( const jit_value_t& lhs  );

  // Imported PTX Unary operations single precision
  // jit_value_t jit_ins_sin_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_acos_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_asin_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_atan_f32( const jit_value_t& lhs  );
  // //jit_value_t jit_ins_ceil_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_cos_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_cosh_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_exp_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_log_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_log10_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_sinh_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_tan_f32( const jit_value_t& lhs  );
  // jit_value_t jit_ins_tanh_f32( const jit_value_t& lhs  );

  // Imported PTX Binary operations single precision
  // jit_value_t jit_ins_pow_f32( const jit_value_t& lhs , const jit_value_t& rhs  );
  // jit_value_t jit_ins_atan2_f32( const jit_value_t& lhs , const jit_value_t& rhs  );

  // Imported PTX Unary operations double precision
  // jit_value_t jit_ins_sin_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_acos_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_asin_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_atan_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_ceil_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_cos_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_cosh_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_exp_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_log_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_log10_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_sinh_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_tan_f64( const jit_value_t& lhs  );
  // jit_value_t jit_ins_tanh_f64( const jit_value_t& lhs  );

  // Imported PTX Binary operations single precision
  // jit_value_t jit_ins_pow_f64( const jit_value_t& lhs , const jit_value_t& rhs  );
  // jit_value_t jit_ins_atan2_f64( const jit_value_t& lhs , const jit_value_t& rhs  );


  // Select
  //jit_value_t jit_ins_selp( const jit_value_t& lhs , const jit_value_t& rhs , const jit_value_t& p );

  void jit_ins_mov( jit_value_t& dest , const jit_value_t& src  );
  void jit_ins_mov( jit_value_t& dest , const std::string& src  );

  void jit_ins_branch( jit_block_t& block );
  void jit_ins_branch( const jit_value_t& cond,  jit_block_t& block_true , jit_block_t& block_false );

  jit_block_t jit_ins_start_new_block();
  void jit_ins_start_block(  jit_block_t& label );
  void jit_ins_comment(  const char * comment );
  void jit_ins_exit();
  void jit_ins_cond_exit( const jit_value_t& cond );

  jit_value_t jit_int_array_indirection( const jit_value_t& idx , jit_llvm_builtin type );

  jit_value_t jit_ins_load ( const jit_value_t& base , 
			     const jit_value_t& offset , 
			     jit_llvm_type type  );

  void jit_ins_store( const jit_value_t& base , 
		      const jit_value_t& offset , 
		      jit_llvm_type type , 
		      const jit_value_t& val  );

  jit_value_t jit_geom_get_linear_th_idx();

  jit_value_t jit_ins_phi( const jit_value_t& v0 , jit_block_t& b0 ,
			   const jit_value_t& v1 , jit_block_t& b1 );

  class JitOp {
  protected:
    virtual std::ostream& writeToStream( std::ostream& stream ) const = 0;
    jit_llvm_type args_type;
  public:
    JitOp( const jit_value_t& lhs , const jit_value_t& rhs ) :
      args_type( jit_type_promote( lhs->get_type() , rhs->get_type() ) ) {}
    jit_llvm_type getArgsType() const { return args_type; }
    virtual jit_llvm_type getDestType() const { return this->getArgsType(); }
    friend std::ostream& operator<< (std::ostream& stream, const JitOp& op) {
      return op.writeToStream(stream);
    }
  };

  class JitOpAdd: public JitOp {
  public:
    JitOpAdd( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fadd" : "add";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpSub: public JitOp {
  public:
    JitOpSub( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fsub" : "sub";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpMul: public JitOp {
  public:
    JitOpMul( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fmul" : "mul";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpDiv: public JitOp {
  public:
    JitOpDiv( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = jit_type_is_float( getDestType() ) ? "fdiv" : "sdiv";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  // class JitOpMulWide: public JitOp {
  // public:
  //   JitOpMulWide( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
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
    JitOpSHL( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "shl";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpSHR: public JitOp {
  public:
    JitOpSHR( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "shr";
      stream << op << " " << getDestType();
      return stream;
    }
  };



  class JitOpNE: public JitOp {
  public:
    JitOpNE( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
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
    JitOpEQ( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
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
    JitOpLE( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
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
    JitOpLT( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
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
    JitOpGE( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
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
    JitOpGT( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
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
    JitOpAnd( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "and";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpOr: public JitOp {
  public:
    JitOpOr( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "or";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpXOr: public JitOp {
  public:
    JitOpXOr( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string op = "xor";
      stream << op << " " << getDestType();
      return stream;
    }
  };

  class JitOpRem: public JitOp {
  public:
    JitOpRem( const jit_value_t& lhs , const jit_value_t& rhs ): JitOp(lhs,rhs) {}
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
