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

namespace QDP {

  //enum jit_ptx_type { f32=0,f64=1,u16=2,u32=3,u64=4,s16=5,s32=6,s64=7,u8=8,b16=9,b32=10,b64=11,pred=12 };

  enum class jit_ptx_type { f32,f64,u16,u32,u64,s16,s32,s64,u8,b16,b32,b64,pred};


  enum class jit_state_space { 
    state_default , 
      state_global , 
      state_local , 
      state_shared 
      };

  std::ostream& operator<< (std::ostream& stream, const jit_state_space& space );
  std::ostream& operator<< (std::ostream& stream, const jit_ptx_type& type );
    

  //
  // MATCHING C TYPES TO PTX TYPES
  //
  template<class T> struct jit_type {};
  template<> struct jit_type<float>            { static constexpr jit_ptx_type value = jit_ptx_type::f32; };
  template<> struct jit_type<double>           { static constexpr jit_ptx_type value = jit_ptx_type::f64; };
  template<> struct jit_type<int>              { static constexpr jit_ptx_type value = jit_ptx_type::s32; };
  template<> struct jit_type<bool>             { static constexpr jit_ptx_type value = jit_ptx_type::pred; };



  class jit_function;
  class jit_label;
  class jit_value;
  class jit_value_reg;
  typedef std::shared_ptr<jit_function>          jit_function_t;
  typedef std::shared_ptr<jit_label>             jit_label_t;


  template<class T>
    std::shared_ptr<T> get(const jit_value & pA) {
    return std::dynamic_pointer_cast< T >( pA );
  }


  const char * jit_get_ptx_type( jit_ptx_type type );
  const char * jit_get_ptx_letter( jit_ptx_type type );
  const char * jit_get_mul_specifier_lo_str( jit_ptx_type type );
  const char * jit_get_div_specifier( jit_ptx_type type );
  const char * jit_get_identifier_local_memory();

  namespace PTX {
    std::map< jit_ptx_type , std::array<const char*,4> > create_ptx_type_matrix(int cc);
    extern std::map< jit_ptx_type , std::array<const char*,4> > ptx_type_matrix;
    extern const std::map< jit_ptx_type , std::map<jit_ptx_type,jit_ptx_type> > map_promote;
    extern const std::map< jit_ptx_type , jit_ptx_type >               map_wide_promote;
  }


  class jit_function {
    std::ostringstream oss_prg;
    std::ostringstream oss_signature;
    std::ostringstream oss_reg_defs;
    typedef std::map<jit_ptx_type,int> RegCountMap;
    RegCountMap reg_count;
    typedef std::vector<std::pair<jit_ptx_type,int> > LocalCountVec;
    LocalCountVec vec_local_count;
    int param_count;
    int local_count;
    bool m_shared;
    std::vector<bool> m_include_math_ptx_unary;
    std::vector<bool> m_include_math_ptx_binary;
  public:
    std::string get_kernel_as_string();
    void set_include_math_ptx_unary(int i) { 
      assert(m_include_math_ptx_unary.size()>i); 
      m_include_math_ptx_unary.at(i) = true; 
    }
    void set_include_math_ptx_binary(int i) { 
      assert(m_include_math_ptx_binary.size()>i); 
      m_include_math_ptx_binary.at(i) = true; 
    }
    void emitShared();
    int local_alloc( jit_ptx_type type, int count );
    void write_reg_defs();
    int get_param_count();
    void inc_param_count();
    jit_function();
    int reg_alloc( jit_ptx_type type );
    std::ostringstream& get_prg();
    std::ostringstream& get_signature();
  };

  extern jit_function_t jit_internal_function;

  void jit_start_new_function();
  jit_function_t jit_get_function();
  std::string jit_get_kernel_as_string();
  CUfunction jit_get_cufunction(const char* fname);
  

  // class jit_function_singleton
  // {
  //   static jit_function singleton {
  //   return singleton;
  //   };





  class jit_label {
    static int count;
    int count_m;
  public:
    jit_label() {
      count_m = count++;
    }
    friend std::ostream& operator<< (std::ostream& stream, const jit_label& lab) {
      stream << "L" << lab.count_m;
      return stream;
    }
  };



  class jit_value {
  public:
    jit_value( const jit_value& rhs );

    explicit jit_value( jit_ptx_type type_ );

    jit_value( int val );
    jit_value( double val );
    explicit jit_value( size_t val );

    // jit_value( double val )
    
    void reg_alloc();
    void assign( const jit_value& rhs );
    jit_value& operator=( const jit_value& rhs );    
    ~jit_value() {}

    jit_ptx_type get_type() const;
    void set_state_space( jit_state_space space );
    jit_state_space get_state_space() const;
    
    //int get_number() const;
    std::string get_name() const;
    bool get_ever_assigned() const { return ever_assigned; }
    void set_ever_assigned() { ever_assigned = true; }
  private:
    bool ever_assigned;
    jit_state_space mem_state;
    jit_ptx_type type;
    int number;
  };


  const char * get_state_space_str( jit_state_space space );


  struct IndexRet {
    IndexRet():
      r_newidx_local(jit_ptx_type::s32),
      r_newidx_buffer(jit_ptx_type::s32),
      r_pred_in_buf(jit_ptx_type::pred),
      r_rcvbuf(jit_ptx_type::u64)
    {}
    jit_value r_newidx_local;
    jit_value r_newidx_buffer;
    jit_value r_pred_in_buf;
    jit_value r_rcvbuf;
  };

  
  



  
  std::string jit_predicate( const jit_value& pred );

  jit_state_space jit_state_promote( jit_state_space ss0 , jit_state_space ss1 );
  jit_ptx_type jit_type_promote(jit_ptx_type t0,jit_ptx_type t1);

  jit_ptx_type jit_bit_type(jit_ptx_type type);
  jit_ptx_type jit_type_wide_promote(jit_ptx_type t0);

  void jit_ins_bar_sync( int a );

  jit_value jit_add_param( jit_ptx_type type );
  jit_value jit_allocate_local( jit_ptx_type type , int count );
  jit_value jit_get_shared_mem_ptr();


  //jit_function_t jit_get_valid_func( jit_function_t f0 ,jit_function_t f1 );

  std::string jit_get_reg_name( const jit_value& val );

  jit_label_t jit_label_create();

  // jit_value_reg_t jit_val_create_new( int type );
  // jit_value_reg_t jit_val_create_from_const( int type , int const_val , jit_value pred=jit_value() );
  // jit_value_reg_t jit_val_create_convert( int type , jit_value val , jit_value pred=jit_value() );
  // jit_value jit_val_create_copy( jit_value val , jit_value pred=jit_value() );


  jit_value jit_val_convert( jit_ptx_type type , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred));


  jit_value jit_geom_get_tidx( );
  jit_value jit_geom_get_ntidx( );
  jit_value jit_geom_get_ctaidx( );

  // Binary operations
  jit_value jit_ins_mul_wide( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_mul( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_div( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_add( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_sub( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_shl( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_shr( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_and( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_or ( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_xor( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_rem( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );

  // ni
  jit_value jit_ins_mod( const jit_value& lhs , const jit_value& rhs );

  // Binary operations returning predicate
  jit_value jit_ins_lt( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_ne( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_eq( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_ge( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_le( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_gt( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );

  // Native PTX Unary operations
  jit_value jit_ins_neg( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_not( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_fabs( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_floor( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_sqrt( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );

  // Imported PTX Unary operations single precision
  jit_value jit_ins_sin_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_acos_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_asin_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_atan_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_ceil_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_cos_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_cosh_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_exp_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_log_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_log10_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_sinh_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_tan_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_tanh_f32( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );

  // Imported PTX Binary operations single precision
  jit_value jit_ins_pow_f32( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_atan2_f32( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );

  // Imported PTX Unary operations double precision
  jit_value jit_ins_sin_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_acos_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_asin_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_atan_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_ceil_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_cos_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_cosh_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_exp_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_log_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_log10_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_sinh_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_tan_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_tanh_f64( const jit_value& lhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );

  // Imported PTX Binary operations single precision
  jit_value jit_ins_pow_f64( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  jit_value jit_ins_atan2_f64( const jit_value& lhs , const jit_value& rhs , const jit_value& pred=jit_value(jit_ptx_type::pred) );


  // Select
  jit_value jit_ins_selp( const jit_value& lhs , const jit_value& rhs , const jit_value& p );

  void jit_ins_mov( jit_value& dest , const jit_value& src , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  void jit_ins_mov( jit_value& dest , const std::string& src , const jit_value& pred=jit_value(jit_ptx_type::pred) );

  void jit_ins_branch( jit_label_t& label , const jit_value& pred=jit_value(jit_ptx_type::pred) );
  void jit_ins_label(  jit_label_t& label );
  void jit_ins_comment(  const char * comment );
  void jit_ins_exit( const jit_value& pred=jit_value(jit_ptx_type::pred) );

  jit_value jit_ins_load ( const jit_value& base , int offset , jit_ptx_type type , const jit_value& pred=jit_value(jit_ptx_type::pred) );


  void jit_ins_store( const jit_value& base , int offset , jit_ptx_type type , const jit_value& val , const jit_value& pred=jit_value(jit_ptx_type::pred) );

  jit_value jit_geom_get_linear_th_idx();


  class JitOp {
  protected:
    virtual std::ostream& writeToStream( std::ostream& stream ) const = 0;
    jit_ptx_type args_type;
  public:
    JitOp( const jit_value& lhs , const jit_value& rhs ) {
      args_type = jit_type_promote( lhs.get_type() , rhs.get_type() );
    }
    jit_ptx_type getArgsType() const { return args_type; }
    virtual jit_ptx_type getDestType() const { return this->getArgsType(); }
    friend std::ostream& operator<< (std::ostream& stream, const JitOp& op) {
      return op.writeToStream(stream);
    }
  };

  class JitOpAdd: public JitOp {
  public:
    JitOpAdd( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "add." 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
  };

  class JitOpSub: public JitOp {
  public:
    JitOpSub( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "sub." 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
  };

  class JitOpMul: public JitOp {
  public:
    JitOpMul( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "mul." 
	     << jit_get_mul_specifier_lo_str( getDestType() ) 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
  };

  class JitOpDiv: public JitOp {
  public:
    JitOpDiv( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "div." 
	     << jit_get_div_specifier( getDestType() ) 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
  };

  class JitOpMulWide: public JitOp {
  public:
    JitOpMulWide( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual jit_ptx_type getDestType() const { 
      return jit_type_wide_promote( getArgsType() );
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string specifier_wide = 
	getArgsType() != getDestType() ? 
	"wide.":
	jit_get_mul_specifier_lo_str( getDestType() );
      stream << "mul." 
	     << specifier_wide
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
  };

  class JitOpSHL: public JitOp {
  public:
    JitOpSHL( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "shl."
	     << jit_get_ptx_type( jit_bit_type( getDestType() ) );
      return stream;
    }
  };

  class JitOpSHR: public JitOp {
  public:
    JitOpSHR( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "shr."
	     << jit_get_ptx_type( jit_bit_type( getDestType() ) );
      return stream;
    }
  };



  class JitOpLT: public JitOp {
  public:
    JitOpLT( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual jit_ptx_type getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.lt."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
  };

  class JitOpNE: public JitOp {
  public:
    JitOpNE( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual jit_ptx_type getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.ne."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
  };

  class JitOpEQ: public JitOp {
  public:
    JitOpEQ( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual jit_ptx_type getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.eq."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
  };

  class JitOpGE: public JitOp {
  public:
    JitOpGE( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual jit_ptx_type getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.ge."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
  };

  class JitOpLE: public JitOp {
  public:
    JitOpLE( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual jit_ptx_type getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.le."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
  };

  class JitOpGT: public JitOp {
  public:
    JitOpGT( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual jit_ptx_type getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.gt."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
  };

  class JitOpAnd: public JitOp {
  public:
    JitOpAnd( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "and." 
	     << jit_get_ptx_type( jit_bit_type( getDestType() ) );
      return stream;
    }
  };

  class JitOpOr: public JitOp {
  public:
    JitOpOr( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "or." 
	     << jit_get_ptx_type( jit_bit_type( getDestType() ) );
      return stream;
    }
  };

  class JitOpXOr: public JitOp {
  public:
    JitOpXOr( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "xor." 
	     << jit_get_ptx_type( jit_bit_type( getDestType() ) );
      return stream;
    }
  };

  class JitOpRem: public JitOp {
  public:
    JitOpRem( const jit_value& lhs , const jit_value& rhs ): JitOp(lhs,rhs) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "rem." 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
  };






  class JitUnaryOp {
  protected:
    virtual std::ostream& writeToStream( std::ostream& stream ) const = 0;
    jit_ptx_type type;
  public:
    JitUnaryOp( jit_ptx_type type_ ): type(type_) {}
    friend std::ostream& operator<< (std::ostream& stream, const JitUnaryOp& op) {
      return op.writeToStream(stream);
    }
  };

  class JitUnaryOpNeg: public JitUnaryOp {
  public:
    JitUnaryOpNeg( jit_ptx_type type_ ): JitUnaryOp(type_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "neg."
	     << jit_get_ptx_type( type );
      return stream;
    }
  };

  class JitUnaryOpNot: public JitUnaryOp {
  public:
    JitUnaryOpNot( jit_ptx_type type_ ): JitUnaryOp(type_) {
      assert( type_ == jit_ptx_type::pred );
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "not."
	     << jit_get_ptx_type( type );
      return stream;
    }
  };

  class JitUnaryOpAbs: public JitUnaryOp {
  public:
    JitUnaryOpAbs( jit_ptx_type type_ ): JitUnaryOp(type_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "abs."
	     << jit_get_ptx_type( type );
      return stream;
    }
  };

  class JitUnaryOpFloor: public JitUnaryOp {
  public:
    JitUnaryOpFloor( jit_ptx_type type_ ): JitUnaryOp(type_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "cvt.rmi."
	     << jit_get_ptx_type( type )
	     << "."
	     << jit_get_ptx_type( type );
      return stream;
    }
  };

  class JitUnaryOpSqrt: public JitUnaryOp {
  public:
    JitUnaryOpSqrt( jit_ptx_type type_ ): JitUnaryOp(type_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "sqrt.";
      if ( DeviceParams::Instance().getMajor() >= 2 )
	stream << "rn.";
      else
	stream << "approx.";
      stream << jit_get_ptx_type( type );
      return stream;
    }
  };

  class JitUnaryOpCeil: public JitUnaryOp {
  public:
    JitUnaryOpCeil( jit_ptx_type type_ ): JitUnaryOp(type_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "cvt.rpi."
	     << jit_get_ptx_type( type )
	     << "."
	     << jit_get_ptx_type( type );
      return stream;
    }
  };


}

#endif
