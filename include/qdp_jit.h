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
#include<stdlib.h>
#include<assert.h>

namespace QDP {



  enum jit_ptx_type { f32=0,f64=1,u16=2,u32=3,u64=4,s16=5,s32=6,s64=7,u8=8,b16=9,b32=10,b64=11,pred=12 };

  //
  // MATCHING C TYPES TO PTX TYPES
  //
  template<class T> struct jit_type {};
  template<> struct jit_type<float>            { enum { value = jit_ptx_type::f32 }; };
  template<> struct jit_type<double>           { enum { value = jit_ptx_type::f64 }; };
  template<> struct jit_type<int>              { enum { value = jit_ptx_type::s32 }; };

  class jit_function;
  class jit_label;
  class jit_value;
  class jit_value_const;
  class jit_value_const_int;
  class jit_value_const_float;
  class jit_value_reg;
  typedef std::shared_ptr<jit_function>          jit_function_t;
  typedef std::shared_ptr<jit_value>             jit_value_t;
  typedef std::shared_ptr<jit_value_reg>         jit_value_reg_t;
  typedef std::shared_ptr<jit_value_const>       jit_value_const_t;
  typedef std::shared_ptr<jit_value_const_int>   jit_value_const_int_t;
  typedef std::shared_ptr<jit_value_const_float> jit_value_const_float_t;
  typedef std::shared_ptr<jit_label>             jit_label_t;

  struct IndexRet {
    jit_value_t r_newidx_local;
    jit_value_t r_newidx_buffer;
    jit_value_t r_pred_in_buf;
    jit_value_t r_rcvbuf;
  };



  const char * jit_get_ptx_type( int type );
  const char * jit_get_ptx_letter( int type );
  const char * jit_get_mul_specifier_lo_str( int type );
  const char * jit_get_div_specifier( int type );
  const char * jit_get_identifier_local_memory();

  namespace PTX {
    extern const std::map< int , std::map<int,int> > map_promote;
    extern const std::map< int , int >               map_wide_promote;
  }

  class jit_function {
    std::string fname;
    std::ostringstream oss_prg;
    std::ostringstream oss_signature;
    std::ostringstream oss_reg_defs;
    std::vector<int> reg_count;
    std::vector<std::pair<int,int> > vec_local_count;     // pair = (type,count)
    int param_count = 0;
    int local_count = 0;
  public:
    int local_alloc( int type, int count );
    void write_reg_defs();
    void write();
    int get_param_count();
    void inc_param_count();
    jit_function( const char * fname_);
    int reg_alloc( int type );
    std::ostringstream& get_prg();
    std::ostringstream& get_signature();
  };



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
  protected:
    int type;
  public:
    virtual ~jit_value() {}
    enum StateSpace { state_global=0 , state_local=1 , state_shared=2 };
    jit_value( int type_ ) : type(type_) {}
    int get_type() const {return type;}
  };



  class jit_value_reg: public jit_value {
    int number;
    jit_function_t func;
  public:
    ~jit_value_reg() {}
    jit_value_reg(jit_function_t func_, int type_);

    void set_local_state();
    void set_state_space( StateSpace ss );
    StateSpace get_state_space();
    const char * get_state_space_str() const;

    int get_number() const;
    jit_function_t get_func() const;
    std::string get_name() const;
  private:
    StateSpace mem_state = state_global;
  };


  class jit_value_const: public jit_value {
  public:
    jit_value_const( int type ) : jit_value(type) {}
    virtual bool  isInt() const      = 0;
    virtual float getAsFloat() const = 0;
    virtual std::string getAsString() const = 0;
  };



  class jit_value_const_int: public jit_value_const {
    int val;
  public:
    jit_value_const_int( int val_): jit_value_const( jit_type<int>::value ), val(val_) {};
    virtual bool  isInt() const { return true; }
    virtual float getAsFloat() const { return val; }
    virtual std::string getAsString() const { 
      std::ostringstream oss; oss << val;
      return oss.str();
    }
    int getValue() {return val;}
  };



  class jit_value_const_float: public jit_value_const {
    float val;
  public:
    jit_value_const_float( float val_): jit_value_const( jit_type<float>::value ), val(val_) {};
    virtual bool  isInt() const { return false; }
    virtual float getAsFloat() const { return val; }
    virtual std::string getAsString() const { 
      std::ostringstream oss; oss << val;
      return oss.str();
    }
    float getValue() {return val;}
  };



  jit_value::StateSpace jit_propagate_state_space( jit_value::StateSpace ss0 , jit_value::StateSpace ss1 );

  std::string jit_predicate( jit_value_t pred );

  int jit_type_promote(int t0,int t1);
  int jit_type_wide_promote(int t0);

  jit_value_t jit_add_param( jit_function_t func , int type );
  jit_value_t jit_allocate_local( jit_function_t func , int type , int count );

  jit_function_t jit_create_function(const char * fname_);
  jit_function_t jit_get_valid_func( jit_function_t f0 ,jit_function_t f1 );

  std::string jit_get_reg_name( jit_value_t val );

  jit_label_t jit_label_create( jit_function_t func );

  jit_value_reg_t jit_val_create_new( jit_function_t func , int type );
  jit_value_reg_t jit_val_create_from_const( jit_function_t func , int type , int const_val , jit_value_t pred=jit_value_t() );
  jit_value_reg_t jit_val_create_convert( jit_function_t func , int type , jit_value_t val , jit_value_t pred=jit_value_t() );
  jit_value_t jit_val_create_copy( jit_value_t val , jit_value_t pred=jit_value_t() );
  jit_value_const_t jit_val_create_const_int( int val );
  jit_value_const_t jit_val_create_const_float( float val );

  jit_value_t jit_geom_get_tidx( jit_function_t func );
  jit_value_t jit_geom_get_ntidx( jit_function_t func );
  jit_value_t jit_geom_get_ctaidx( jit_function_t func );

  // Binary operations
  jit_value_t jit_ins_mul_wide( jit_value_t lhs , jit_value_t rhs , jit_value_t pred=jit_value_t() );
  jit_value_t jit_ins_mul( jit_value_t lhs , jit_value_t rhs , jit_value_t pred=jit_value_t() );
  jit_value_t jit_ins_div( jit_value_t lhs , jit_value_t rhs , jit_value_t pred=jit_value_t() );
  jit_value_t jit_ins_add( jit_value_t lhs , jit_value_t rhs , jit_value_t pred=jit_value_t() );
  jit_value_t jit_ins_sub( jit_value_t lhs , jit_value_t rhs , jit_value_t pred=jit_value_t() );

  // ni
  jit_value_t jit_ins_or( jit_value_t lhs , jit_value_t rhs );
  jit_value_t jit_ins_and( jit_value_t lhs , jit_value_t rhs );
  jit_value_t jit_ins_shl( jit_value_t lhs , jit_value_t rhs );
  jit_value_t jit_ins_shr( jit_value_t lhs , jit_value_t rhs );
  jit_value_t jit_ins_xor( jit_value_t lhs , jit_value_t rhs );
  jit_value_t jit_ins_mod( jit_value_t lhs , jit_value_t rhs );

  // Binary operations returning predicate
  jit_value_t jit_ins_lt( jit_value_t lhs , jit_value_t rhs , jit_value_t pred=jit_value_t() );
  jit_value_t jit_ins_ne( jit_value_t lhs , jit_value_t rhs , jit_value_t pred=jit_value_t() );
  jit_value_t jit_ins_eq( jit_value_t lhs , jit_value_t rhs , jit_value_t pred=jit_value_t() );

  // Unary operations
  jit_value_t jit_ins_neg( jit_value_t lhs , jit_value_t pred=jit_value_t() );

  void jit_ins_mov_no_create( jit_value_t dest , jit_value_t src , jit_value_t pred=jit_value_t() );

  void jit_ins_branch( jit_function_t func , jit_label_t& label , jit_value_t pred=jit_value_t() );
  void jit_ins_label(  jit_function_t func , jit_label_t& label );
  void jit_ins_comment(  jit_function_t func , const char * comment );
  void jit_ins_exit( jit_function_t func , jit_value_t pred=jit_value_t() );

  jit_value_t jit_ins_load ( jit_value_t base , int offset , int type , jit_value_t pred=jit_value_t() );
  void        jit_ins_store( jit_value_t base , int offset , int type , jit_value_t val , jit_value_t pred=jit_value_t() );

  jit_value_t jit_geom_get_linear_th_idx( jit_function_t func );


  class JitOp {
  protected:
    virtual std::ostream& writeToStream( std::ostream& stream ) const = 0;
    int type_lhs;
    int type_rhs;
  public:
    JitOp( int type_lhs_ , int type_rhs_ ): type_lhs(type_lhs_), type_rhs(type_rhs_) {}
    int getArgsType() const { return jit_type_promote( type_lhs , type_rhs ); };
    virtual int getDestType() const { return this->getArgsType(); }
    virtual float operator()(float f0, float f1) const = 0;
    virtual int operator()(int i0, int i1) const = 0;
    friend std::ostream& operator<< (std::ostream& stream, const JitOp& op) {
      return op.writeToStream(stream);
    }
  };

  class JitOpAdd: public JitOp {
  public:
    JitOpAdd( int type_lhs_ , int type_rhs_ ): JitOp(type_lhs_,type_rhs_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "add." 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
    virtual float operator()(float f0, float f1) const { return f0+f1; }
    virtual int operator()(int i0, int i1) const { return i0+i1; }
  };

  class JitOpSub: public JitOp {
  public:
    JitOpSub( int type_lhs_ , int type_rhs_ ): JitOp(type_lhs_,type_rhs_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "sub." 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
    virtual float operator()(float f0, float f1) const { return f0-f1; }
    virtual int operator()(int i0, int i1) const { return i0-i1; }
  };

  class JitOpMul: public JitOp {
  public:
    JitOpMul( int type_lhs_ , int type_rhs_ ): JitOp(type_lhs_,type_rhs_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "mul." 
	     << jit_get_mul_specifier_lo_str( getDestType() ) 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
    virtual float operator()(float f0, float f1) const { return f0*f1; }
    virtual int operator()(int i0, int i1) const { return i0*i1; }
  };

  class JitOpDiv: public JitOp {
  public:
    JitOpDiv( int type_lhs_ , int type_rhs_ ): JitOp(type_lhs_,type_rhs_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "div." 
	     << jit_get_div_specifier( getDestType() ) 
	     << jit_get_ptx_type( getDestType() );
      return stream;
    }
    virtual float operator()(float f0, float f1) const { return f0*f1; }
    virtual int operator()(int i0, int i1) const { return i0*i1; }
  };

  class JitOpMulWide: public JitOp {
  public:
    JitOpMulWide( int type_lhs_ , int type_rhs_ ): JitOp(type_lhs_,type_rhs_) {}
    virtual int getDestType() const { 
      std::cout << "IN WIDE\n";
      return jit_type_wide_promote( getArgsType() );
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      std::string specifier_wide = 
	jit_type_wide_promote( getDestType() ) != getDestType() ? 
	"wide.":
	jit_get_mul_specifier_lo_str( getDestType() );
      stream << "mul." 
	     << specifier_wide
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
    virtual float operator()(float f0, float f1) const { return f0*f1; }
    virtual int operator()(int i0, int i1) const { return i0*i1; }
  };

  class JitOpLT: public JitOp {
  public:
    JitOpLT( int type_lhs_ , int type_rhs_ ): JitOp(type_lhs_,type_rhs_) {}
    virtual int getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.lt."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
    virtual float operator()(float f0, float f1) const { return 0; }
    virtual int operator()(int i0, int i1) const { return 0; }
  };

  class JitOpNE: public JitOp {
  public:
    JitOpNE( int type_lhs_ , int type_rhs_ ): JitOp(type_lhs_,type_rhs_) {}
    virtual int getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.ne."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
    virtual float operator()(float f0, float f1) const { return 0; }
    virtual int operator()(int i0, int i1) const { return 0; }
  };

  class JitOpEQ: public JitOp {
  public:
    JitOpEQ( int type_lhs_ , int type_rhs_ ): JitOp(type_lhs_,type_rhs_) {}
    virtual int getDestType() const {
      return jit_ptx_type::pred;
    }
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "setp.eq."
	     << jit_get_ptx_type( getArgsType() );
      return stream;
    }
    virtual float operator()(float f0, float f1) const { return 0; }
    virtual int operator()(int i0, int i1) const { return 0; }
  };






  class JitUnaryOp {
  protected:
    virtual std::ostream& writeToStream( std::ostream& stream ) const = 0;
    int type;
  public:
    JitUnaryOp( int type_ ): type(type_) {}
    virtual float operator()(float f0) const = 0;
    virtual int operator()(int i0) const = 0;
    friend std::ostream& operator<< (std::ostream& stream, const JitUnaryOp& op) {
      return op.writeToStream(stream);
    }
  };

  class JitUnaryOpNeg: public JitUnaryOp {
  public:
    JitUnaryOpNeg( int type_ ): JitUnaryOp(type_) {}
    virtual std::ostream& writeToStream( std::ostream& stream ) const {
      stream << "neg."
	     << jit_get_ptx_type( type );
      return stream;
    }
    virtual float operator()(float f0) const { return -f0; }
    virtual int operator()(int i0) const { return -i0; }
  };


}

#endif
