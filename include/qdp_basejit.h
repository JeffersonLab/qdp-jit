#ifndef QDP_BASEJIT_H
#define QDP_BASEJIT_H

namespace QDP {

  template<class T>
  class BaseJIT
  {
    llvm::Value *     m_base = nullptr;
    NestedIndex index;
    JitDataFormat data_format_;
    Shape                shape_;

  public:

    BaseJIT()    {  }

    BaseJIT(const BaseJIT& rhs): m_base(rhs.m_base), index(rhs.index), data_format_(rhs.data_format_), shape_(rhs.shape_)
    {
    }


    BaseJIT( Shape shape ) : shape_(shape) {}

    
    BaseJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape ) : m_base(base), index(args) , data_format_(fmt), shape_(shape) {}



    void setup( llvm::Value * base , JitDataFormat fmt , NestedIndex args )
    {
      m_base = base;
      data_format_ = fmt;
      index = args;
    }

    
    void setup( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape )
    {
      m_base = base;
      data_format_ = fmt;
      index = args;
      shape_ = shape;
    }

    virtual void post_setup() {}

    virtual int this_size() const = 0;

    virtual int getN() const = 0;


    
    int words( Shape shape )
    {
      shape_ = shape;
      post_setup();
      
      return this_size() * elem(0).words();
    }


    int words() const
    {
      return this_size() * elem(0).words();
    }
    
    
    void setup( const T& rhs )
    {
      m_base = rhs.get_base();
      index = rhs.get_index();
      data_format_ = rhs.data_format();
      shape_ = rhs.shape();
    }


    

    

    llvm::Value *     get_base() const { return m_base; }
    NestedIndex get_index() const { return index; }
    JitDataFormat data_format() const { return data_format_; }
    Shape shape() const { return shape_; }

    
    
    T elem() const
    {
      NestedIndex idv(index);
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }

    
    T elem(llvm::Value* i) const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(getN()) , i );
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }

    
    T elem(int i) const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(getN()) , llvm_create_value(i) );
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }

    
    T elem( llvm::Value* i , llvm::Value* j ) const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(this_size()) , llvm_add( j , llvm_mul( llvm_create_value( getN() ) , i ) ) );
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }
    
    
    T elem( int i , llvm::Value* j ) const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(this_size()) , llvm_add( j , llvm_mul( llvm_create_value( getN() ) , llvm_create_value(i) ) ) );
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }

    
    T elem( llvm::Value* i , int j ) const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(this_size()) , llvm_add( llvm_create_value(j) , llvm_mul( llvm_create_value( getN() ) , i ) ) );
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }
    

    T elem( int i , int j ) const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(this_size()) , llvm_add( llvm_create_value(j) , llvm_mul( llvm_create_value(getN()) , llvm_create_value(i) ) ) );
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }

    
    T real() const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(this_size()) , llvm_create_value(0) );
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }

    
    T imag() const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(this_size()) , llvm_create_value(1) );
      T F( m_base , data_format_ , idv , shape_ );
      return F;
    }

  };





  template<class T>
  class BaseJIT< WordJIT<T> >
  {
    llvm::Value *     m_base;
    NestedIndex index;
    JitDataFormat data_format_;
    
  public:
    BaseJIT() {}

    BaseJIT( llvm::Value * base , JitDataFormat fmt , NestedIndex args ) : m_base(base), index(args) , data_format_(fmt) {}

    Shape shape() const { return Shape(); }

    void setup( WordJIT<T> rhs )
    {
      m_base = rhs.get_base();
      index = rhs.get_index();
      data_format_ = rhs.data_format();
    }


    void setup( llvm::Value * base , JitDataFormat fmt , NestedIndex args , Shape shape )
    {
      m_base = base;
      data_format_ = fmt;
      index = args;
    }

    
    void setup( llvm::Value * base , JitDataFormat fmt , NestedIndex args )
    {
      m_base = base;
      data_format_ = fmt;
      index = args;
    }

    
    void setup( llvm::Value * base , JitDataFormat fmt )
    {
      m_base = base;
      data_format_ = fmt;
    }


    llvm::Value*      get_base() const { return m_base; }
    NestedIndex get_index() const { return index; }
    JitDataFormat data_format() const { return data_format_; }

    
    int words() const
    {
      return getN();
    }
    
    int words( Shape shape )
    {
      return getN();
    }

    
    
    virtual int getN() const = 0;


    WordJIT<T> elem( int i ) const
    {
      NestedIndex idv(index);
      idv.push_back( llvm_create_value(getN()) , llvm_create_value(i) );
      WordJIT<T> F( m_base , data_format_ , idv );
      return F;
    }

  };

  
  
}

#endif
