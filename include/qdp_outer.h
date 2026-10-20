#ifndef QDP_OUTER_H
#define QDP_OUTER_H

#include "qdp_config.h"


namespace QDP {


  template<class T>
  class OScalar: public QDPType<T, OScalar<T> >
  {
    Shape shape_;
    typedef typename WordType<T>::Type_t    WT;
    WT* f = nullptr;
    bool owns_memory_  = false;

  public:
    typedef T SubType_t;

    Shape shape() const { return shape_; }


    explicit OScalar(const Shape& v): shape_(fit_shape<T>(v))
    {
      alloc_mem();

#ifdef QDP_DEEP_LOG
      zero_me();
#endif
    }
    
    OScalar(): shape_( fit_shape<T>(Shape::current()) )
    {
      alloc_mem();

#ifdef QDP_DEEP_LOG
      zero_me();
#endif
    }

    void zero_me()
    {
      for ( int i = 0 ; i < WordSize<T>::value(shape()) ; ++i )
	f[i] = (WT)0;
    }

    void print() const
    {
      for ( int i = 0 ; i < WordSize<T>::value(shape()) ; ++i )
	std::cout << f[i] << " ";
      std::cout << std::endl;
    }


    typename WordType<T>::Type_t get_word_value() const
    {
      assert(f);
      return *f;
    }

    
    ~OScalar()
    {
      free_mem();
    }

    
    OScalar(const OScalar& rhs): shape_(rhs.shape_) 
    {
      assert(rhs.f);

      alloc_mem();

      copy_words( rhs );
    }


    OScalar(OScalar&& rhs) noexcept:
      shape_(rhs.shape_),
      f(rhs.f),
      owns_memory_(rhs.owns_memory_)
    {
      rhs.f = nullptr;
      rhs.owns_memory_ = false;
    }


    OScalar(const Shape& v, const OScalar& rhs): shape_(fit_shape<T>(v))
    {
      check_shape(rhs);
      alloc_mem();
      this->assign(rhs);
    }

    
    OScalar(const typename WordType<T>::Type_t& rhs): shape_( fit_shape<T>(Shape::current()) )
    {
      alloc_mem();

      typedef OScalar<typename InternalScalar<T>::Type_t>  Scalar_t;

      // since InternalScalar, no need to pass dynamic sizes
      Scalar_t scalar;
      *scalar.get_ptr() = rhs;

      this->assign( scalar );
    }


    OScalar(const Shape& shape, const typename WordType<T>::Type_t& rhs): shape_(fit_shape<T>(shape))
    {
      alloc_mem();

      typedef OScalar<typename InternalScalar<T>::Type_t>  Scalar_t;

      // since InternalScalar, no need to pass dynamic sizes
      Scalar_t scalar;
      *scalar.get_ptr() = rhs;

      this->assign( scalar );
    }

    

    OScalar(const Zero& rhs): shape_( fit_shape<T>(Shape::current()) )
    {
      alloc_mem();

      this->assign(rhs);
    }


    OScalar(const Shape& shape,const Zero& rhs): shape_(fit_shape<T>(shape))
    {
      alloc_mem();

      this->assign(rhs);
    }

    

    template<class T1>
    OScalar(const OScalar<T1>& rhs): shape_(fit_shape_from<T,T1>(rhs.shape(), Shape::current()))
    {
      alloc_mem();
      
      this->assign(rhs);
    }


    template<class RHS, class T1>
    OScalar(const QDPExpr<RHS, OScalar<T1> >& rhs):
      shape_(fit_shape_from<T,T1>(shape_of(rhs), Shape::current()))
    {
      
      alloc_mem();
      
      this->assign(rhs);
    }


    inline
    OScalar& operator=(const typename WordType<T>::Type_t& rhs)
    {
      ensure_mem();
      return this->assign(rhs);
    }

    
    inline
    OScalar& operator=(const Zero& rhs)
    {
      ensure_mem();
      return this->assign(rhs);
    }

    
    template<class T1,class C1>
    inline
    OScalar& operator=(const QDPType<T1,C1>& rhs)
    {
      if ( !shape_accepts_rhs( shape_ , rhs ) )
	{
	  free_mem();
	  shape_ = rhs.shape();
	  alloc_mem();
	}
      ensure_mem();
	
      return this->assign(rhs);
    }

    
    template<class T1,class C1>
    inline
    OScalar& operator=(const QDPExpr<T1,C1>& rhs)
    {
      Shape tmp = shape_of(rhs);
      if ( !shape_accepts_rhs( shape_ , rhs ) )
	{
	  free_mem();
	  shape_ = tmp;
	  alloc_mem();
	}
      ensure_mem();

      return this->assign(rhs);
    }


    inline
    OScalar& operator=(const OScalar& rhs)
    {
      if ( this == &rhs )
	return *this;

      if ( rhs.shape() != shape_ )
	{
	  free_mem();
	  shape_ = rhs.shape();
	  alloc_mem();
	}
      ensure_mem();

      copy_words( rhs );
      return *this;
    }


    inline
    OScalar& operator=(OScalar&& rhs) noexcept
    {
      std::swap( shape_ , rhs.shape_ );
      std::swap( f , rhs.f );
      std::swap( owns_memory_ , rhs.owns_memory_ );
      return *this;
    }


    template<class RHS>
    void check_shape(const RHS& rhs) const
    {
      check_shape_accepts_rhs("OScalar", shape_, rhs);
    }


    
    void alloc_mem()
    {
      int w = WordSize<T>::value(shape());
      f = new WT[w];
      owns_memory_ = true;
    }
    
    void free_mem()
    {
      if (owns_memory_) delete[] f;
      f = nullptr;
      owns_memory_ = false;
    }

    void ensure_mem()
    {
      if (!f)
	alloc_mem();
    }

    void copy_words( const OScalar& rhs )
    {
      assert(rhs.f);
      std::copy( rhs.f , rhs.f + sizeInWords() , f );
    }

    

    T elem() const
    {
      assert(f);
      T F( shape() , f );
      return F;
    }

    const typename WordType<T>::Type_t* get_ptr() const
    {
      assert(f);
      return f;
    }

    typename WordType<T>::Type_t* get_ptr() 
    {
      assert(f);
      return f;
    }


    int sizeInWords() const
    {
      return WordSize<T>::value( shape() );
    }
    
    void writeTo(std::ostream& out) const
    {
      assert(f);
      out.write( (const char*)f , sizeof(WT) * sizeInWords() );
    }

    void readFrom(std::istream& in) const
    {
      assert(f);
      in.read( (char*)f , sizeof(WT) * sizeInWords() );
    }

    
  };


  
template<class T>
istream& operator>>(istream& s, OScalar<T>& d)
{
  return s >> d.elem();
}

  
template<class T>
inline
ostream& operator<<(ostream& s, const OScalar<T>& d)
{
  return s << d.elem();
}



template<class RHS, class T1>
ostream& operator<<(ostream& s, const QDPExpr<RHS, OScalar<T1> >& l)
{
  typedef OScalar<T1> C1;
  return s << C1(l);
}


template<class T>
TextReader& operator>>(TextReader& txt, OScalar<T>& d)
{
  return txt >> d.elem();
}


template<class T>
StandardInputStream& operator>>(StandardInputStream& is, OScalar<T>& d)
{
  return is >> d.elem();
}


template<class T>
inline
TextWriter& operator<<(TextWriter& txt, const OScalar<T>& d)
{
  return txt << d.elem();
}


template<class T>
StandardOutputStream& operator<<(StandardOutputStream& s, const OScalar<T>& d)
{
  return s << d.elem();
}


template<class RHS, class T1>
StandardOutputStream& operator<<(StandardOutputStream& s, const QDPExpr<RHS, OScalar<T1> >& l)
{
  typedef OScalar<T1> C1;
  return s << C1(l);
}

#ifndef QDP_NO_LIBXML2
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const OScalar<T>& d)
{
  return xml << d.elem();
}

template<class T>
inline
void read(XMLReader& xml, const string& path, OScalar<T>& d)
{
  read(xml, path, d.elem());
}
#endif


struct ElemLeaf
{
  inline ElemLeaf() { }
};




  template<class T> 
  class OLattice: public QDPType<T, OLattice<T> >
  {
    Shape shape_;
    JitDataFormat data_format_ = default_data_format();
    
    size_t site_words_;
    int cache_id_ = -1;
    LatticeLayout layout_;
    bool owns_memory_ = false;
    
  public:

    typedef T SubType_t;

    Shape shape() const { return shape_; }


    JitDataFormat data_format() const { return data_format_; }

    static JitDataFormat storable_data_format( JitDataFormat fmt )
    {
      if constexpr ( !std::is_floating_point_v< typename WordType<T>::Type_t > )
	fmt.storage = JitStorage::Native;
      return fmt;
    }

    static JitDataFormat default_data_format()
    {
      return storable_data_format( qdp_jit_config_get_default_data_format() );
    }

    static JitDataFormat checked_data_format( JitDataFormat fmt )
    {
      if constexpr ( !std::is_floating_point_v< typename WordType<T>::Type_t > )
	if ( fmt.half() )
	  QDP_error_exit( "OLattice: half storage requires a floating-point word type" );
      return fmt;
    }

   
    template<class T1>
    OLattice(const QDPSubType< T1 , OLattice<T> >& rhs) = delete;

    
    explicit OLattice(const Shape& v):
      shape_(fit_shape<T>(v))
    {
      alloc_mem();
    }


    OLattice(LatticeLayout layout, const Shape& v):
      shape_(fit_shape<T>(v))
    {
      alloc_mem(layout);
    }


    OLattice():
      shape_( fit_shape<T>(Shape::current()) )
    {
      alloc_mem();
    }

    
    explicit OLattice( JitDataFormat _data_format ):
      shape_( fit_shape<T>(Shape::current()) ),
      data_format_( checked_data_format(_data_format) )
    {
      alloc_mem();
    }

    
    OLattice(const OLattice& rhs):
      shape_(rhs.shape_),
      data_format_( rhs.data_format_ )
    {
      alloc_mem( rhs.layout() );
#ifdef QDP_DEEP_LOG
      bool log_status = jit_config_deep_log();
      if (log_status)
	jit_config_deep_log_stop();
#endif
      this->assign(rhs);
#ifdef QDP_DEEP_LOG
      if (log_status)
	jit_config_deep_log_start();
#endif
    }

    template<class T1>
    OLattice(const OScalar<T1>& rhs):
      shape_(fit_shape_from<T,T1>(rhs.shape(), Shape::current()))
    {
      alloc_mem();
      this->assign(rhs);
    }

    template<class T1>
    OLattice(const OLattice<T1>& rhs):
      shape_(fit_shape_from<T,T1>(rhs.shape(), Shape::current())),
      data_format_( storable_data_format(rhs.data_format()) )
    {
      alloc_mem(rhs.layout());
      this->assign(rhs);
    }


    template<class RHS, class T1>
    OLattice(const QDPExpr<RHS, OLattice<T1> >& rhs):
      shape_(fit_shape_from<T,T1>(shape_of(rhs), Shape::current()))
    {
      alloc_mem(get_layout_from_rhs( rhs ));
      this->assign(rhs);
    }

    
    template<class RHS, class T1>
    OLattice(const QDPExpr<RHS, OScalar<T1> >& rhs):
      shape_(fit_shape_from<T,T1>(shape_of(rhs), Shape::current()))
    {
      alloc_mem();
      this->assign(rhs);
    }



    OLattice( typename WordType<T>::Type_t rhs):
      shape_(fit_shape<T>(Shape::current()))
    {
      alloc_mem();

      typedef OScalar<typename InternalScalar<T>::Type_t>  Scalar_t;
      this->assign(Scalar_t(rhs));
    }

    
    OLattice(const Zero& rhs):
      shape_(fit_shape<T>(Shape::current())) 
    {
      alloc_mem();
      this->assign(rhs);
    }

    
    OLattice(const Shape& v, const OLattice& rhs):
      shape_(fit_shape<T>(v)),
      data_format_( rhs.data_format_ )
    {
      check_shape(rhs);
      alloc_mem( rhs.layout() );
      this->assign(rhs);
    }


    OLattice(const Shape& shape, const Zero& rhs):
      shape_(fit_shape<T>(shape))
    {
      alloc_mem();
      this->assign(rhs);
    }


    OLattice(OLattice&& rhs) noexcept
      : shape_(rhs.shape_),
	data_format_(rhs.data_format_),
	site_words_(rhs.site_words_),
	cache_id_(rhs.cache_id_),
	layout_(rhs.layout_),
	owns_memory_(rhs.owns_memory_)
    {
      rhs.cache_id_ = -1;
      rhs.owns_memory_ = false;
    }


    
    ~OLattice()
    {
      free_mem();
    }

    
    
    OLattice& operator=(const typename WordType<T>::Type_t& rhs)
    {
      return this->assign(rhs);
    }


    OLattice& operator=(const Zero& rhs)
    {
      return this->assign(rhs);
    }

    
    template<class T1>
    OLattice& operator=(const OLattice<T1>& rhs)
    {
      check_shape(rhs);

      return this->assign(rhs);
    }


    template<class T1>
    OLattice& operator=(const OScalar<T1>& rhs)
    {
      check_shape(rhs);
      return this->assign(rhs);
    }

    
    template<class T1,class T2>
    OLattice& operator=(const QDPExpr<T1,OScalar<T2> >& rhs)
    {
      check_shape(rhs);
      return this->assign(rhs);
    }

    
    template<class T1,class T2>
    OLattice& operator=(const QDPExpr<T1,OLattice<T2> >& rhs)
    {
      check_shape(rhs);
      return this->assign(rhs);
    }

    

    template<class RHS>
    void check_shape(const RHS& rhs) const
    {
      check_shape_accepts_rhs("OLattice", shape_, rhs);
    }



    
    OLattice& operator=(const OLattice& rhs)
    {
      check_shape(rhs);
      
#ifdef QDP_DEEP_LOG
      bool log_status = jit_config_deep_log();
      if (log_status)
	jit_config_deep_log_stop();
#endif
      this->assign(rhs);
#ifdef QDP_DEEP_LOG
      if (log_status)
	jit_config_deep_log_start();
#endif
      return *this;
    }

    
    OLattice& operator=(OLattice&& rhs)
    {
      if ( this != &rhs && rhs.owns_memory_ &&
	   layout_ == rhs.layout_ && shape_ == rhs.shape_ && data_format_ == rhs.data_format_ )
	{
	  std::swap( cache_id_ , rhs.cache_id_ );
	  std::swap( owns_memory_ , rhs.owns_memory_ );
	  return *this;
	}
      return *this = static_cast<const OLattice&>(rhs);
    }

    
    OSubLattice<T> operator[](const Subset& s)
    {return OSubLattice<T>(*this,const_cast<Subset&>(s));}


    const typename WordType<T>::Type_t* get_ptr() const
    {
      typename WordType<T>::Type_t *f;
      QDP_get_global_cache().getHostPtr( (void**)&f , cache_id_ );

      return f;
    }

    
    T elem(int i) const
    {
      typename WordType<T>::Type_t *f;
      QDP_get_global_cache().getHostPtr( (void**)&f , cache_id_ );

      T F( this->shape() , f + i * site_words_ );
      return F;
    }

    
    void writeElemTo(int i, std::ostream& out) const
    {
      typename WordType<T>::Type_t *f;
      QDP_get_global_cache().getHostPtr( (void**)&f , cache_id_ );

      out.write( (const char*)(f + i * site_words_) , sizeof(typename WordType<T>::Type_t) * WordSize<T>::value( this->shape() ) );
    }


    void readElemFrom(int i, std::istream& in)
    {
      typename WordType<T>::Type_t *f;
      QDP_get_global_cache().getHostPtr( (void**)&f , cache_id_ );

      in.read( (char*)(f + i * site_words_) , sizeof(typename WordType<T>::Type_t) * WordSize<T>::value( this->shape() ) );
    }


    inline void moveToFastMemoryHint(bool copy=false) {}
    inline void revertFromFastMemoryHint(bool copy=false) {}

    

    void static copyHostDevice( bool toDev , void * outPtr , void * inPtr , int _layout_ref , Shape shape , JitDataFormat data_format )
    {
      if (toDev)
	llvm_jit_stats_lattice2dev();
      else
	llvm_jit_stats_lattice2host();

      typedef typename WordType<T>::Type_t W;

      W* hst = (W*)( toDev ? inPtr : outPtr );
      void* dev = toDev ? outPtr : inPtr;

      const size_t lim_rea = GetLimit<T,2>::limit(shape);
      const size_t lim_col = GetLimit<T,1>::limit(shape);
      const size_t lim_spi = GetLimit<T,0>::limit(shape);
      const size_t sites   = Layout::from_index( _layout_ref ).sitesOnNode();

      auto copy = [&]( auto* dev_data , auto convert_to_dev , auto convert_to_hst )
      {
	for ( size_t site = 0 ; site < sites ; site++ ) {
	  for ( size_t reality = 0 ; reality < lim_rea ; reality++ ) {
	    for ( size_t color = 0 ; color < lim_col ; color++ ) {
	      for ( size_t spin = 0 ; spin < lim_spi ; spin++ ) {
		size_t hst_idx = (((site)*lim_spi + spin) * lim_col + color ) * lim_rea + reality;

		size_t dev_idx;
		if ( data_format.order == JitDataOrder::SiteFast )
		  dev_idx = ( ( reality * lim_col + color ) * lim_spi + spin ) * sites + site;
		else 
		  dev_idx = hst_idx;

		if (toDev)
		  dev_data[dev_idx] = convert_to_dev( hst[hst_idx] );
		else
		  hst[hst_idx] = convert_to_hst( dev_data[dev_idx] );
	      }
	    }
	  }
	}
      };

      if ( data_format.half() )
	{
	  if constexpr ( std::is_floating_point_v<W> )
	    copy( (jit_half_t*)dev ,
		  []( W x ) { return qdp_to_half( x ); } ,
		  []( jit_half_t h ) { return (W)qdp_from_half( h ); } );
	  else
	    QDP_error_exit( "OLattice: half storage requires a floating-point word type" );
	}
      else
	copy( (W*)dev , []( W x ) { return x; } , []( W x ) { return x; } );
     }

    int getId() const {
      return cache_id_;
    }

    LatticeLayout layout() const {
      if (layout_.index() < 0)
	QDP_error_exit("OLattice::layout: invalid layout");
      return layout_;
    }




  protected:

    void alloc_mem( LatticeLayout layout = Layout::current() )
    {
      layout_ = layout;

      site_words_ = WordSize<T>::value( this->shape() );

      assert( site_words_ > 0 );
      
      const size_t words = (size_t)layout.sitesOnNode() * site_words_;
      const size_t host_size = words * sizeof(typename WordType<T>::Type_t);
      const size_t dev_size  = data_format_.half() ? words * sizeof(jit_half_t) : host_size;
      
      cache_id_ = QDP_get_global_cache().addConverted( dev_size , host_size , &copyHostDevice , layout_.index() , this->shape() , data_format_ );
      owns_memory_ = true;

      assert(cache_id_ >= 0);
    }
    
    void free_mem()
    { 
      if (cache_id_ >= 0 && owns_memory_)
	{
	  QDP_get_global_cache().signoff( cache_id_ );
	}
      owns_memory_ = false;
      cache_id_ = -1;
      layout_ = LatticeLayout();
      site_words_ = 0;
    }


  };



template<class T>
struct LeafFunctor<OLattice<T>, CollectLayouts>
{
  typedef int Type_t;
  inline static Type_t apply(const OLattice<T> &a, CollectLayouts &f) {
    f.layouts.push_back( a.layout().index() );
    return 0;
  }
};



template<class T>
struct LeafFunctor<OScalar<T>, ShiftPhase1>
{
  typedef int Type_t;
  inline static Type_t apply(const OScalar<T> &a, ShiftPhase1 &f) {
    return 0;
  }
};

template<class T>
struct LeafFunctor<OLattice<T>, ShiftPhase1>
{
  typedef int Type_t;
  inline static Type_t apply(const OLattice<T> &a, ShiftPhase1 &f) {
    if ( f.in_shift )
      check_shift_layout( a.layout().index() , f.layout_ref );
    return 0;
  }
};


template<class T>
struct LeafFunctor<OScalar<T>, ShiftPhase2>
{
  typedef int Type_t;
  inline static Type_t apply(const OScalar<T> &a, ShiftPhase2 &f) {
    return 0;
  }
};

template<class T>
struct LeafFunctor<OLattice<T>, ShiftPhase2>
{
  typedef int Type_t;
  inline static Type_t apply(const OLattice<T> &a, ShiftPhase2 &f) {
    return 0;
  }
};



template<class T>
class WordSize< OScalar<T> >
{
public:
  static int value( const Shape& ret )
  {
    return WordSize<T>::value( ret );
  }
};



template<class T> 
struct JITType<OLattice<T> >
{
  typedef OLatticeJIT<typename JITType<T>::Type_t>  Type_t;
};


template<class T> 
struct JITType<OScalar<T> >
{
  typedef OScalarJIT<typename JITType<T>::Type_t>  Type_t;
};


template<class T>
struct LeafFunctor<OLattice<T>, MGParamLeaf>
{
  typedef typename JITType< OLattice<T> >::Type_t  Type_t;
  inline static
  Type_t apply(const OLattice<T>& a, MGParamLeaf& p) 
  {
    const PlanEntry& e = p.next( PlanEntry::Kind::Lattice );
    ParamRef    base_addr = p.lattice_base< typename WordType<T>::Type_t >();
    return Type_t( base_addr , p.getLatSizeParamRef( e.layout_slot ) , e.shape , e.layout_slot , e.data_format );
  }
};


template<class T>
struct LeafFunctor<OScalar<T>, MGParamLeaf>
{
  typedef typename JITType< OScalar<T> >::Type_t  Type_t;
  inline static
  Type_t apply(const OScalar<T>& a, MGParamLeaf& p)
  {
    const PlanEntry& e = p.next( PlanEntry::Kind::Scalar );
    return Type_t( llvm_add_param< typename WordType<T>::Type_t * >() , e.shape );
  }
};




template<class T>
struct LeafFunctor<OLattice<T>, DummyJitLeaf>
{
  typedef typename JITType< T >::Type_t  Type_t;
  inline static
  Type_t apply(const OLattice<T>& a, DummyJitLeaf& p) 
  {
    return Type_t( a.shape() );
  }
};


template<class T>
struct LeafFunctor<OScalar<T>, DummyJitLeaf>
{
  typedef typename JITType< T >::Type_t  Type_t;
  inline static
  Type_t apply(const OScalar<T>& a, DummyJitLeaf& p)
  {
    return Type_t( a.shape() );
  }
};




template<class T>
struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< T > > > > >, MGParamLeaf>
{
  typedef typename JITType< OScalar< PScalar< PScalar < RScalar< Word< T > > > > > >::Type_t  Type_t;
  inline static
  Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< T > > > > >& a, MGParamLeaf& p)
  {
    p.next( PlanEntry::Kind::Literal );
    return Type_t( llvm_add_param< T >() , true );
  }
};



template<class T>
struct LeafFunctor<OScalarJIT<T>, MGViewLeaf>
{
  typedef T Type_t;
  inline static
  Type_t apply( OScalarJIT<T> s, MGViewLeaf& v)
  {
    return s.elem();
  }
};



template<class T>
struct LeafFunctor<OLattice<T>, SelfAssignTag>
{
  typedef int Type_t;
  inline static
  Type_t apply(const OLattice<T>& a, SelfAssignTag& p) 
  {
    return ( p.shift_depth > 0 && a.getId() == p.id ) ? 1:0;
  }
};






template<class T>
struct CreateLeaf<OScalar<T> >
{
  typedef OScalar<T> Leaf_t;
  //typedef Reference<OScalar<T> > Leaf_t;
  inline static
  Leaf_t make(const OScalar<T> &a) { return Leaf_t(a); }
};

template<class T>
struct CreateLeaf<OLattice<T> >
{
  //typedef OLattice<T> Leaf_t;
  typedef Reference<OLattice<T> > Leaf_t;
  inline static
  Leaf_t make(const OLattice<T> &a) { return Leaf_t(a); }
};


template<class T>
struct LeafFunctor<OScalar<T>, ElemLeaf>
{
  typedef T Type_t;
  //typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, ElemLeaf &f)
    {return Type_t(a.elem());}
};


template<class T>
struct LeafFunctor<OScalar<T>, EvalLeaf1>
{
  typedef T Type_t;
  //typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, EvalLeaf1 &f)
    {return Type_t(a.elem());}
};

template<class T>
struct LeafFunctor<OLattice<T>, EvalLeaf1>
{
  //typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OLattice<T> &a, EvalLeaf1 &f)
    {return Type_t(a.elem(f.val1()));}
};



template<class T>
struct WordType<OScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<OLattice<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T> 
struct SinglePrecType<OScalar<T> >
{
  typedef OScalar<typename SinglePrecType<T>::Type_t> Type_t;
};

template<class T>
struct DoublePrecType<OScalar<T> >
{
  typedef OScalar<typename DoublePrecType<T>::Type_t> Type_t;
};

template<class T> 
struct SinglePrecType<OLattice<T> >
{
  typedef OLattice<typename SinglePrecType<T>::Type_t> Type_t;
};

template<class T>
struct DoublePrecType<OLattice<T> >
{
  typedef OLattice<typename DoublePrecType<T>::Type_t> Type_t;
};



  
template<class T>
struct InternalScalar<OScalar<T> > {
  typedef OScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct InternalScalar<OLattice<T> > {
  typedef OScalar<typename InternalScalar<T>::Type_t>  Type_t;
};



template<class T>
struct PrimitiveScalar<OScalar<T> > {
  typedef OScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<OLattice<T> > {
  typedef OLattice<typename PrimitiveScalar<T>::Type_t>  Type_t;
};



template<class T>
struct LatticeScalar<OScalar<T> > {
  typedef OScalar<typename LatticeScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<OLattice<T> > {
  typedef OScalar<typename LatticeScalar<T>::Type_t>  Type_t;
};



template<class T>
struct RealScalar<OScalar<T> > {
  typedef OScalar<typename RealScalar<T>::Type_t>  Type_t;
};

template<class T>
struct RealScalar<OLattice<T> > {
  typedef OScalar<typename RealScalar<T>::Type_t>  Type_t;
};




template<class T1, class Op>
struct UnaryReturn<OScalar<T1>, Op> {
  typedef OScalar<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class Op>
struct UnaryReturn<OLattice<T1>, Op> {
  typedef OLattice<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, Op> {
  typedef OScalar<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OScalar<T3>, Op> {
  typedef OScalar<typename BinaryReturn<T2, T3, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class Op>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};



template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};



template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OScalar<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OLattice<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};



template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OLattice<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};


template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OScalar<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};





template<class T>
struct UnaryReturn<OScalar<T>, FnPeekSite> {
  typedef OScalar<typename UnaryReturn<T, FnPeekSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnSum> {
  typedef OScalar<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnSumMulti > {
  typedef multi1d<OScalar<typename UnaryReturn<T, FnSumMulti>::Type_t> >  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnNorm2 > {
  typedef OScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnGlobalMax> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnGlobalMin> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnLocalNorm2 > {
  typedef OScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnLocalInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnLocalColorInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnLocalInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



template<class T2>
struct BinaryReturn<GammaType, OScalar<T2>, OpGammaTypeMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<OScalar<T2>, GammaType, OpMultiplyGammaType> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


template<class T2>
struct BinaryReturn<GammaTypeDP, OScalar<T2>, OpGammaTypeMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<OScalar<T2>, GammaTypeDP, OpMultiplyGammaType> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};





template<class T>
struct UnaryReturn<OScalar<T>, OpNot > {
  typedef OScalar<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};


template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLT > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpGT > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpGE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpEQ > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpNE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpAnd > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpOr > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLeftShift > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpRightShift > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};



template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpAddAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpSubtractAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpMultiplyAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpDivideAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpModAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseOrAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseAndAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseXorAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLeftShiftAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpRightShiftAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OScalar<T3>, FnColorContract> {
  typedef OScalar<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<OLattice<T>, FnGetSite> {
  typedef OScalar<typename UnaryReturn<T, FnGetSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnPeekSite> {
  typedef OScalar<typename UnaryReturn<T, FnPeekSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnSum > {
  typedef OScalar<typename ScalarType< typename UnaryReturn<T, FnSum>::Type_t >::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnGlobalMax> {
  typedef OScalar<typename ScalarType< typename UnaryReturn<T, FnGlobalMax>::Type_t >::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnGlobalMin> {
  typedef OScalar<typename ScalarType< typename UnaryReturn<T, FnGlobalMin>::Type_t >::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnSumMulti > {
  typedef multi1d<OScalar<typename ScalarType< typename UnaryReturn<T, FnSumMulti>::Type_t>::Type_t > >  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnNorm2 > {
  typedef OScalar<typename ScalarType< typename UnaryReturn<T, FnNorm2>::Type_t>::Type_t >  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct > {
  typedef OScalar<typename ScalarType< typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>::Type_t >  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal > {
  typedef OScalar<typename ScalarType< typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>::Type_t >  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnLocalNorm2 > {
  typedef OLattice<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnLocalColorInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



template<class T2>
struct BinaryReturn<GammaType, OLattice<T2>, OpGammaTypeMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<OLattice<T2>, GammaType, OpMultiplyGammaType> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<GammaTypeDP, OLattice<T2>, OpGammaTypeMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2>
struct BinaryReturn<OLattice<T2>, GammaTypeDP, OpMultiplyGammaType> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};





template<class T>
struct UnaryReturn<OLattice<T>, OpNot > {
  typedef OLattice<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};


template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpAddAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpSubtractAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpMultiplyAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpDivideAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpModAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseOrAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseAndAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseXorAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLeftShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpRightShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OLattice<T3>, FnColorContract> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};



template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnInnerProduct > {
  typedef OScalar<typename ScalarType<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnInnerProductReal > {
  typedef OScalar<typename ScalarType<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnInnerProduct > {
  typedef OScalar<typename ScalarType<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnInnerProductReal > {
  typedef OScalar<typename ScalarType<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnLocalColorInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnLocalColorInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalColorInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpAddAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpSubtractAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpMultiplyAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpDivideAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpModAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseOrAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseAndAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseXorAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLeftShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpRightShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 




template<class T> 
inline typename WordType< OScalar<T> >::Type_t
toWordType(const OScalar<T>& s) 
{
  return toWordType(s.elem());
}


template<class T>
struct UnaryReturn<OScalar<T>, FnGetSite> {
  typedef OScalar<typename UnaryReturn<T, FnGetSite>::Type_t>  Type_t;
};


template<class T>
inline typename UnaryReturn<OScalar<T>, FnGetSite>::Type_t
getSite(const OScalar<T>& s1, int innersite)
{
  typename UnaryReturn<OScalar<T>, FnGetSite>::Type_t  d;

  d.elem() = getSite(s1.elem(), innersite);
  return d;
}


template<class T> 
void zero_rep(OScalar<T>& dest) 
{
  dest.zero_me();
}


template<class T1, class T2> 
void copymask(OScalar<T2>& dest, const OScalar<T1>& mask, const OScalar<T2>& s1) 
{
  copymask(dest.elem(), mask.elem(), s1.elem());
}



template<class T>
void random(OScalar<T>& d);


} // namespace QDP

#endif
