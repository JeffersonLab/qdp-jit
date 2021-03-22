#ifndef QDP_MULTI_OUTER
#define QDP_MULTI_OUTER


namespace QDP {

  /*! @defgroup multi  Multi-dimensional arrays
   *
   * Container classes that provide 1D, 2D, 3D and 4D multidimensional
   * array semantics.
   *
   * @{
   */


  //! Container for a multi-dimensional 1D array


  template<class T1> class multi1d< OScalar<T1> >
  {
  public:
    multi1d() {
      //std::cout << __PRETTY_FUNCTION__ << " " << this << "  copymem = " << copymem << "  F = " << F << "  id = " << id << "\n";
    }

    multi1d(OScalar<T1>* f, int ns1, int id_  )
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      F=f;
      n1=ns1;
      copymem=true;
      id = id_;
    }

    explicit multi1d(int ns1) {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      resize(ns1);
    }

    ~multi1d()
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "  copymem = " << copymem << "  F = " << F << "  id = " << id << "\n";

      if (! copymem)
	{
	  delete[] F;
	  if ( id != -1 )
	    QDP_get_global_cache().signoff( id );
	}
    }

    //! Copy constructor
    // Copy from s, into slow memory
    multi1d(const multi1d& s)
    {
      resize(s.n1);

      for(int i=0; i < n1; ++i)
	F[i] = s.F[i];

#if 0
      std::cout << __PRETTY_FUNCTION__ << " "
		<< this
		<< "  copymem = " << copymem
		<< "  F = " << F
		<< "  id = " << id
		<< "  n = " << n1
		<< "\n";
#endif
    }

    //! Size of array
    int size() const {return n1;}
    int size1() const {return n1;}

    //! Equal operator uses underlying = of T
    /*! Default = */
    multi1d& operator=(const multi1d& s1)
    {
      if (size() != s1.size())   // a simple check avoids resizing always
	resize(s1.size());

      for(int i=0; i < n1; ++i)
	F[i] = s1.F[i];
      return *this;
    }


    template<class T2>
    multi1d< OScalar<T1> >& operator=(const T2& s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (F == 0 || id < 0)
	{
	  cerr << "multi1d: left hand side not initialized in =" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] = s1;
      return *this;
    }

    //! Add-replace on each element
    /*! Uses underlying += */
    multi1d< OScalar<T1> >& operator+=(const multi1d< OScalar<T1> >& s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (size() != s1.size())
	{
	  cerr << "multi1d: Sizes incompatible in +=" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] += s1.F[i];
      return *this;
    }

    //! Add-replace on each element
    /*! Uses underlying += */
    multi1d< OScalar<T1> >& operator+=(const OScalar<T1>& s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (F == 0 || id < 0)
	{
	  cerr << "multi1d: left hand side not initialized in +=" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] += s1;
      return *this;
    }

    //! Subtract-replace on each element
    /*! Uses underlying -= */
    multi1d< OScalar<T1> >& operator-=(const multi1d< OScalar<T1> >& s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (size() != s1.size())
	{
	  cerr << "multi1d: Sizes incompatible in -=" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] -= s1.F[i];
      return *this;
    }

    //! Subtract-replace on each element
    /*! Uses underlying -= */
    multi1d< OScalar<T1> >& operator-=(const OScalar<T1> & s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (F == 0 || id < 0)
	{
	  cerr << "multi1d: left hand side not initialized in -=" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] -= s1;
      return *this;
    }

    //! Mult-replace on each element
    /*! Uses underlying *= */
    multi1d< OScalar<T1> >& operator*=(const multi1d< OScalar<T1> >& s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (size() != s1.size())
	{
	  cerr << "multi1d: Sizes incompatible in *=" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] *= s1.F[i];
      return *this;
    }

    //! Mult-replace on each element
    /*! Uses underlying *= */
    multi1d< OScalar<T1> >& operator*=(const OScalar<T1>& s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (F == 0 || id < 0)
	{
	  cerr << "multi1d: left hand side not initialized in *=" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] *= s1;
      return *this;
    }

    //! Divide-replace on each element
    /*! Uses underlying /= */
    multi1d< OScalar<T1> >& operator/=(const multi1d< OScalar<T1> >& s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (size() != s1.size())
	{
	  cerr << "multi1d: Sizes incompatible in /=" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] /= s1.F[i];
      return *this;
    }

    //! Divide-replace on each element
    /*! Uses underlying /= */
    multi1d< OScalar<T1> >& operator/=(const OScalar<T1>& s1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "\n";
      if (F == 0 || id < 0)
	{
	  cerr << "multi1d: left hand side not initialized in /=" << endl;
	  exit(1);
	}

      for(int i=0; i < n1; ++i)
	F[i] /= s1;
      return *this;
    }


    void copyD2H() {
      assert( id >= 0 );
      QDP_get_global_cache().copyD2H( id );
    }

    // globalSumArray uses slice()
    // however, in qdp-jit a pointer to an array of OScalar cannot be used
    // to access the contained values. So, an explicitly different name is used
    // to get access to the host values.
    //
    //const T* slice() const {return F;}
    T1* slice_host() const {
      assert( id >= 0 );
      void* ptr;
      QDP_get_global_cache().getHostPtr( &ptr , id );
      return static_cast<T1*>(ptr);
    }

    //! Return ref to an element
    OScalar<T1>& operator()(int i) {
      assert( i >= 0 && i < n1 );
      return F[i];
    }

    //! Return const ref to an element
    const OScalar<T1>& operator()(int i) const {
      assert( i >= 0 && i < n1 );
      return F[i];
    }

    //! Return ref to an element
    OScalar<T1>& operator[](int i) {
      assert( i >= 0 && i < n1 );
      return F[i];
    }

    //! Return const ref to an element
    const OScalar<T1>& operator[](int i) const {
      assert( i >= 0 && i < n1 );
      return F[i];
    }


    void resize(int ns1)
    {
      // std::cout << __PRETTY_FUNCTION__ << " " << this << "  copymem = " << copymem << "  F = " << F << "  id = " << id << "\n";

      assert( ns1 > 0 );
      assert(!copymem);

      //QDPIO::cout << "resize multi1d<OScalar> specialization\n";
      if (id >= 0) {
	QDP_get_global_cache().signoff( id );
      }

      n1 = ns1;
      //size_t size = n1 * sizeof(T1);

      delete[] F;
      F = new(nothrow) OScalar<T1>[n1];
      if ( F == 0x0 ) { 
	QDP_error_exit("Unable to allocate memory in multi1d::resize(%d)\n",ns1);
      }

      std::vector<void*> tmp;
      for ( int i = 0 ; i < n1 ; ++i )
	{
	  tmp.push_back( (void*)F[i].get_raw_F() );
	}
      id = QDP_get_global_cache().addArray( sizeof(T1) , n1 , tmp );      

      // Initialize all elements as the appropriate array element
#if 1
      for ( int i = 0 ; i < n1 ; ++i )
	{
	  //F[i].setElemNum( i + id_offset );
	  F[i].setElemNum( i );
	  F[i].setId(id);
	}
#endif
      
      // std::cout << __PRETTY_FUNCTION__ << "(exit) " << this << "  copymem = " << copymem << "  F = " << F << "  id = " << id << "\n";
    }

    int getId() const {
      return id;
    }

  private:
    bool copymem = false;
    int n1 = 0;
    int id = -1;
    OScalar<T1>* F = NULL;
  };


  template<class T1> 
  void zero_rep(multi1d< OScalar<T1> >& dest) 
  {
    assert( dest.getId() >= 0 );
    QDP_get_global_cache().zero_rep( dest.getId() );
  }
  

  template<class T1> class multi2d< OScalar<T1> >
  {
  public:
    typedef OScalar<T1> T;
    
    multi2d() {}

    //multi2d(OScalar<T1>* f, int ns2, int ns1, int id_ , int id_offset_ )
    multi2d(OScalar<T1>* f, int ns2, int ns1, int id_ )
    {
      F=f;
      n1=ns1;
      n2=ns2;
      copymem=true;
      id = id_;
      //id_offset = id_offset_;
    }

    explicit multi2d(int ns2, int ns1) {resize(ns2,ns1);}
  

    ~multi2d()
    {
      if (! copymem)
	{
	  delete[] F;
	  if ( id > 0 )
	    QDP_get_global_cache().signoff( id );
	}
    }


    //! Copy constructor
    multi2d(const multi2d& s)
    {
      resize(s.n2,s.n1);

      for(int i=0; i < n1 * n2; ++i)
	F[i] = s.F[i];
    }

  
    void resize(int ns2, int ns1) 
    {
      assert(!copymem);

      //QDPIO::cout << "resize multi2d<OScalar> specialization\n";
      if (id >= 0) {
	QDP_get_global_cache().signoff( id );
      }

      n1 = ns1;
      n2 = ns2;
      int sz = n1 * n2;

      delete[] F;
      F = new(nothrow) OScalar<T1>[sz];
      if ( F == 0x0 ) { 
	QDP_error_exit("Unable to allocate memory in multi1d::resize(%d)\n",ns1);
      }

      std::vector<void*> tmp;
      for ( int i = 0 ; i < sz ; ++i )
	{
	  tmp.push_back( (void*)F[i].get_raw_F() );
	}
      id = QDP_get_global_cache().addArray( sizeof(T1) , sz , tmp );


#if 1
      // Initialize all elements as the appropriate array element
      for ( int i = 0 ; i < sz ; ++i )
	{
	  //F[i].setElemNum( i + id_offset );
	  F[i].setElemNum( i );
	  F[i].setId(id);
	}
#endif
    }

    int getId() const {
      return id;
    }


    //! Size of array
    int size1() const {return n1;}
    int size2() const {return n2;}

    //! Another variant on the size of the 2d array
    int nrows() const {return n2;}
    int ncols() const {return n1;}

    //! Equal operator uses underlying = of T
    multi2d<T>& operator=(const multi2d<T>& s1)
    {
      resize(s1.size2(), s1.size1());   // always resize

      for(int i=0; i < n1 * n2; ++i)
	F[i] = s1.F[i];
      return *this;
    }


    template<class T2>
    multi2d< OScalar<T1> >& operator=(const T2& s1)
    {
      if (F == 0 || id < 0)
      {
	cerr << "multi2d: left hand side not initialized in =" << endl;
	exit(1);
      }

      for(int i=0; i < n1 * n2; ++i)
	F[i] = s1;
      return *this;
    }

#if 0
    //! Return ref to a row slice
    const T* slice(int j) const {return F+n1*j;}
#endif
  
    //! Return ref to an element
    //T& operator()(int j, int i) {return F[i+n1*j];}

    OScalar<T1>& operator()(int j, int i) {
      assert( i >= 0 && i < n1 );
      assert( j >= 0 && j < n2 );
      //F[i].setElemNum( i + n1 * j + id_offset );
      //F[i].setId( id );
      //return F[i];
      return F[ i + n1 * j ];
    }

  
    //! Return const ref to an element
    //const T& operator()(int j, int i) const {return F[i+n1*j];}

    const OScalar<T1>& operator()(int j, int i) const {
      assert( i >= 0 && i < n1 );
      assert( j >= 0 && j < n2 );
      //F[i].setElemNum( i + n1 * j + id_offset );
      //F[i].setId( id );
      //return F[i];
      return F[ i + n1 * j ];
    }


    multi1d< OScalar<T1> > operator[](int j) {
      assert( j >= 0 && j < n2 );
      //return multi1d< OScalar<T1> >( F + j * n1 , n1 , id , j * n1 );
      return multi1d< OScalar<T1> >( F + j * n1 , n1 , id );
    }

  
    const multi1d< OScalar<T1> > operator[](int j) const {
      assert( j >= 0 && j < n2 );
      //return multi1d< OScalar<T1> >( F + j * n1 , n1 , id , j * n1 );
      return multi1d< OScalar<T1> >( F + j * n1 , n1 , id );
    }

  
  private:
    bool copymem = false;
    int n1 = 0;
    int n2 = 0;
    //int sz = 0;
    int id = -1;
    //int id_offset = 0;
    OScalar<T1>* F = NULL;
  };







  template<class T1> class multi3d< OScalar<T1> >
  {
  public:
    typedef OScalar<T1> T;
    
    multi3d() {}
    
    explicit multi3d(int ns3, int ns2, int ns1) {resize(ns3,ns2,ns1);}
  
    multi3d(OScalar<T1>* f, int ns3, int ns2, int ns1, int id_ )
    {
      F=f;
      n1=ns1;
      n2=ns2;
      n3=ns3;
      copymem=true;
      id = id_;
    }

    ~multi3d()
    {
      if (! copymem)
	{
	  delete[] F;
	  if ( id > 0 )
	    QDP_get_global_cache().signoff( id );
	}
    }

    //! Copy constructor
    multi3d(const multi3d& s)
    {
      resize(s.n3,s.n2,s.n1);

      for(int i=0; i < n1 * n2 * n3 ; ++i)
	F[i] = s.F[i];
    }

    
    void resize(int ns3, int ns2 , int ns1) 
    {
      assert(!copymem);

      //QDPIO::cout << "resize multi2d<OScalar> specialization\n";
      if (id >= 0) {
	QDP_get_global_cache().signoff( id );
      }

      n1 = ns1;
      n2 = ns2;
      n3 = ns3;
      int sz = n1 * n2 * n3;

      delete[] F;
      F = new(nothrow) OScalar<T1>[sz];
      if ( F == 0x0 ) { 
	QDP_error_exit("Unable to allocate memory in multi1d::resize(%d)\n",ns1);
      }

      std::vector<void*> tmp;
      for ( int i = 0 ; i < sz ; ++i )
	{
	  tmp.push_back( (void*)F[i].get_raw_F() );
	}
      id = QDP_get_global_cache().addArray( sizeof(T1) , sz , tmp );


#if 1
      // Initialize all elements as the appropriate array element
      for ( int i = 0 ; i < sz ; ++i )
	{
	  //F[i].setElemNum( i + id_offset );
	  F[i].setElemNum( i );
	  F[i].setId(id);
	}
#endif
    }

    int getId() const {
      return id;
    }


    //! Size of array
    int size1() const {return n1;}
    int size2() const {return n2;}
    int size3() const {return n3;}

    //! Another variant on the size of the 3d array
    int leftSize()   const {return n3;}
    int middleSize() const {return n2;}
    int rightSize()  const {return n1;}

    //! Equal operator uses underlying = of T
    multi3d<T>& operator=(const multi3d<T>& s1)
    {
      resize(s1.size3(), s1.size2(), s1.size1());

      for(int i=0; i < n1*n2*n3; ++i)
	F[i] = s1.F[i];
      return *this;
    }

#if 0
    //! Return ref to a row slice
    const T* slice(int k, int j) const {return F+n1*(j+n2*(k));}
#endif

    template<class T2>
    multi3d< OScalar<T1> >& operator=(const T2& s1)
    {
      if (F == 0 || id < 0)
      {
	cerr << "multi3d: left hand side not initialized in =" << endl;
	exit(1);
      }

      for(int i=0; i < n1*n2*n3; ++i)
	F[i] = s1;
      return *this;
    }

    
    //! Return ref to an element
    //T& operator()(int k, int j, int i) {return F[i+n1*(j+n2*(k))];}

    OScalar<T1>& operator()(int k, int j, int i) {
      assert( i >= 0 && i < n1 );
      assert( j >= 0 && j < n2 );
      assert( k >= 0 && k < n3 );
      //F[i].setElemNum( i + n1 * ( j + n2 * k ) );
      //F[i].setId( id );
      //return F[i];
      return F[i + n1 * ( j + n2 * k )];
    }

  
    //! Return const ref to an element
    //

    const OScalar<T1>& operator()(int k, int j, int i) const {
      assert( i >= 0 && i < n1 );
      assert( j >= 0 && j < n2 );
      assert( k >= 0 && k < n3 );
      //F[i].setElemNum( i + n1 * ( j + n2 * k ) );
      //F[i].setId( id );
      //return F[i];
      return F[i + n1 * ( j + n2 * k )];
    }


    // //! Return ref to an element
    // multi1d<T> operator[](int j) {return multi1d<T>(F+j*n1,n1);}

    // multi1d< OScalar<T1> > operator[](int j) {
    //   assert( j >= 0 && j < n2 );
    //   return multi1d< OScalar<T1> >( F + j * n1 , n1 , id , j * n1 );
    // }

    
    //multi2d<T> operator[](int k) {return multi2d<T>( F+n1*n2*k ,n2,n1);}

    multi2d< OScalar<T1> > operator[](int k) {
      assert( k >= 0 && k < n3 );
      //return multi2d< OScalar<T1> >( F + n1 * n2 * k , n2 , n1 , id , n1 * n2 * k );
      return multi2d< OScalar<T1> >( F + n1 * n2 * k , n2 , n1 , id );
    }

  
    //! Return const ref to an element
    //const multi1d<T> operator[](int j) const {return multi1d<T>(F+j*n1,n1);}

    const multi2d< OScalar<T1> > operator[](int k) const {
      assert( k >= 0 && k < n3 );
      //return multi2d< OScalar<T1> >( F + n1 * n2 * k , n2 , n1 , id , n1 * n2 * k );
      return multi2d< OScalar<T1> >( F + n1 * n2 * k , n2 , n1 , id );
    }

    // const multi1d< OScalar<T1> > operator[](int j) const {
    //   assert( j >= 0 && j < n2 );
    //   return multi1d< OScalar<T1> >( F + j * n1 , n1 , id , j * n1 );
    // }

  
  private:
    bool copymem = false;
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int id = -1;
    OScalar<T1>* F = NULL;
  };










  template<class T1> class multi4d< OScalar<T1> >
  {
  public:
    typedef OScalar<T1> T;
    
    multi4d() {}
    
    explicit multi4d(int ns4, int ns3, int ns2, int ns1) {resize(ns4,ns3,ns2,ns1);}
  
    multi4d(OScalar<T1>* f, int ns4, int ns3, int ns2, int ns1, int id_ )
    {
      F=f;
      n1=ns1;
      n2=ns2;
      n3=ns3;
      n4=ns4;
      copymem=true;
      id = id_;
    }

    ~multi4d()
    {
      if (! copymem)
	{
	  delete[] F;
	  if ( id > 0 )
	    QDP_get_global_cache().signoff( id );
	}
    }

    //! Copy constructor
    multi4d(const multi4d& s)
    {
      resize(s.n4,s.n3,s.n2,s.n1);

      for(int i=0; i < n1 * n2 * n3 *n4 ; ++i)
	F[i] = s.F[i];
    }

    
    void resize(int ns4, int ns3, int ns2 , int ns1) 
    {
      assert(!copymem);

      //QDPIO::cout << "resize multi2d<OScalar> specialization\n";
      if (id >= 0) {
	QDP_get_global_cache().signoff( id );
      }

      n1 = ns1;
      n2 = ns2;
      n3 = ns3;
      n4 = ns4;
      int sz = n1 * n2 * n3 * n4;

      delete[] F;
      F = new(nothrow) OScalar<T1>[sz];
      if ( F == 0x0 ) { 
	QDP_error_exit("Unable to allocate memory in multi1d::resize(%d)\n",ns1);
      }

      std::vector<void*> tmp;
      for ( int i = 0 ; i < sz ; ++i )
	{
	  tmp.push_back( (void*)F[i].get_raw_F() );
	}
      id = QDP_get_global_cache().addArray( sizeof(T1) , sz , tmp );


#if 1
      // Initialize all elements as the appropriate array element
      for ( int i = 0 ; i < sz ; ++i )
	{
	  //F[i].setElemNum( i + id_offset );
	  F[i].setElemNum( i );
	  F[i].setId(id);
	}
#endif
    }

    int getId() const {
      return id;
    }


    //! Size of array
    int size1() const {return n1;}
    int size2() const {return n2;}
    int size3() const {return n3;}
    int size4() const {return n4;}

    //! Equal operator uses underlying = of T
    multi4d<T>& operator=(const multi4d<T>& s1)
    {
      resize(s1.size4(), s1.size3(), s1.size2(), s1.size1());

      for(int i=0; i < n1*n2*n3*n4; ++i)
	F[i] = s1.F[i];
      return *this;
    }

#if 0
    //! Return ref to a row slice
    const T* slice(int k, int j) const {return F+n1*(j+n2*(k));}
#endif

    template<class T2>
    multi4d< OScalar<T1> >& operator=(const T2& s1)
    {
      if (F == 0 || id < 0)
      {
	cerr << "multi4d: left hand side not initialized in =" << endl;
	exit(1);
      }

      for(int i=0; i < n1*n2*n3*n4; ++i)
	F[i] = s1;
      return *this;
    }

    
    //! Return ref to an element
    //T& operator()(int k, int j, int i) {return F[i+n1*(j+n2*(k))];}

    OScalar<T1>& operator()(int l, int k, int j, int i) {
      assert( i >= 0 && i < n1 );
      assert( j >= 0 && j < n2 );
      assert( k >= 0 && k < n3 );
      assert( l >= 0 && l < n4 );
      //F[i].setElemNum( i + n1 * ( j + n2 * k ) );
      //F[i].setId( id );
      //return F[i];
      return F[i + n1 * ( j + n2 * ( k + n3 * l ) )];
    }

  
    //! Return const ref to an element
    //

    const OScalar<T1>& operator()(int l, int k, int j, int i) const {
      assert( i >= 0 && i < n1 );
      assert( j >= 0 && j < n2 );
      assert( k >= 0 && k < n3 );
      assert( l >= 0 && l < n4 );
      //F[i].setElemNum( i + n1 * ( j + n2 * k ) );
      //F[i].setId( id );
      //return F[i];
      return F[i + n1 * ( j + n2 * ( k + n3 * l ) )];
    }


    // //! Return ref to an element
    // multi1d<T> operator[](int j) {return multi1d<T>(F+j*n1,n1);}

    // multi1d< OScalar<T1> > operator[](int j) {
    //   assert( j >= 0 && j < n2 );
    //   return multi1d< OScalar<T1> >( F + j * n1 , n1 , id , j * n1 );
    // }

    
    //multi2d<T> operator[](int k) {return multi2d<T>( F+n1*n2*k ,n2,n1);}

    multi3d< OScalar<T1> > operator[](int l) {
      assert( l >= 0 && l < n4 );
      //return multi2d< OScalar<T1> >( F + n1 * n2 * k , n2 , n1 , id , n1 * n2 * k );
      return multi3d< OScalar<T1> >( F + n1 * n2 * n3 * l , n3 , n2 , n1 , id );
    }

  
    //! Return const ref to an element
    //const multi1d<T> operator[](int j) const {return multi1d<T>(F+j*n1,n1);}

    const multi3d< OScalar<T1> > operator[](int l) const {
      assert( l >= 0 && l < n4 );
      //return multi2d< OScalar<T1> >( F + n1 * n2 * k , n2 , n1 , id , n1 * n2 * k );
      return multi3d< OScalar<T1> >( F + n1 * n2 * n3 * l , n3 , n2 , n1 , id );
    }

    // const multi1d< OScalar<T1> > operator[](int j) const {
    //   assert( j >= 0 && j < n2 );
    //   return multi1d< OScalar<T1> >( F + j * n1 , n1 , id , j * n1 );
    // }

  
  private:
    bool copymem = false;
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int n4 = 0;
    int id = -1;
    OScalar<T1>* F = NULL;
  };




  
} // QDP



#endif
