#ifndef QDP_FUNCTIONS_H
#define QDP_FUNCTIONS_H




namespace QDP {

  namespace Layout
  {
    //! coord[mu]  <- mu  : fill with lattice coord in mu direction
    LatticeInteger latticeCoordinate(int mu);
  }




  //-----------------------------------------------------------------------------
  template<class T, class T1, class Op, class RHS>
  void evaluate(OScalar<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
		const Subset& s, bool ok_to_log = true)
  {
    ScalarEval::run( dest , op , rhs );
  }


  //-----------------------------------------------------------------------------
  //! OLattice Op Scalar(Expression(source)) under an Subset
  /*! 
   * OLattice Op Expression, where Op is some kind of binary operation 
   * involving the destination 
   */
  template<class T, class T1, class Op, class RHS>
  void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
		const Subset& s) //bool ok_to_log = true 
  {
#if defined(QDP_USE_PROFILING)	 
    static QDPProfile_t prof(dest, op, rhs);
    prof.start_time();
#endif

    Eval::run( dest , op , rhs , s , "eval_lat_sca" , false );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }

  


  template<class T, class T1, class Op, class RHS>
  void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs,
		const Subset& s) // bool ok_to_log = true 
  {
#if defined(QDP_USE_PROFILING)	 
    static QDPProfile_t prof(dest, op, rhs);
    prof.start_time();
#endif

    Eval::run( dest , op , rhs , s );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }




  template<class T, class C1, class Op, class RHS>
  void evaluate_subtype_type(OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs,
			     const Subset& s)
  {
    QDP_error_exit("evaluate_subtype_type: not implemented");
  }


  

  template<class T, class T1, class Op>
  void operator_subtype_subtype(OSubLattice<T>& dest, const Op& op, const OSubLattice<T1>& rhs, const Subset& s)
  {
    QDP_error_exit("operator_subtype_subtype: not implemented");
  }


  

  


  //! dest = (mask) ? s1 : dest
  template<class T1, class T2>
  void copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnCopyMask(), "copymask", dest, mask , s1);
    prof.start_time();
#endif

    CopyMask::run( dest , mask , s1 );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }







  template<class T>
  void 
  random(OLattice<T>& d, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnRandom(), "random", d);
    prof.start_time();
#endif

    Random::run( d , s );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }

  
  //! dest  = random  
  template<class T>
  void random(OLattice<T>& d)
  {
    random(d,all);
  }
  


  //! dest  = random
  template<class T>
  void random(OScalar<T>& d)
  {
    Random::run( d );
  }


  //! dest  = gaussian   under a subset
  template<class T>
  void gaussian(OLattice<T>& d, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnRandom(), "gaussian", d);
    prof.start_time();
#endif

    Random::run_gaussian( d , s );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }


  template<class T>
  void gaussian(OLattice<T>& d)
  {
    gaussian(d,all);
  }

  
  template<class T> 
  inline
  void zero_rep(OLattice<T>& dest, const Subset& s) 
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnRandom(), "zero_rep", dest);
    prof.start_time();
#endif

    ZeroRep::run( dest , s );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }


  template<class T> 
  void zero_rep_subtype(OSubLattice<T>& dest, const Subset& s) 
  {
    QDP_error_exit("zero_rep_subtype: not implemented");
  }

  template<class T> 
  void zero_rep(OLattice<T>& dest) 

  {
    zero_rep(dest,all);
  }




  //-----------------------------------------------
  // Global sums
  //! OScalar = sum(OScalar) under an explicit subset
  /*!
   * Allow a global sum that sums over the lattice, but returns an object
   * of the same primitive type. E.g., contract only over lattice indices
   */
  template<class RHS, class T>
  typename UnaryReturn<OScalar<T>, FnSum>::Type_t
  sum(const QDPExpr<RHS,OScalar<T> >& s1, const Subset& s)
  {
    Shape shape = shape_of( s1 );

    typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d(shape);

    evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

    return d;
  }


  //! OScalar = sum(OScalar)
  /*!
   * Allow a global sum that sums over the lattice, but returns an object
   * of the same primitive type. E.g., contract only over lattice indices
   */
  template<class RHS, class T>
  typename UnaryReturn<OScalar<T>, FnSum>::Type_t
  sum(const QDPExpr<RHS,OScalar<T> >& s1)
  {
    Shape shape = shape_of( s1 );
    typename UnaryReturn<OScalar<T>, FnSum>::Type_t d(shape);

    evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

    return d;
  }




  template<class RHS, class T>
  typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t
  sumMulti(const QDPExpr<RHS,OScalar<T> >& s1, const Set& ss)
  {
    Shape shape = shape_of( s1 );

    ShapeScope qv_scope(shape);

    typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

    qv_scope.close();

    for(int i=0; i < ss.numSubsets(); ++i)
      evaluate(dest[i],OpAssign(),s1,all);

    return dest;
  }


  
  template<ConceptHasNoShift RHS, class T1>
  typename UnaryReturn< OLattice<T1> , FnSum>::Type_t
  sum(const QDPExpr<RHS, OLattice<T1> >& rhs, const Subset& s)
  {
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t Ret_t;

    Shape shape = shape_of(rhs);

    Ret_t d(shape);
    zero_rep(d);

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(d, OpAssign(), rhs);
    prof.start_time();
#endif

    Reduce::run<Reduce::Sum, typename Ret_t::SubType_t>( rhs , Reduce::Segments( s , get_layout_from_rhs( rhs ) ) ,
							   shape , jit_sizeof(d) , d.get_ptr() );

    QDPInternal::globalSum(d);

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif

#ifdef QDP_DEEP_LOG
    QDP_get_global_logger().log(d);
#endif

    return d;
  }


  
  template<ConceptHasShift RHS, class T1>
  typename UnaryReturn< OLattice<T1> , FnSum>::Type_t
  sum(const QDPExpr<RHS, OLattice<T1> >& rhs, const Subset& s)
  {
    OLattice<T1> tmp = rhs;
    return sum(tmp,s);
  }
  


  
  template<class T1, class RHS>
  typename UnaryReturn< OLattice<T1> , FnSum>::Type_t
  sum(const QDPExpr<RHS,OLattice<T1> >& rhs)
  {
    return sum(rhs,all);
  }



  template<ConceptHasNoShift RHS, class T1>
  typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t
  sumMulti( const QDPExpr<RHS, OLattice<T1> >& rhs, const Set& ss)
  {
    typedef typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t Ret_t;
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;
    typedef typename WordType<T2>::Type_t T2WT;

    Shape shape = shape_of(rhs);

    const int numsubsets = ss.numSubsets();
    assert(numsubsets > 0);

    ShapeScope qv_scope( shape );
    Ret_t dest( numsubsets );
    qv_scope.close();

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof( FnSumMulti() , dest );
    prof.start_time();
#endif

    const int words = jit_sizeof_words( dest[0] );
    std::vector<T2WT> values( words * numsubsets , T2WT(0) );

    Reduce::run<Reduce::Sum, T2>( rhs , Reduce::Segments( ss , get_layout_from_rhs( rhs ) ) ,
				  shape , jit_sizeof(dest[0]) , values.data() );

    for ( int i = 0 ; i < numsubsets ; ++i )
      memcpy( dest[i].get_ptr() , &values[ words * i ] , words * sizeof(T2WT) );

    QDPInternal::globalSumArray(dest);

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif

#ifdef QDP_DEEP_LOG
    for (int i = 0 ; i < numsubsets ; ++i )
      {
	QDP_get_global_logger().log(dest[i]);
      }
#endif

    return dest;
  }

  template<ConceptHasShift RHS, class T1>
  typename UnaryReturn< OLattice<T1> , FnSumMulti>::Type_t
  sumMulti(const QDPExpr<RHS, OLattice<T1> >& rhs, const Set& ss)
  {
    OLattice<T1> tmp = rhs;
    return sumMulti(tmp,ss);
  }


  
  


  // Not sure if needed

  template<class T>
  multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>
  sumMulti(const multi1d< OScalar<T> >& s1, const Set& ss)
  {
    Shape shape = s1[0].shape();

    ShapeScope qv_scope(shape);
    
    multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t> dest(s1.size(), ss.numSubsets());

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof( FnSumMulti() , s1 );
    prof.start_time();
#endif

    for(int i=0; i < dest.size1(); ++i)
      for(int j=0; j < dest.size2(); ++j)
	dest(j,i) = s1[j];

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif


    return dest;
  }




  template<class T>
  multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>
  sumMulti(const multi1d< OLattice<T> >& s1, const Set& ss)
  {
    Shape shape = s1[0].shape();

    ShapeScope qv_scope(shape);
    
    multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t> dest(s1.size(), ss.numSubsets());

    for(int k=0; k < s1.size(); ++k)
      {
	dest[k] = sumMulti( s1[k] , ss );
      }

    return dest;
  }





  template<class T>
  inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
  norm2(const multi1d< OScalar<T> >& s1)
  {
    Shape shape = s1[0].shape();
    typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t  d(shape);

    // Possibly loop entered
    zero_rep(d.elem());

    for(int n=0; n < s1.size(); ++n)
      {
	d += norm2( s1[n] );
      }

    return d;
  }



  //! OScalar = sum(OScalar)  under an explicit subset
  /*! Discards subset */
  template<class T>
  inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
  norm2(const multi1d< OScalar<T> >& s1, const Subset& s)
  {
    return norm2(s1);
  }




  template<class T1, class T2>
  bool multi1dOnOneLayout(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
  {
    for(int n=0; n < s1.size(); ++n)
      if (s1[n].layout() != s1[0].layout())
	return false;
    for(int n=0; n < s2.size(); ++n)
      if (s2[n].layout() != s1[0].layout())
	return false;
    return true;
  }


  //! OScalar = norm2(multi1d<OLattice>) under an explicit subset
  /*!
   * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
   *
   * Sum over the lattice
   * Allow a global sum that sums over all indices
   */
  template<class T>
  inline typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t
  norm2(const multi1d< OLattice<T> >& s1, const Subset& s)
  {
    Shape shape = s1[0].shape();
    typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t  d(shape);

    // Initialize
    d = 0.;

    if (multi1dOnOneLayout(s1, s1))
      {
	d = Reduce::sum_elements(s1.size(), [&](int n) { return localNorm2(s1[n]); }, s);
	return d;
      }

    for(int n=0; n < s1.size(); ++n)
      {
	d += norm2( s1[n] , s );
      }
    
    return d;
  }




  //! OScalar = norm2(multi1d<OLattice>)
  /*!
   * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
   *
   * Sum over the lattice
   * Allow a global sum that sums over all indices
   */
  template<class T>
  inline typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t
  norm2(const multi1d< OLattice<T> >& s1)
  {
    return norm2(s1,all);
  }




  //-----------------------------------------------------------------------------
  //! OScalar = innerProduct(multi1d<source1>,multi1d<source2>))
  /*!
   * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
   *
   * Sum over the lattice
   * Allow a global sum that sums over all indices
   */
  template<class T1, class T2>
  inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t
  innerProduct(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2)
  {
    Shape shape = s1[0].shape();

    typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t  d(shape);

    // Possibly loop entered
    zero_rep(d.elem());

    for(int n=0; n < s1.size(); ++n)
      {
	d += sum(localInnerProduct(s1[n],s2[n]));
      }
    
    return d;
  }

  
  //! OScalar = sum(OScalar)  under an explicit subset
  /*! Discards subset */
  template<class T1, class T2>
  inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t
  innerProduct(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2,
	       const Subset& s)
  {
    return innerProduct(s1,s2);
  }



  //! OScalar = innerProduct(multi1d<OLattice>,multi1d<OLattice>) under an explicit subset
  /*!
   * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
   *
   * Sum over the lattice
   * Allow a global sum that sums over all indices
   */
  template<class T1, class T2>
  inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t
  innerProduct(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2,
	       const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    qdp_jit_CPU_add(__PRETTY_FUNCTION__);
#endif

    Shape shape = s1[0].shape();
    
    typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t  d(shape);

    // Possibly loop entered
    zero_rep(d.elem());

    if (multi1dOnOneLayout(s1, s2))
      {
	d = Reduce::sum_elements(s1.size(), [&](int n) { return localInnerProduct(s1[n],s2[n]); }, s);
	return d;
      }

    for(int n=0; n < s1.size(); ++n)
      {
	d += sum(localInnerProduct(s1[n],s2[n]),s);
      }
    
    return d;
  }


  //! OScalar = innerProduct(multi1d<OLattice>,multi1d<OLattice>)
  /*!
   * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
   *
   * Sum over the lattice
   * Allow a global sum that sums over all indices
   */
  template<class T1, class T2>
  inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t
  innerProduct(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
  {
    return innerProduct(s1,s2,all);
  }



  //-----------------------------------------------------------------------------
  //! OScalar = innerProductReal(multi1d<source1>,multi1d<source2>))
  /*!
   * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
   *
   * Sum over the lattice
   * Allow a global sum that sums over all indices
   */
  template<class T1, class T2>
  inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t
  innerProductReal(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2)
  {
    Shape shape = s1[0].shape();

    typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t  d(shape);

    // Possibly loop entered
    zero_rep(d.elem());

    for(int n=0; n < s1.size(); ++n)
      {
	d += sum(localInnerProductReal(s1[n],s2[n]));
      }

    return d;
  }

  //! OScalar = sum(OScalar)  under an explicit subset
  /*! Discards subset */
  template<class T1, class T2>
  inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t
  innerProductReal(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2,
		   const Subset& s)
  {
    return innerProductReal(s1,s2);
  }



  //! OScalar = innerProductReal(multi1d<OLattice>,multi1d<OLattice>) under an explicit subset
  /*!
   * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
   *
   * Sum over the lattice
   * Allow a global sum that sums over all indices
   */
  template<class T1, class T2>
  inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t
  innerProductReal(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2,
		   const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    qdp_jit_CPU_add(__PRETTY_FUNCTION__);
#endif
    
    Shape shape = s1[0].shape();
    typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t  d(shape);

    // Possibly loop entered
    zero_rep(d.elem());

    if (multi1dOnOneLayout(s1, s2))
      {
	d = Reduce::sum_elements(s1.size(), [&](int n) { return localInnerProductReal(s1[n],s2[n]); }, s);
	return d;
      }

    for(int n=0; n < s1.size(); ++n)
      {
	d += sum(localInnerProductReal(s1[n],s2[n]),s);
      }

    return d;
  }


  //! OScalar = innerProductReal(multi1d<OLattice>,multi1d<OLattice>)
  /*!
   * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
   *
   * Sum over the lattice
   * Allow a global sum that sums over all indices
   */
  template<class T1, class T2>
  inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t
  innerProductReal(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
  {
    return innerProductReal(s1,s2,all);
  }




  //-----------------------------------------------
  // Global max and min
  // NOTE: there are no subset version of these operations. It is very problematic
  // and QMP does not support them.
  //! OScalar = globalMax(OScalar)
  /*!
   * Find the maximum an object has across the lattice
   */
  template<class RHS, class T>
  typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t
  globalMax(const QDPExpr<RHS,OScalar<T> >& s1)
  {
    Shape shape = shape_of(s1);
    typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t  d(shape);

    evaluate(d,OpAssign(),s1,all);   // since OScalar, no global max needed

    return d;
  }


  //! OScalar = globalMax(OLattice)
  /*!
   * Find the maximum an object has across the lattice
   */
  template<class RHS, class T>
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
  globalMax(const QDPExpr<RHS,OLattice<T> >& s1)
  {
    OLattice<T> l(s1);
    return globalMax(l);
  }


  //! OScalar = globalMin(OScalar)
  /*!
   * Find the minimum an object has across the lattice
   */
  template<class RHS, class T>
  typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t
  globalMin(const QDPExpr<RHS,OScalar<T> >& s1)
  {
    Shape shape = shape_of(s1);
    typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t  d(shape);

    evaluate(d,OpAssign(),s1,all);   // since OScalar, no global min needed

    return d;
  }


  //! OScalar = globalMin(OLattice)
  /*!
   * Find the minimum an object has across the lattice
   */
  template<class RHS, class T>
  typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t
  globalMin(const QDPExpr<RHS,OLattice<T> >& s1)
  {
    OLattice<T> x(s1);
    OLattice<T> y(-x);
    return -globalMax(y);
  }



  template<class T>
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
  globalMax(const OLattice<T>& s1)
  {
    typedef typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t Ret_t;

    Shape shape = s1.shape();

    Ret_t d(shape);

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(d, OpAssign(), PETE_identity(s1));
    prof.start_time();
#endif

    Reduce::run<Reduce::Max, typename Ret_t::SubType_t>( PETE_identity(s1) , Reduce::Segments( all , s1.layout() ) ,
							   shape , jit_sizeof(d) , d.get_ptr() );

    QDPInternal::globalMax(d);

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif

#ifdef QDP_DEEP_LOG
    QDP_get_global_logger().log(d);
#endif
    
    return d;
  }



  template<ConceptHasNoShift RHS, class T>
  bool
  isfinite(const QDPExpr<RHS,OLattice<T> >& s1)
  {
    bool finite = true;

    Reduce::run<Reduce::AllFinite, Boolean::SubType_t>( s1 , Reduce::Segments( all , get_layout_from_rhs( s1 ) ) ,
							  shape_of( s1 ) , sizeof(finite) , &finite );

    QDPInternal::globalAnd(finite);

    return finite;
  }

  template<ConceptHasShift RHS, class T>
  bool
  isfinite(const QDPExpr<RHS,OLattice<T> >& s1)
  {
    OLattice<T> d( s1 );
    return isfinite( d );
  }

  template<class T1>
  bool
  isfinite(const OLattice<T1>& s1)
  {
    typedef typename CreateLeaf< OLattice<T1> >::Leaf_t Leaf_t;
    return isfinite( QDPExpr< Leaf_t , OLattice<T1> >( CreateLeaf< OLattice<T1> >::make(s1) ) );
  }

  

  template<class T>
  bool
  isfinite(const OScalar<T>& s1)
  {
    const typename WordType<T>::Type_t* f = s1.get_ptr();
    for ( int i = 0 ; i < WordSize<T>::value( s1.shape() ) ; ++i )
      if ( ! std::isfinite( f[i] ) )
	return false;
    return true;
  }


  //! bool = isfinite(expression): evaluated first
  template<class RHS, class T>
  bool
  isfinite(const QDPExpr<RHS,OScalar<T> >& s1)
  {
    OScalar<T> d( shape_of( s1 ) );
    evaluate( d , OpAssign() , s1 , all );
    return isfinite( d );
  }

  template<class T1> bool isnormal(const OLattice<T1>& s1) { return isfinite(s1); }
  template<class T1> bool isnan(const OLattice<T1>& s1) { return !isfinite(s1); }
  template<class T1> bool isinf(const OLattice<T1>& s1) { return !isfinite(s1); }




  template<ConceptHasNoShift RHS, class T1>
  OLattice< typename UnaryReturn<T1, FnSum >::Type_t >
  blockSum( const QDPExpr<RHS, OLattice<T1> >& rhs, const multi1d<int>& blocking )
  {
    return BlockSum::run( rhs , blocking );
  }


  template<ConceptHasShift RHS, class T1>
  OLattice< typename UnaryReturn<T1, FnSum >::Type_t >
  blockSum( const QDPExpr<RHS, OLattice<T1> >& rhs, const multi1d<int>& blocking )
  {
    OLattice<T1> tmp = rhs;
    return blockSum(tmp,blocking);
  }


  template<class T, class C>
  OLattice< typename UnaryReturn<T, FnSum >::Type_t >
  blockSum(const QDPType<T,C>& s1 , const multi1d<int>& blocking )
  {
    return blockSum( PETE_identity(s1) , blocking );
  }




  //-----------------------------------------------------------------------------
  // Peek and poke at individual sites. This is very architecture specific
  // NOTE: these two routines assume there is no underlying inner grid

  //! Extract site element
  /*! @ingroup group1
    @param l  source to examine
    @param coord Nd lattice coordinates to examine
    @return single site object of the same primitive type
    @ingroup group1
    @relates QDPType */
  template<class T1>
  inline OScalar<T1>
  peekSite(const OScalar<T1>& l, const multi1d<int>& coord)
  {
    return l;
  }

  //! Extract site element
  /*! @ingroup group1
    @param l  source to examine
    @param coord Nd lattice coordinates to examine
    @return single site object of the same primitive type
    @ingroup group1
    @relates QDPType */
  template<class RHS, class T1>
  inline OScalar<T1>
  peekSite(const QDPExpr<RHS,OScalar<T1> > & l, const multi1d<int>& coord)
  {
    // For now, simply evaluate the expression and then call the function
    typedef OScalar<T1> C1;
  
    return peekSite(C1(l), coord);
  }


  template<class T1>
  OScalar<T1>
  peekSite(const OLattice<T1>& l, const multi1d<int>& coord)
  {
    OScalar<T1> dest(l.shape());

    int nodenum = l.layout().nodeNumber(coord);

    Site::run_peek( l , dest , coord );

    QMP_barrier();

    QDPInternal::sendToPrimaryNode(dest, nodenum);

    QDPInternal::broadcast(dest);

    return dest;
  }



  template<class T1>
  OLattice<T1>&
  pokeSite(OLattice<T1>& l, const OScalar<typename ScalarType<T1>::Type_t>& r, const multi1d<int>& coord)
  {
    Site::run_poke( l , r , coord );

    QMP_barrier();

    return l;
  }

  
  

  //! Copy data values from field src to array dest
  /*! @ingroup group1
    @param dest  target to update
    @param src   QDP source to insert
    @param s     subset
    @ingroup group1
    @relates QDPType */
  template<class T , class T2>
  inline void 
  QDP_extract(multi1d<OScalar<T> >& dest, const OLattice<T2>& src, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof("QDP_extract", dest, src);
    prof.start_time();
#endif

    Extract::run( dest , src , s );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }

  //! Inserts data values from site array src.
  /*! @ingroup group1
    @param dest  QDP target to update
    @param src   source to insert
    @param s     subset
    @ingroup group1
    @relates QDPType */
  template<class T>
  inline void 
  QDP_insert(OLattice<T>& dest, const multi1d<OScalar< typename ScalarType<T>::Type_t > >& src, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnRandom(), "QDP_insert", dest);
    prof.start_time();
#endif

    Extract::run_insert( dest , src , s );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }




  
} // QDP
#endif
