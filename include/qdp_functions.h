// -*- C++ -*-

/*! @file
 * @brief Outer lattice routines specific to a parallel platform with scalar layout
 */

#ifndef QDP_FUNCTIONS_H
#define QDP_FUNCTIONS_H



namespace QDP {

  namespace Layout
  {
    //! coord[mu]  <- mu  : fill with lattice coord in mu direction
    LatticeInteger latticeCoordinate(int mu);
  }


  //-----------------------------------------------------------------------------
  //! OLattice Op Scalar(Expression(source)) under an Subset
  /*! 
   * OLattice Op Expression, where Op is some kind of binary operation 
   * involving the destination 
   */
  template<class T, class T1, class Op, class RHS>
  void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
		const Subset& s)
  {
#if defined(QDP_USE_PROFILING)	 
    static QDPProfile_t prof(dest, op, rhs);
    prof.start_time();
#endif
	
    static JitFunction function;
    if (function.empty())
      function_lat_sca_build(function ,dest, op, rhs);

    function_lat_sca_exec(function, dest, op, rhs, s);

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }

  


  template<class T, class T1, class Op, class RHS>
  void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs,
		const Subset& s)
  {
#if defined(QDP_USE_PROFILING)	 
    static QDPProfile_t prof(dest, op, rhs);
    prof.start_time();
#endif

    static JitFunctionMap function_map;

    auto key = get_dyn_key( op , rhs , s );

    //QDPIO::cout << "dyn key = " << key << std::endl;

    if (function_map[key].empty())
      function_build(function_map[key], key, dest, op, rhs, s);

    function_exec(function_map[key], dest, op, rhs, s);

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }




  template<class T, class C1, class Op, class RHS>
  void evaluate_subtype_type(OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs,
			     const Subset& s)
  {
#if defined(QDP_USE_PROFILING)	 
    static QDPProfile_t prof(dest, op, rhs);
    prof.start_time();
#endif

    static JitFunction function;

    if (function.empty())
      function_subtype_type_build(function, dest, op, rhs);

    function_subtype_type_exec(function, dest, op, rhs, s);
#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }


  

  template<class T, class T1, class Op>
  void operator_type_subtype(OLattice<T>& dest, const Op& op, const OSubLattice<T1>& rhs, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(dest, op, rhs);
    prof.start_time();
#endif

    static JitFunction function;

    if (function.empty())
      operator_type_subtype_build(function, dest, op, rhs);

    operator_type_subtype_exec(function, dest, op, rhs, s);

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }


  

  template<class T, class T1, class Op>
  void operator_subtype_subtype(OSubLattice<T>& dest, const Op& op, const OSubLattice<T1>& rhs, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(dest, op, rhs);
    prof.start_time();
#endif

    static JitFunction function;

    if (function.empty())
      operator_subtype_subtype_build(function, dest, op, rhs);

    operator_subtype_subtype_exec(function, dest, op, rhs, s);

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
}



  
  template<class T, class T1, class Op, class RHS>
  void evaluate_subtype(OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(dest, op, rhs);
    prof.start_time();
#endif

    static JitFunction function;

    if (function.empty())
      function_lat_sca_subtype_build(function, dest, op, rhs);

    function_lat_sca_subtype_exec(function, dest, op, rhs, s);

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }



  

  
  //! dest = (mask) ? s1 : dest
  template<class T1, class T2>
  void copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnCopyMask(), "copymask", dest, mask , s1);
    prof.start_time();
#endif

    static JitFunction function;

    if (function.empty())
      function_copymask_build(function, dest , mask , s1 );

    function_copymask_exec(function, dest , mask , s1 );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }



  //! dest  = random    under a subset
  template<class T>
  void 
  random(OLattice<T>& d, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnRandom(), "random", d);
    prof.start_time();
#endif
    static JitFunction function;

    Seed seed_tmp;

    if (function.empty())
      function_random_build(function, d , seed_tmp );

    function_random_exec(function, d, s , seed_tmp );

    RNG::get_RNG_Internals()->ran_seed = seed_tmp;

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }


  //! dest  = random  
  /*! This implementation is correct for no inner grid */
  template<class T>
  void 
  random(OScalar<T>& d)
  {
    Seed seed = RNG::get_RNG_Internals()->ran_seed;
    Seed skewed_seed = RNG::get_RNG_Internals()->ran_seed * RNG::get_RNG_Internals()->ran_mult;

    fill_random(d.elem(), seed, skewed_seed, RNG::get_RNG_Internals()->ran_mult);

    RNG::get_RNG_Internals()->ran_seed = seed;  // The seed from any site is the same as the new global seed
  }




  //! dest  = random  
  template<class T>
  void random(OLattice<T>& d)
  {
    random(d,all);
  }




  //! dest  = gaussian   under a subset
  template<class T>
  void gaussian(OLattice<T>& d, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnRandom(), "gaussian", d);
    prof.start_time();
#endif

    OLattice<T>  r1, r2;

    random(r1,s);
    random(r2,s);

    static JitFunction function;

    if (function.empty())
      function_gaussian_build(function, d , r1 , r2 );

    function_gaussian_exec(function, d, r1, r2, s );

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

    static JitFunction function;

    if (function.empty())
      function_zero_rep_build(function, dest );

    function_zero_rep_exec( function , dest , s );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
  }


  template<class T> 
  void zero_rep_subtype(OSubLattice<T>& dest, const Subset& s) 
  {
#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(FnRandom(), "zero_rep", dest);
    prof.start_time();
#endif

    static JitFunction function;

    if (function.empty())
      function_zero_rep_subtype_build(function, dest );

    function_zero_rep_subtype_exec( function , dest , s );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif
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
    typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d;

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
    typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d;

    evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

    return d;
  }




  template<class RHS, class T>
  typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t
  sumMulti(const QDPExpr<RHS,OScalar<T> >& s1, const Set& ss)
  {
    typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

    for(int i=0; i < ss.numSubsets(); ++i)
      evaluate(dest[i],OpAssign(),s1,all);

    return dest;
  }

#if 1
  template<class RHS, class T>
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
  sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
  {
    OLattice<T> lat;
    lat = s1;
    return sumMulti(lat,ss);
  }
#endif



  // Not sure if needed
#if 1
  template<class T>
  multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>
  sumMulti(const multi1d< OScalar<T> >& s1, const Set& ss)
  {
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
#endif


  template<class T>
  multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>
  sumMulti(const multi1d< OLattice<T> >& s1, const Set& ss)
  {
#if defined(QDP_USE_PROFILING)
    qdp_jit_CPU_add(__PRETTY_FUNCTION__);
#endif
    
    multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t> dest(s1.size(), ss.numSubsets());

    // Initialize result with zero
    for(int i=0; i < dest.size1(); ++i)
      for(int j=0; j < dest.size2(); ++j)
	zero_rep(dest(j,i));

    // Loop over all sites and accumulate based on the coloring 
    const multi1d<int>& lat_color =  ss.latticeColoring();

    for(int k=0; k < s1.size(); ++k)
      {
	const OLattice<T>& ss1 = s1[k];
	const int nodeSites = Layout::sitesOnNode();
	for(int i=0; i < nodeSites; ++i) 
	  {
	    int j = lat_color[i];
	    dest(k,j).elem() += ss1.elem(i);
	  }
      }

    // Do a global sum on the result
    QDPInternal::globalSumArray(dest);

    return dest;
  }


  template<class T>
  inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
  norm2(const multi1d< OScalar<T> >& s1)
  {
    typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t  d;

    // Possibly loop entered
    zero_rep(d.elem());

    for(int n=0; n < s1.size(); ++n)
      {
	OScalar<T>& ss1 = s1[n];
	d.elem() += localNorm2(ss1.elem());
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
#if defined(QDP_USE_PROFILING)
    qdp_jit_CPU_add(__PRETTY_FUNCTION__);
#endif
    
    typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t  d;

    // Possibly loop entered
    zero_rep(d.elem());

    const int *tab = s.siteTable().slice();
    for(int n=0; n < s1.size(); ++n)
      {
	const OLattice<T>& ss1 = s1[n];
	for(int j=0; j < s.numSiteTable(); ++j) 
	  {
	    int i = tab[j];
	    d.elem() += localNorm2(ss1.elem(i));
	  }
      }

    // Do a global sum on the result
    QDPInternal::globalSum(d);

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
    typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t  d;

    // Possibly loop entered
    zero_rep(d.elem());

    for(int n=0; n < s1.size(); ++n)
      {
	OScalar<T1>& ss1 = s1[n];
	OScalar<T2>& ss2 = s2[n];
	d.elem() += localInnerProduct(ss1.elem(),ss2.elem());
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
    
    typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t  d;

    // Possibly loop entered
    zero_rep(d.elem());

    const int *tab = s.siteTable().slice();
    for(int n=0; n < s1.size(); ++n)
      {
	const OLattice<T1>& ss1 = s1[n];
	const OLattice<T2>& ss2 = s2[n];
	for(int j=0; j < s.numSiteTable(); ++j) 
	  {
	    int i = tab[j];
	    d.elem() += localInnerProduct(ss1.elem(i),ss2.elem(i));
	  }
      }

    // Do a global sum on the result
    QDPInternal::globalSum(d);

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
    typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t  d;

    // Possibly loop entered
    zero_rep(d.elem());

    for(int n=0; n < s1.size(); ++n)
      {
	OScalar<T1>& ss1 = s1[n];
	OScalar<T2>& ss2 = s2[n];
	d.elem() += localInnerProductReal(ss1.elem(),ss2.elem());
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
    
    typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t  d;

    // Possibly loop entered
    zero_rep(d.elem());

    const int *tab = s.siteTable().slice();
    for(int n=0; n < s1.size(); ++n)
      {
	const OLattice<T1>& ss1 = s1[n];
	const OLattice<T2>& ss2 = s2[n];
	for(int j=0; j < s.numSiteTable(); ++j) 
	  {
	    int i = tab[j];
	    d.elem() += localInnerProductReal(ss1.elem(i),ss2.elem(i));
	  }
      }

    // Do a global sum on the result
    QDPInternal::globalSum(d);

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
    typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t  d;

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
    typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t  d;

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
    typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t  d;
    QDPIO::cerr << "globalMin not implemented\n";
    QDP_abort(1);
#if 0

    // Loop always entered so unroll
    d.elem() = forEach(s1, EvalLeaf1(0), OpCombine());   // SINGLE NODE VERSION FOR NOW

    const int vvol = Layout::sitesOnNode();
    for(int i=1; i < vvol; ++i) 
      {
	typename UnaryReturn<T, FnGlobalMin>::Type_t  dd = 
	  forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

	if (toBool(dd < d.elem()))
	  d.elem() = dd;
      }

    // Do a global min on the result
    QDPInternal::globalMin(d); 
#endif
    return d;
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
    dest.resize(s.numSiteTable());
    
    static JitFunction function;
    if (function.empty())
      function_extract_build(function ,dest, src);

    function_extract_exec(function, dest, src, s);

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
  QDP_insert(OLattice<T>& dest, const multi1d<OScalar<T> >& src, const Subset& s)
  {
#if defined(QDP_USE_PROFILING)
    qdp_jit_CPU_add(__PRETTY_FUNCTION__);
#endif
    
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) 
      {
	int i = tab[j];
	dest.elem(i) = src[i].elem();
      }
  }




  
} // QDP
#endif
