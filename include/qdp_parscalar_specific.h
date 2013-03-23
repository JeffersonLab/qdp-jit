// -*- C++ -*-

/*! @file
 * @brief Outer lattice routines specific to a parallel platform with scalar layout
 */

#ifndef QDP_PARSCALAR_SPECIFIC_H
#define QDP_PARSCALAR_SPECIFIC_H

#include "qmp.h"

namespace QDP {

// Use separate defs here. This will cause subroutine calls under g++

//-----------------------------------------------------------------------------
// Layout stuff specific to a parallel architecture

namespace Layout
{
  //! coord[mu]  <- mu  : fill with lattice coord in mu direction
  LatticeInteger latticeCoordinate(int mu);
}

//-----------------------------------------------------------------------------
// Internal ops with ties to QMP

/////////////////////////////////////////////////////////
// Threading evaluate with openmp and qmt implementation
//
// by Xu Guo, EPCC, 16 June 2008
/////////////////////////////////////////////////////////

//! user argument for the evaluate function:
// "OLattice Op Scalar(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
struct u_arg{
    u_arg(
        OLattice<T>& d_,
        const QDPExpr<RHS,OScalar<T1> >& r_,
        const Op& op_,
        const int *tab_
    ) : d(d_), r(r_), op(op_), tab(tab_) {}
    
    OLattice<T>& d;
    const QDPExpr<RHS,OScalar<T1> >& r;
    const Op& op;
    const int *tab;
   };

//! user function for the evaluate function:
// "OLattice Op Scalar(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
void ev_userfunc(int lo, int hi, int myId, u_arg<T,T1,Op,RHS> *a)
{
   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OScalar<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const Op& op= a->op;

      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
   }
}

//! user argument for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
struct user_arg{
    user_arg(
        OLattice<T>& d_,
        const QDPExpr<RHS,OLattice<T1> >& r_,
        const Op& op_,
        const int *tab_ ) : d(d_), r(r_), op(op_), tab(tab_) {}

        OLattice<T>& d;
        const QDPExpr<RHS,OLattice<T1> >& r;
        const Op& op;
        const int *tab;
   };

//! user function for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
void evaluate_userfunc(int lo, int hi, int myId, user_arg<T,T1,Op,RHS> *a)
{

   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OLattice<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const Op& op= a->op;

      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
   }
}

//! include the header file for dispatch
#include "qdp_dispatch.h"



//-----------------------------------------------------------------------------
//! OLattice Op Scalar(Expression(source)) under an Subset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
	      const Subset& s)
{
#if 1
  //std::cout << __PRETTY_FUNCTION__ << "\n";

// cerr << "In evaluateSubset(olattice,oscalar)\n";

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

  static CUfunction function;

  // Build the function
  if (function == NULL)
    {
      //std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
      function = function_lat_sca_build(dest, op, rhs);
      //std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
    }
  else
    {
      //std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
    }

  // Execute the function
  function_lat_sca_exec(function, dest, op, rhs, s);


  // int numSiteTable = s.numSiteTable();
  // u_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());
  // dispatch_to_threads< u_arg<T,T1,Op,RHS> >(numSiteTable, a, ev_userfunc);

#endif

#if 0
  ///////////////////
  // Original code
  //////////////////

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    //fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
    op(dest.elem(i), forEach(rhs, ElemLeaf(), OpCombine()));
    //  op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif
#endif
}


//! OLattice Op OLattice(Expression(source)) under an Subset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs,
	      const Subset& s)
{
#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

  //
  // Do the cross check with CPU result ?
  //
#if 0
  OLattice<T> dest0;
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      OpAssign()( dest0.elem(i) , dest.elem(i) );
    }

  static CUfunction function;

  // Build the function
  if (function == NULL)
    {
      //std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
      function = function_build(dest0, op, rhs);
      //std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
    }
  else
    {
      //std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
    }

  // Execute the function
  function_exec(function, dest0, op, rhs, s);

  //const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
    }


  size_t diffs=0;
  for(int j=0; j < s.numSiteTable(); ++j) {
    int i = tab[j];
    typename WordType<T>::Type_t * f_cpu = (typename WordType<T>::Type_t *)&dest0.elem(i);
    typename WordType<T>::Type_t * f_gpu = (typename WordType<T>::Type_t *)&dest.elem(i);
    for(int w=0 ; w < sizeof(T)/sizeof(typename WordType<T>::Type_t) ; w++ ) {
      bool different = false;
      if (f_gpu[w] != 0) {
	if ( ( fabs(f_gpu[w]) > 1.0e-4 ) &&
	     ( fabs(f_cpu[w]) > 1.0e-4 ) )
	  if ( fabs(f_cpu[w]/f_gpu[w]-1.0) > 0.15 ) 
	    different = true;
      } else {
	if ( fabs(f_cpu[w]) > 1.0e-5 )
	  different = true;
      }

      if (different) {      
	diffs++;
	std::cout << "site = " << i 
		  << "   cpu = " << f_cpu[w] 
		  << "   gpu = " << f_gpu[w] 
		  << "   factor = " << f_cpu[w]/f_gpu[w]
		  << "    diff = " << f_cpu[w] - f_gpu[w] << "\n";
      }
    }
    if (diffs > 10000)
      break;
  }

  if (diffs > 0) {
    std::cout << __PRETTY_FUNCTION__ << " numsitetable = " << s.numSiteTable() << "\n";    
    QDP_error_exit("Differences!");
  }
#else
  static CUfunction function;

  // Build the function
  if (function == NULL)
    {
      //std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
      function = function_build(dest, op, rhs);
      //std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
    }
  else
    {
      //std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
    }

  // Execute the function
  function_exec(function, dest, op, rhs, s);
#endif


#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif
}


//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T1, class T2>
void copymask(OSubLattice<T2> d, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  OLattice<T2>& dest = d.field();
  const Subset& s = d.subset();

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    copymask(dest.elem(i), mask.elem(i), s1.elem(i));
  }
}

//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  //std::cout << __PRETTY_FUNCTION__ << "\n";
  static CUfunction function;
  // Build the function
  if (function == NULL)
    {
      //std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
      function = function_copymask_build( dest , mask , s1 );
      //std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
    }
  else
    {
      //std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
    }

  // Execute the function
  function_copymask_exec(function, dest , mask , s1 );

  // int nodeSites = Layout::sitesOnNode();
  // for(int i=0; i < nodeSites; ++i) 
  //   copymask(dest.elem(i), mask.elem(i), s1.elem(i));
}


#if 0
//-----------------------------------------------------------------------------
// Random numbers
namespace RNG
{
  extern Seed ran_seed;
  extern Seed ran_mult;
  extern Seed ran_mult_n;
  extern LatticeSeed *lattice_ran_mult;
}
#endif


//! dest  = random  
/*! This implementation is correct for no inner grid */
template<class T>
void 
random(OScalar<T>& d)
{
  Seed seed = RNG::ran_seed;
  Seed skewed_seed = RNG::ran_seed * RNG::ran_mult;

  fill_random(d.elem(), seed, skewed_seed, RNG::ran_mult);

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}


//! dest  = random    under a subset
template<class T>
void 
random(OLattice<T>& d, const Subset& s)
{
  assert(!"ni");
#if 0
  static CUfunction function;

  Seed seed_tmp;

  // Build the function
  if (function == NULL)
    {
      //std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
      function = function_random_build( d , seed_tmp );
      //std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
    }
  else
    {
      //std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
    }

  // Execute the function
  function_random_exec(function, d, s , seed_tmp );

  RNG::ran_seed = seed_tmp;

#if 0
  Seed seed;
  Seed skewed_seed;

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    seed = RNG::ran_seed;
    skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(i);
    fill_random(d.elem(i), seed, skewed_seed, RNG::ran_mult_n);
  }

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
#endif
#endif
}



//! dest  = random   under a subset
template<class T>
void random(OSubLattice<T> dd)
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();

  random(d,s);
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
  assert(!"ni");
#if 0
  OLattice<T>  r1, r2;

  random(r1,s);
  random(r2,s);

  static CUfunction function;

  // Build the function
  if (function == NULL)
    {
      //std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
      function = function_gaussian_build( d , r1 , r2 );
      //std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
    }
  else
    {
      //std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
    }

  // Execute the function
  function_gaussian_exec(function, d, r1, r2, s );

#if 0
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
  }
#endif
#endif
}



//! dest  = gaussian   under a subset
template<class T>
void gaussian(OSubLattice<T> dd)
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();

  gaussian(d,s);
}


//! dest  = gaussian
template<class T>
void gaussian(OLattice<T>& d)
{
  gaussian(d,all);
}



//-----------------------------------------------------------------------------
// Broadcast operations
//! dest  = 0 
template<class T> 
inline
void zero_rep(OLattice<T>& dest, const Subset& s) 
{
#if 1
  static CUfunction function;

  if (function == NULL)
    {
      function = function_zero_rep_build( dest );
    }
  else
    {
      //std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
    }

  function_zero_rep_exec( function , dest , s );

#else
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    zero_rep(dest.elem(i));
  }
#endif
}


//! dest  = 0 
template<class T>
void zero_rep(OSubLattice<T> dd) 
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();
  
  zero_rep(d,s);
}


//! dest  = 0 
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

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = sum(OLattice)  under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
#if 1
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  //QDP_info("sum(Expr,Subset)");
  OLattice<T> l;
  l[s]=s1;
  return sum(l,s);
}
#else
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());
  }

  // Do a global sum on the result
  QDPInternal::globalSum(d);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}
#endif

//! OScalar = sum(OLattice)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
#if 1
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1)
{
  OLattice<T> l;
  l=s1;
  return sum(l,all);
}
#else
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1)
{
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered - could unroll
  zero_rep(d.elem());
  const int nodeSites = Layout::sitesOnNode();

  for(int i=0; i < nodeSites; ++i) 
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());

  // Do a global sum on the result
  QDPInternal::globalSum(d);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}
#endif

//-----------------------------------------------------------------------------
// Multiple global sums 
//! multi1d<OScalar> dest  = sumMulti(OScalar,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t
sumMulti(const QDPExpr<RHS,OScalar<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
    evaluate(dest[i],OpAssign(),s1,all);


#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//! multi1d<OScalar> dest  = sumMulti(OLattice,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
{
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Initialize result with zero
  for(int k=0; k < ss.numSubsets(); ++k)
    zero_rep(dest[k]);

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();
  const int nodeSites = Layout::sitesOnNode();

  for(int i=0; i < nodeSites; ++i) 
  {
    int j = lat_color[i];
    dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());
  }

  // Do a global sum on the result
  QDPInternal::globalSumArray(dest);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//-----------------------------------------------------------------------------
// Multiple global sums on an array
//! multi2d<OScalar> dest  = sumMulti(multi1d<OScalar>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class T>
multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>
sumMulti(const multi1d< OScalar<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t> dest(s1.size(), ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest(0,0), OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < dest.size1(); ++i)
    for(int j=0; j < dest.size2(); ++j)
      dest(j,i) = s1[j];

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//! multi2d<OScalar> dest  = sumMulti(multi1d<OLattice>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class T>
multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>
sumMulti(const multi1d< OLattice<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t> dest(s1.size(), ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

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

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//-----------------------------------------------------------------------------
//! OScalar = norm2(trace(adj(multi1d<source>)*multi1d<source>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
norm2(const multi1d< OScalar<T> >& s1)
{
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnNorm2(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T>& ss1 = s1[n];
    d.elem() += localNorm2(ss1.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnNorm2(), s1[0]);
  prof.time -= getClockTime();
#endif

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

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T1>& ss1 = s1[n];
    OScalar<T2>& ss2 = s2[n];
    d.elem() += localInnerProduct(ss1.elem(),ss2.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0]);
  prof.time -= getClockTime();
#endif

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

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T1>& ss1 = s1[n];
    OScalar<T2>& ss2 = s2[n];
    d.elem() += localInnerProductReal(ss1.elem(),ss2.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0]);
  prof.time -= getClockTime();
#endif

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

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global max needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = globalMax(OLattice)
/*!
 * Find the maximum an object has across the lattice
 */
#if 0
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
globalMax(const QDPExpr<RHS,OLattice<T> >& s1)
{
  OLattice<T> l(s1);
  return globalMax(l);
}
#else
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
globalMax(const QDPExpr<RHS,OLattice<T> >& s1)
{
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered so unroll
  d.elem() = forEach(s1, EvalLeaf1(0), OpCombine());   // SINGLE NODE VERSION FOR NOW

  const int vvol = Layout::sitesOnNode();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMax>::Type_t  dd = 
      forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

    if (toBool(dd > d.elem()))
      d.elem() = dd;
  }

  // Do a global max on the result
  QDPInternal::globalMax(d); 

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}
#endif


//! OScalar = globalMin(OScalar)
/*!
 * Find the minimum an object has across the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t
globalMin(const QDPExpr<RHS,OScalar<T> >& s1)
{
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global min needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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
  std::cout << __PRETTY_FUNCTION__ << "\n";

  typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

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

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
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



//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class T1>
inline OScalar<T1>
peekSite(const OLattice<T1>& l, const multi1d<int>& coord)
{
  std::cout << __PRETTY_FUNCTION__ << "\n";

  OScalar<T1> dest;
  int nodenum = Layout::nodeNumber(coord);

  // Find the result somewhere within the machine.
  // Then we must get it to node zero so we can broadcast it
  // out to all nodes
  if (Layout::nodeNumber() == nodenum)
    dest.elem() = l.elem(Layout::linearSiteIndex(coord));
  else
    zero_rep(dest.elem());

  // Send result to primary node via some mechanism
  QDPInternal::sendToPrimaryNode(dest, nodenum);

  // Now broadcast back out to all nodes
  QDPInternal::broadcast(dest);

  return dest;
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
peekSite(const QDPExpr<RHS,OLattice<T1> > & l, const multi1d<int>& coord)
{
  // For now, simply evaluate the expression and then call the function
  typedef OLattice<T1> C1;
  
  return peekSite(C1(l), coord);
}


//! Insert site element
/*! @ingroup group1
  @param l  target to update
  @param r  source to insert
  @param coord Nd lattice coordinates where to insert
  @return object of the same primitive type but of promoted lattice type
  @ingroup group1
  @relates QDPType */
template<class T1>
inline OLattice<T1>&
pokeSite(OLattice<T1>& l, const OScalar<T1>& r, const multi1d<int>& coord)
{
  std::cout << __PRETTY_FUNCTION__ << "\n";

  if (Layout::nodeNumber() == Layout::nodeNumber(coord))
    l.elem(Layout::linearSiteIndex(coord)) = r.elem();

  return l;
}


//! Copy data values from field src to array dest
/*! @ingroup group1
  @param dest  target to update
  @param src   QDP source to insert
  @param s     subset
  @ingroup group1
  @relates QDPType */
template<class T>
inline void 
QDP_extract(multi1d<OScalar<T> >& dest, const OLattice<T>& src, const Subset& s)
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest[i].elem() = src.elem(i);
  }
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
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest.elem(i) = src[i].elem();
  }
}









//-----------------------------------------------------------------------------

//! Binary output
/*! Assumes no inner grid */
template<class T>
inline
void write(BinaryWriter& bin, const OScalar<T>& d)
{
  bin.writeArray((const char *)&(d.elem()), 
		 sizeof(typename WordType<T>::Type_t), 
		 sizeof(T) / sizeof(typename WordType<T>::Type_t));
}


//! Binary input
/*! Assumes no inner grid */
template<class T>
void read(BinaryReader& bin, OScalar<T>& d)
{
  bin.readArray((char*)&(d.elem()), 
		sizeof(typename WordType<T>::Type_t), 
		sizeof(T) / sizeof(typename WordType<T>::Type_t)); 
}



// There are 2 main classes of binary/xml reader/writer methods.
// The first is a simple/portable but inefficient method of send/recv
// to/from the destination node.
// The second method (the else) is a more efficient roll-around method.
// However, this method more constrains the data layout - it must be
// close to the original lexicographic order.
// For now, use the direct send method

//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

//! XML output
template<class T>  
XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
{
  T recv_buf;

  xml.openTag("OLattice");
  XMLWriterAPI::AttributeList alist;

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);

    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
      recv_buf = d.elem(linear);

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
#if 1
      // All nodes participate
      QDPInternal::route((void *)&recv_buf, node, 0, sizeof(T));
#else
      if (Layout::primaryNode())
	QDPInternal::recvFromWait((void *)&recv_buf, node, sizeof(T));

      if (Layout::nodeNumber() == node)
	QDPInternal::sendToWait((void *)&recv_buf, 0, sizeof(T));
#endif
    }

    if (Layout::primaryNode())
    {
      std::ostringstream os;
      os << coord[0];
      for(int i=1; i < coord.size(); ++i)
	os << " " << coord[i];

      alist.clear();
      alist.push_back(XMLWriterAPI::Attribute("site", site));
      alist.push_back(XMLWriterAPI::Attribute("coord", os.str()));

      xml.openTag("elem", alist);
      xml << recv_buf;
      xml.closeTag();
    }
  }

  xml.closeTag(); // OLattice
  return xml;
}


//! Write a lattice quantity
/*! This code assumes no inner grid */
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb);

//! Binary output
/*! Assumes no inner grid */
template<class T>
void write(BinaryWriter& bin, const OLattice<T>& d)
{
  writeOLattice(bin, (const char *)&(d.elem(0)), 
		sizeof(typename WordType<T>::Type_t), 
		sizeof(T) / sizeof(typename WordType<T>::Type_t));
}

//! Write a single site of a lattice quantity
/*! This code assumes no inner grid */
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb,
		   const multi1d<int>& coord);

//! Write a single site of a lattice quantity
/*! Assumes no inner grid */
template<class T>
void write(BinaryWriter& bin, const OLattice<T>& d, const multi1d<int>& coord)
{
  writeOLattice(bin, (const char *)&(d.elem(0)), 
		sizeof(typename WordType<T>::Type_t), 
		sizeof(T) / sizeof(typename WordType<T>::Type_t),
		coord);
}

//! Write a single site of a lattice quantity
/*! This code assumes no inner grid */
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb,
		   const Subset& sub);

//! Write a single site of a lattice quantity
/*! Assumes no inner grid */
template<class T>
void write(BinaryWriter& bin, OSubLattice<T> dd)
{
  const OLattice<T>& d = dd.field();

  writeOLattice(bin, (const char *)&(d.elem(0)), 
		sizeof(typename WordType<T>::Type_t), 
		sizeof(T) / sizeof(typename WordType<T>::Type_t),
		dd.subset());
}


//! Read a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb);

//! Binary input
/*! Assumes no inner grid */
template<class T>
void read(BinaryReader& bin, OLattice<T>& d)
{
  readOLattice(bin, (char *)&(d.elem(0)), 
	       sizeof(typename WordType<T>::Type_t), 
	       sizeof(T) / sizeof(typename WordType<T>::Type_t));
}

//! Read a single site of a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb, 
		  const multi1d<int>& coord);

//! Read a single site of a lattice quantity
/*! Assumes no inner grid */
template<class T>
void read(BinaryReader& bin, OLattice<T>& d, const multi1d<int>& coord)
{
  readOLattice(bin, (char *)&(d.elem(0)), 
	       sizeof(typename WordType<T>::Type_t), 
	       sizeof(T) / sizeof(typename WordType<T>::Type_t),
	       coord);
}

//! Read a single site of a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb, 
		  const Subset& sub);

//! Read a single site of a lattice quantity
/*! Assumes no inner grid */
template<class T>
void read(BinaryReader& bin, OSubLattice<T> d)
{
  readOLattice(bin, (char *)(d.field().getF()),
	       sizeof(typename WordType<T>::Type_t), 
	       sizeof(T) / sizeof(typename WordType<T>::Type_t),
	       d.subset());
}



// **************************************************************
// Special support for slices of a lattice
namespace LatticeTimeSliceIO 
{
  //! Lattice time slice reader
  void readOLatticeSlice(BinaryReader& bin, char* data, 
			 size_t size, size_t nmemb,
			 int start_lexico, int stop_lexico);

  void writeOLatticeSlice(BinaryWriter& bin, const char* data, 
			  size_t size, size_t nmemb,
			  int start_lexico, int stop_lexico);


  // Read a time slice of a lattice quantity (time must be most slowly varying)
  template<class T>
  void readSlice(BinaryReader& bin, OLattice<T>& data, 
		 int start_lexico, int stop_lexico)
  {
    readOLatticeSlice(bin, (char *)&(data.elem(0)), 
		      sizeof(typename WordType<T>::Type_t), 
		      sizeof(T) / sizeof(typename WordType<T>::Type_t),
		      start_lexico, stop_lexico);
  }


  // Write a time slice of a lattice quantity (time must be most slowly varying)
  template<class T>
  void writeSlice(BinaryWriter& bin, const OLattice<T>& data, 
		  int start_lexico, int stop_lexico)
  {
    writeOLatticeSlice(bin, (const char *)&(data.elem(0)), 
		       sizeof(typename WordType<T>::Type_t), 
		       sizeof(T) / sizeof(typename WordType<T>::Type_t),
		       start_lexico, stop_lexico);
  }

} // namespace LatticeTimeSliceIO

} // namespace QDP
#endif
