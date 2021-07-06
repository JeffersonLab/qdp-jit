#ifndef QDP_JITFUNC_SPINLOOP_H
#define QDP_JITFUNC_SPINLOOP_H

#include<type_traits>


namespace QDP {


  namespace
  {
    template <class T>
    void print_type()
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << std::endl;
    }
  }





  template<class T>
  struct Strip
  {
    typedef T Type_t;
  };

  template<class T,int N>
  struct Strip<Reference<PSpinMatrix<T,N> > >
  {
    typedef PSpinMatrix<float,N> Type_t;
  };

  template<class T,int N>
  struct Strip<PSpinMatrix<T,N> >
  {
    typedef PSpinMatrix<float,N> Type_t;
  };

  template<class T,int N>
  struct Strip<Reference<PSpinVector<T,N> > >
  {
    typedef PSpinVector<float,N> Type_t;
  };

  template<class T,int N>
  struct Strip<PSpinVector<T,N> >
  {
    typedef PSpinVector<float,N> Type_t;
  };

  template<class T>
  struct Strip<Reference<PScalar<T> > >
  {
    typedef PScalar<float> Type_t;
  };

  template<class T>
  struct Strip<PScalar<T> >
  {
    typedef PScalar<float> Type_t;
  };



  template<class T>
  struct HasProp
  {
    constexpr static bool value = false;
  };

  template<class Op, class A, class B>
  struct HasProp< BinaryNode<Op,A,B> >
  {
    constexpr static bool value = HasProp<A>::value || HasProp<B>::value;
  };

  template<class Op, class A, class B, class C>
  struct HasProp< TrinaryNode<Op,A,B,C> >
  {
    constexpr static bool value = HasProp<B>::value || HasProp<C>::value;
  };

  template<class Op, class A>
  struct HasProp< UnaryNode<Op,A> >
  {
    constexpr static bool value = HasProp<A>::value;
  };

  template<class T>
  struct HasProp< Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<T> >, Nc >, Ns >, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<T> >, Nc >, Ns > > > > >
  {
    constexpr static bool value = true;
  };


  template<class A>
  struct EvalToSpinMatrix
  {
    typedef typename ForEach<A , JIT2BASE , TreeCombine >::Type_t A_a;
    constexpr static bool value = is_same<typename Strip< typename ForEach<A_a, EvalLeaf1, OpCombine>::Type_t >::Type_t , PSpinMatrix<float, 4> >::value;
  };

  template<class A>
  struct EvalToSpinScalar
  {
    typedef typename ForEach<A , JIT2BASE , TreeCombine >::Type_t A_a;
    constexpr static bool value = is_same<typename Strip< typename ForEach<A_a, EvalLeaf1, OpCombine>::Type_t >::Type_t , PScalar<float> >::value;
  };




  template<typename T> concept ConceptEvalToSpinMatrix = EvalToSpinMatrix<T>::value;
  template<typename T> concept ConceptEvalToSpinScalar = EvalToSpinScalar<T>::value;


  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<OpMultiply, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, OpMultiply, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<OpMultiply, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "mul(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;

      print_type<A>();
      print_type<TypeA_t>();
      print_type<Type_t>();
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );
    
      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, OpMultiply, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , f.index_first() , index_k ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k , f.index_second() ) , c),
		  expr.operation(), c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };



  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<OpAdjMultiply, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, OpAdjMultiply, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<OpAdjMultiply, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "adjMul(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;

      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, OpAdjMultiply, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_k , f.index_first()  ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k , f.index_second() ) , c),
		  expr.operation(), c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };



  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinScalar B, class CTag>
  struct ForEach<BinaryNode<OpAdjMultiply, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, OpAdjMultiply, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<OpAdjMultiply, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "adjMul(spinmat,spinscalar) " << EvalToSpinMatrix<A>::value << std::endl;

      return Combine2<TypeA_t, TypeB_t, OpAdjMultiply, CTag>::
	combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , f.index_second() , f.index_first()  ) , c),
		ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f ) , c),
		expr.operation(), c);
    }
  };



  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<OpMultiplyAdj, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, OpMultiplyAdj, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<OpMultiplyAdj, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "mulAdj(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;

      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, OpMultiplyAdj, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , f.index_first() , index_k ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , f.index_second() , index_k ) , c),
		  expr.operation(), c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };


  template<ConceptEvalToSpinScalar A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<OpMultiplyAdj, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, OpMultiplyAdj, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<OpMultiplyAdj, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "mulAdj(spinscalar,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;

      return Combine2<TypeA_t, TypeB_t, OpMultiplyAdj, CTag>::
	combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f ) , c),
		ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , f.index_second() , f.index_first() ) , c),
		expr.operation(), c);
    }
  };


  template<ConceptEvalToSpinScalar A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<OpAdjMultiplyAdj, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, OpAdjMultiplyAdj, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<OpAdjMultiplyAdj, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "adjMulAdj(spinscalar,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;

      return Combine2<TypeA_t, TypeB_t, OpAdjMultiplyAdj, CTag>::
	combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f ) , c),
		ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , f.index_second() , f.index_first() ) , c),
		expr.operation(), c);
    }
  };


  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinScalar B, class CTag>
  struct ForEach<BinaryNode<OpAdjMultiplyAdj, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, OpAdjMultiplyAdj, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<OpAdjMultiplyAdj, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "adjMulAdj(spinmat,spinscalar) " << EvalToSpinMatrix<A>::value << std::endl;

      return Combine2<TypeA_t, TypeB_t, OpAdjMultiplyAdj, CTag>::
	combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , f.index_second() , f.index_first() ) , c),
		ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f ) , c),
		expr.operation(), c);
    }
  };





  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<OpAdjMultiplyAdj, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, OpAdjMultiplyAdj, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<OpAdjMultiplyAdj, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "adjMulAdj(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;

      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, OpAdjMultiplyAdj, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_k , f.index_first() ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , f.index_second() , index_k ) , c),
		  expr.operation(), c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };




  template<class A,class B>
  struct Combine2<PColorMatrixREG<A,3>, PColorMatrixREG<B,3>, FnQuarkContractXX, OpCombine>
  {
    typedef typename BinaryReturn<PColorMatrixREG<A,3>, PColorMatrixREG<B,3>, FnQuarkContractXX>::Type_t Type_t;
    inline static
    Type_t combine(const PColorMatrixREG<A,3>& a, const PColorMatrixREG<B,3>& b, const FnQuarkContractXX& op, OpCombine)
    {
      return quarkContractXX(a, b);
    }
  };


  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnQuarkContract13, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnQuarkContract13, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "quarkContract13(prop,prop) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // f=(i,j)
      // (A o B)^{i,j} = A^{k,i} o B^{k,j}
      
      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_k , f.index_first() ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k , f.index_second() ) , c),
		  FnQuarkContractXX() , c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };

  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnQuarkContract14, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnQuarkContract14, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "quarkContract14(prop,prop) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // f=(i,j)
      // (A o B)^{i,j} = A^{k,i} o B^{k,j}
      
      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_k , f.index_first() ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , f.index_second() , index_k ) , c),
		  FnQuarkContractXX() , c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };

  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnQuarkContract23, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnQuarkContract23, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "quarkContract23(prop,prop) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // f=(i,j)
      // (A o B)^{i,j} = A^{k,i} o B^{k,j}
      
      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , f.index_first() , index_k ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k , f.index_second() ) , c),
		  FnQuarkContractXX() , c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };

  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnQuarkContract24, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnQuarkContract24, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "quarkContract24(prop,prop) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // f=(i,j)
      // (A o B)^{i,j} = A^{k,i} o B^{k,j}
      
      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , f.index_first()  , index_k ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , f.index_second() , index_k ) , c),
		  FnQuarkContractXX() , c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };

  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnQuarkContract12, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnQuarkContract12, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "quarkContract12(prop,prop) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // f=(i,j)
      // (A o B)^{i,j} = A^{k,i} o B^{k,j}
      
      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_k         , index_k          ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , f.index_first() , f.index_second() ) , c),
		  FnQuarkContractXX() , c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };

  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnQuarkContract34, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnQuarkContract34, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "quarkContract34(prop,prop) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // f=(i,j)
      // (A o B)^{i,j} = A^{k,i} o B^{k,j}
      
      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , f.index_first() , f.index_second() ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k         , index_k          ) , c),
		  FnQuarkContractXX() , c);
      }
      index_k.end();
      return stack.elemREGint(0);
    }
  };

  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnTraceSpinQuarkContract13, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnTraceSpinQuarkContract13, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c) 
    {
      QDPIO::cout << "traceSpinQuarkContract13(prop,prop) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // f=(i,j)
      // (A o B)^{i,j} = A^{k,i} o B^{k,j}

      JitForLoop index_i(0,4);
      {
	JitForLoop index_k(0,4);
	{
	  stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnQuarkContractXX, CTag>::
	    combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_k , index_i ) , c),
		    ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k , index_i ) , c),
		    FnQuarkContractXX() , c);
	}
	index_k.end();
      }
      index_i.end();
    
      return stack.elemREGint(0);
    }
  };







  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnTraceMultiply, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnTraceMultiply, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnTraceMultiply, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "traceMultiply(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // tr(AB) = A^ik B^ki
      
      JitForLoop index_i(0,4);
      {
	JitForLoop index_k(0,4);
	{
	  stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnTraceMultiply, CTag>::
	    combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_i , index_k ) , c),
		    ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k , index_i ) , c),
		    expr.operation(), c);
	}
	index_k.end();
      }
      index_i.end();
      return stack.elemREGint(0);
    }
  };



  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnTraceSpinMultiply, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnTraceSpinMultiply, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnTraceSpinMultiply, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "traceSpinMultiply(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // tr(AB) = A^ik B^ki
      
      JitForLoop index_i(0,4);
      {
	JitForLoop index_k(0,4);
	{
	  stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, OpMultiply, CTag>::
	    combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_i , index_k ) , c),
		    ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k , index_i ) , c),
		    OpMultiply(), c);
	}
	index_k.end();
      }
      index_i.end();
      return stack.elemREGint(0);
    }
  };




  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnTraceColorMultiply, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnTraceColorMultiply, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnTraceColorMultiply, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "traceColorMultiply(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // AB = A^ik B^kj
      
      JitForLoop index_k(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnTraceColorMultiply, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , f.index_first() , index_k )          , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_k         , f.index_second() ) , c),
		  FnTraceColorMultiply(), c);
      }
      index_k.end();

      return stack.elemREGint(0);
    }
  };



  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnLocalInnerProduct, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnLocalInnerProduct, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnLocalInnerProduct, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "localInnerProduct(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // AB = A^ik B^kj
      
      JitForLoop index_i(0,4);
      {
	JitForLoop index_j(0,4);
	{
	  stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnLocalInnerProduct, CTag>::
	    combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_i , index_j ) , c),
		    ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_i , index_j ) , c),
		    FnLocalInnerProduct(), c);
	}
	index_j.end();
      }
      index_i.end();

      return stack.elemREGint(0);
    }
  };


  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinScalar B, class CTag>
  struct ForEach<BinaryNode<FnLocalInnerProduct, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnLocalInnerProduct, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnLocalInnerProduct, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "localInnerProduct(spinmat,spinscalar) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // AB = A^ii B
      
      JitForLoop index_i(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnLocalInnerProduct, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_i , index_i ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f ) , c),
		  FnLocalInnerProduct(), c);
      }
      index_i.end();

      return stack.elemREGint(0);
    }
  };


  template<ConceptEvalToSpinScalar A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnLocalInnerProduct, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnLocalInnerProduct, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnLocalInnerProduct, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "localInnerProduct(spinscalar,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // AB = A^ii B
      
      JitForLoop index_i(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnLocalInnerProduct, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_i , index_i ) , c),
		  FnLocalInnerProduct(), c);
      }
      index_i.end();

      return stack.elemREGint(0);
    }
  };


  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnLocalInnerProductReal, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnLocalInnerProductReal, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnLocalInnerProductReal, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "localInnerProduct(spinmat,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // AB = A^ik B^kj
      
      JitForLoop index_i(0,4);
      {
	JitForLoop index_j(0,4);
	{
	  stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnLocalInnerProductReal, CTag>::
	    combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_i , index_j ) , c),
		    ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_i , index_j ) , c),
		    FnLocalInnerProductReal(), c);
	}
	index_j.end();
      }
      index_i.end();

      return stack.elemREGint(0);
    }
  };


  template<ConceptEvalToSpinMatrix A, ConceptEvalToSpinScalar B, class CTag>
  struct ForEach<BinaryNode<FnLocalInnerProductReal, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnLocalInnerProductReal, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnLocalInnerProductReal, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "localInnerProduct(spinmat,spinscalar) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // AB = A^ii B
      
      JitForLoop index_i(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnLocalInnerProductReal, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f , index_i , index_i ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f ) , c),
		  FnLocalInnerProductReal(), c);
      }
      index_i.end();

      return stack.elemREGint(0);
    }
  };


  template<ConceptEvalToSpinScalar A, ConceptEvalToSpinMatrix B, class CTag>
  struct ForEach<BinaryNode<FnLocalInnerProductReal, A, B>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename ForEach<B, ViewSpinLeaf, CTag>::Type_t TypeB_t;
    typedef typename Combine2<TypeA_t, TypeB_t, FnLocalInnerProductReal, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const BinaryNode<FnLocalInnerProductReal, A, B> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "localInnerProduct(spinscalar,spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // AB = A^ii B
      
      JitForLoop index_i(0,4);
      {
	stack.elemJITint(0) += Combine2<TypeA_t, TypeB_t, FnLocalInnerProductReal, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.left(),  ViewSpinLeaf( f ) , c),
		  ForEach<B, ViewSpinLeaf, CTag>::apply(expr.right(), ViewSpinLeaf( f , index_i , index_i ) , c),
		  FnLocalInnerProductReal(), c);
      }
      index_i.end();

      return stack.elemREGint(0);
    }
  };







  template<ConceptEvalToSpinMatrix A, class CTag>
  struct ForEach<UnaryNode<FnRealTrace, A>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnRealTrace, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const UnaryNode<FnRealTrace, A> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "realTrace(spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // tr(A) = A^ii
      
      JitForLoop index_i(0,4);
      {
	stack.elemJITint(0) += Combine1<TypeA_t, FnRealTrace, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.child(),  ViewSpinLeaf( f , index_i , index_i ) , c),
		  expr.operation(), c);
      }
      index_i.end();
      return stack.elemREGint(0);
    }
  };




  template<ConceptEvalToSpinMatrix A, class CTag>
  struct ForEach<UnaryNode<FnTrace, A>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnTrace, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const UnaryNode<FnTrace, A> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "trace(spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // tr(A) = A^ii
      
      JitForLoop index_i(0,4);
      {
	stack.elemJITint(0) += Combine1<TypeA_t, FnTrace, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.child(),  ViewSpinLeaf( f , index_i , index_i ) , c),
		  expr.operation(), c);
      }
      index_i.end();
      return stack.elemREGint(0);
    }
  };



  template<ConceptEvalToSpinMatrix A, class CTag>
  struct ForEach<UnaryNode<FnAdjoint, A>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnAdjoint, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const UnaryNode<FnAdjoint, A> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "adj(spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      return Combine1<TypeA_t, FnAdjoint, CTag>::
	combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.child(),  ViewSpinLeaf( f , f.index_second() , f.index_first() ) , c),
		expr.operation(), c);
    }
  };



  template<ConceptEvalToSpinMatrix A, class CTag>
  struct ForEach<UnaryNode<FnImagTrace, A>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnImagTrace, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const UnaryNode<FnImagTrace, A> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "imagTrace(spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // tr(A) = A^ii
      
      JitForLoop index_i(0,4);
      {
	stack.elemJITint(0) += Combine1<TypeA_t, FnImagTrace, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.child(),  ViewSpinLeaf( f , index_i , index_i ) , c),
		  expr.operation(), c);
      }
      index_i.end();
      return stack.elemREGint(0);
    }
  };


  template<ConceptEvalToSpinMatrix A, class CTag>
  struct ForEach<UnaryNode<FnTraceSpin, A>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnTraceSpin, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const UnaryNode<FnTraceSpin, A> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "traceSpin(spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // tr(A) = A^ii
      
      JitForLoop index_i(0,4);
      {
	stack.elemJITint(0) += Combine1<TypeA_t, OpIdentity, CTag>::
	  combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.child(),  ViewSpinLeaf( f , index_i , index_i ) , c),
		  OpIdentity(), c);
      }
      index_i.end();
      return stack.elemREGint(0);
    }
  };



  template<ConceptEvalToSpinMatrix A, class CTag>
  struct ForEach<UnaryNode<FnTransposeSpin, A>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnTransposeSpin, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const UnaryNode<FnTransposeSpin, A> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "transposeSpin(spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      // A^ij = A^ji
      
      return Combine1<TypeA_t, OpIdentity, CTag>::
	combine(ForEach<A, ViewSpinLeaf, CTag>::
		apply(expr.child(),  ViewSpinLeaf( f , f.index_second() , f.index_first() ) , c),
		OpIdentity(), c);

    }
  };




  //template<ConceptEvalToSpinMatrix A,class CTag>
  template<class A,class CTag>
  struct ForEach<UnaryNode<FnPeekSpinMatrixREG, A>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnPeekSpinMatrixREG, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const UnaryNode<FnPeekSpinMatrixREG, A> &expr, const ViewSpinLeaf &f, const CTag &c)
    {
      QDPIO::cout << "peekSpinMatrix(spinmat) " << std::endl;
      print_type<TypeA_t>();
    
      return expr.child().
	elem( f.getLayout() , f.getIndex() ).
	getRegElem( expr.operation().get_row_jit() , expr.operation().get_col_jit() );
    }
  };



  template<ConceptEvalToSpinMatrix A, class CTag>
  struct ForEach<UnaryNode<FnLocalNorm2, A>, ViewSpinLeaf, CTag >
  {
    typedef typename ForEach<A, ViewSpinLeaf, CTag>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnLocalNorm2, CTag>::Type_t Type_t;

    inline static
    Type_t apply(const UnaryNode<FnLocalNorm2, A> &expr, const ViewSpinLeaf &f,	const CTag &c)
    {
      QDPIO::cout << "localNorm2(spinmat) " << EvalToSpinMatrix<A>::value << std::endl;
    
      JitStackArray< Type_t , 1 > stack;
      zero_rep( stack.elemJITint(0) );

      // localNorm2(A) = sum_ij A^ij
      
      JitForLoop index_i(0,4);
      {
	JitForLoop index_j(0,4);
	{
	  stack.elemJITint(0) += Combine1<TypeA_t, FnLocalNorm2, CTag>::
	    combine(ForEach<A, ViewSpinLeaf, CTag>::apply(expr.child(),  ViewSpinLeaf( f , index_i , index_j ) , c),
		    expr.operation(), c);
	}
	index_j.end();
      }
      index_i.end();
      return stack.elemREGint(0);
    }
  };




  template<class T,int N>
  T viewSpinJit( OLatticeJIT< PSpinMatrixJIT<T,N> >& dest , const ViewSpinLeaf& v )
  {
    if (v.loops.size() != 2)
      {
	QDPIO::cout << "at viewSpinJit(spinmat) but not 2 indices provided" << std::endl;
	QDP_abort(1);
      }
  
    return dest.elem( v.getLayout() , v.getIndex() ).getJitElem( v.loops[0].index() , v.loops[1].index() );
  }


  template<class T,int N>
  T viewSpinJit( OLatticeJIT< PSpinVectorJIT<T,N> >& dest , const ViewSpinLeaf& v )
  {
    if (v.loops.size() != 1)
      {
	QDPIO::cout << "at viewSpinJit(spinvec) but not 1 index provided" << std::endl;
	QDP_abort(1);
      }
  
    return dest.elem( v.getLayout() , v.getIndex() ).getJitElem( v.loops[0].index() );
  }


  template<class T>
  T viewSpinJit( OLatticeJIT< PScalarJIT<T> >& dest , const ViewSpinLeaf& v )
  {
    if (v.loops.size() != 0)
      {
	QDPIO::cout << "at viewSpinJit(pscalar) but not 0 indices provided " << std::endl;
	QDP_abort(1);
      }
  
    return dest.elem( v.getLayout() , v.getIndex() ).getJitElem();
  }


  

  template<class T, class Op>
  struct CreateLoops
  {
    static void apply( std::vector< JitForLoop >& loops , const Op& op )
    {
      QDPIO::cout << "dest: no spin loops created\n";
    }
  };


  template<class T, int N, class Op>
  struct CreateLoops< PSpinMatrix<T,N> , Op >
  {
    static void apply( std::vector< JitForLoop >& loops , const Op& op )
    {
      QDPIO::cout << "create spin mat\n";
	
      QDPIO::cout << "loop(0,N)\n";
      loops.push_back( JitForLoop(0,N) );
  
      QDPIO::cout << "loop(0,N)\n";
      loops.push_back( JitForLoop(0,N) );
    }
  };


  template<class T, int N>
  struct CreateLoops< PSpinMatrix<T,N> , FnPokeSpinMatrixREG >
  {
    static void apply( std::vector< JitForLoop >& loops , const FnPokeSpinMatrixREG& op )
    {
      QDPIO::cout << "pokeSpinMat with two 1-loops\n";

      llvm::Value* row = op.jitRow();
      llvm::Value* col = op.jitCol();
  
      loops.push_back( JitForLoop( row , llvm_add( row , llvm_create_value( 1 ) ) ) );
      loops.push_back( JitForLoop( col , llvm_add( col , llvm_create_value( 1 ) ) ) );
    }
  };

} // QDP
#endif
