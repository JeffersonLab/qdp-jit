#ifndef QDP_FUNCTIONS_SPECIAL_H
#define QDP_FUNCTIONS_SPECIAL_H


namespace QDP
{


  // template<>
  // void evaluate<
  //   PScalar<PScalar<RScalar<Word<float> > > >,
  //   PScalar<PScalar<RScalar<Word<float> > > >,
  //   OpAssign,
  //   BinaryNode<OpAdd, Reference<QDPType<PScalar<PScalar<RScalar<Word<float> > > >, OLattice<PScalar<PScalar<RScalar<Word<float> > > > > > >, Reference<QDPType<PScalar<PScalar<RScalar<Word<float> > > >, OLattice<PScalar<PScalar<RScalar<Word<float> > > > > > > >
  //   >

  // void evaluate( OLattice< PScalar<PScalar<RScalar<Word<float> > > > >& dest,
  // 		 const OpAssign& op,
  // 		 const QDPExpr< BinaryNode<OpAdd, Reference<QDPType<PScalar<PScalar<RScalar<Word<float> > > >, OLattice<PScalar<PScalar<RScalar<Word<float> > > > > > >, Reference<QDPType<PScalar<PScalar<RScalar<Word<float> > > >, OLattice<PScalar<PScalar<RScalar<Word<float> > > > > > > > ,OLattice< PScalar<PScalar<RScalar<Word<float> > > > > >& rhs,
  // 		 const Subset& s);

#if 0
  void evaluate( OLattice< PScalar<PScalar<RComplex<Word<float> > > > >& dest,
  		 const OpAssign& op,
  		 const QDPExpr< BinaryNode<OpMultiply,
		 Reference<QDPType<PScalar<PScalar<RComplex<Word<float> > > >, OLattice<PScalar<PScalar<RComplex<Word<float> > > > > > >,
		 Reference<QDPType<PScalar<PScalar<RComplex<Word<float> > > >, OLattice<PScalar<PScalar<RComplex<Word<float> > > > > > > > ,
		 OLattice< PScalar<PScalar<RComplex<Word<float> > > > > >& rhs,
  		 const Subset& s);
#endif

  
  void evaluate( OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> >& dest,
		 const OpAssign& op,
		 const QDPExpr<
		 BinaryNode<FnQuarkContract13,
		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > >,
  		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > > >  ,
		 OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > >& rhs,
		 const Subset& s);

  void evaluate( OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> >& dest,
		 const OpAssign& op,
		 const QDPExpr<
		 BinaryNode<FnQuarkContract14,
		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > >,
  		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > > >  ,
		 OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > >& rhs,
		 const Subset& s);

  void evaluate( OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> >& dest,
		 const OpAssign& op,
		 const QDPExpr<
		 BinaryNode<FnQuarkContract23,
		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > >,
  		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > > >  ,
		 OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > >& rhs,
		 const Subset& s);

  void evaluate( OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> >& dest,
		 const OpAssign& op,
		 const QDPExpr<
		 BinaryNode<FnQuarkContract24,
		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > >,
  		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > > >  ,
		 OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > >& rhs,
		 const Subset& s);

}

#endif
