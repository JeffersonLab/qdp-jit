#include "qdp.h"
#include "cuda_special.h"

namespace QDP
{

  //void QDP::evaluate(QDP::OLattice<T>&, const Op&, const QDP::QDPExpr<RHS, QDP::OLattice<T1> >&, const QDP::Subset&) [with T = QDP::PSpinMatrix<QDP::PColorMatrix<QDP::RComplex<QDP::Word<float> >, 3>, 4>; T1 = QDP::PSpinMatrix<QDP::PColorMatrix<QDP::RComplex<QDP::Word<float> >, 3>, 4>; Op = QDP::OpAssign; RHS = QDP::BinaryNode<QDP::FnQuarkContract13, QDP::Reference<QDP::QDPType<QDP::PSpinMatrix<QDP::PColorMatrix<QDP::RComplex<QDP::Word<float> >, 3>, 4>, QDP::OLattice<QDP::PSpinMatrix<QDP::PColorMatrix<QDP::RComplex<QDP::Word<float> >, 3>, 4> > > >, QDP::Reference<QDP::QDPType<QDP::PSpinMatrix<QDP::PColorMatrix<QDP::RComplex<QDP::Word<float> >, 3>, 4>, QDP::OLattice<QDP::PSpinMatrix<QDP::PColorMatrix<QDP::RComplex<QDP::Word<float> >, 3>, 4> > > > >]

  void evaluate( OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> >& dest,
		 const OpAssign& op,
		 const QDPExpr<
		 BinaryNode<FnQuarkContract13,
		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > >,
  		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > > >  ,
		 OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > >& rhs,
		 const Subset& s)
  {
    //QDPIO::cout << "in template specialization quarkContract13 \n";
    
    typedef PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> T;
    
    std::vector<QDPCache::ArgKey> ids = {
      dest.getId(),
      static_cast<const OLattice<T>*>(&rhs.expression().left())->getId(),
      static_cast<const OLattice<T>*>(&rhs.expression().right())->getId()
    };

    std::vector<void*> args = QDP_get_global_cache().get_kernel_args( ids , false );

    // std::cout << args[0] << "\n";
    // std::cout << args[1] << "\n";
    // std::cout << args[2] << "\n";

    evaluate_special_quarkContract13( Layout::sitesOnNode() , args );    
  }


  void evaluate( OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> >& dest,
		 const OpAssign& op,
		 const QDPExpr<
		 BinaryNode<FnQuarkContract14,
		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > >,
  		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > > >  ,
		 OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > >& rhs,
		 const Subset& s)
  {
    //QDPIO::cout << "in template specialization quarkContract14 \n";

    typedef PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> T;

    std::vector<QDPCache::ArgKey> ids = {
      dest.getId(),
      static_cast<const OLattice<T>*>(&rhs.expression().left())->getId(),
      static_cast<const OLattice<T>*>(&rhs.expression().right())->getId()
    };

    std::vector<void*> args = QDP_get_global_cache().get_kernel_args( ids , false );

    // std::cout << args[0] << "\n";
    // std::cout << args[1] << "\n";
    // std::cout << args[2] << "\n";

    evaluate_special_quarkContract14( Layout::sitesOnNode() , args );    
  }


  void evaluate( OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> >& dest,
		 const OpAssign& op,
		 const QDPExpr<
		 BinaryNode<FnQuarkContract23,
		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > >,
  		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > > >  ,
		 OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > >& rhs,
		 const Subset& s)
  {
    //QDPIO::cout << "in template specialization quarkContract23 \n";

    typedef PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> T;

    std::vector<QDPCache::ArgKey> ids = {
      dest.getId(),
      static_cast<const OLattice<T>*>(&rhs.expression().left())->getId(),
      static_cast<const OLattice<T>*>(&rhs.expression().right())->getId()
    };

    std::vector<void*> args = QDP_get_global_cache().get_kernel_args( ids , false );

    // std::cout << args[0] << "\n";
    // std::cout << args[1] << "\n";
    // std::cout << args[2] << "\n";

    evaluate_special_quarkContract23( Layout::sitesOnNode() , args );
  }

  
  void evaluate( OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> >& dest,
		 const OpAssign& op,
		 const QDPExpr<
		 BinaryNode<FnQuarkContract24,
		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > >,
  		   Reference<QDPType<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4>, OLattice<PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > > > >  ,
		 OLattice< PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> > >& rhs,
		 const Subset& s)
  {
    //QDPIO::cout << "in template specialization quarkContract24 \n";

    typedef PSpinMatrix<PColorMatrix<RComplex<Word<float> >, 3>, 4> T;

    std::vector<QDPCache::ArgKey> ids = {
      dest.getId(),
      static_cast<const OLattice<T>*>(&rhs.expression().left())->getId(),
      static_cast<const OLattice<T>*>(&rhs.expression().right())->getId()
    };

    std::vector<void*> args = QDP_get_global_cache().get_kernel_args( ids , false );

    // std::cout << args[0] << "\n";
    // std::cout << args[1] << "\n";
    // std::cout << args[2] << "\n";

    evaluate_special_quarkContract24( Layout::sitesOnNode() , args );
  }

  

  
} //namespace
