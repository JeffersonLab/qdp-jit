// -*- C++ -*-

/*! @file
 * @brief Print profiling info
 *
 * Diagnostics to print profiling.
 */

#ifndef QDP_PROFILE_INCLUDE
#define QDP_PROFILE_INCLUDE

namespace QDP {

typedef unsigned long  QDPTime_t;

//! Get the wallclock time
/*!
  \return The wallclock time (since Epoch) in seconds.
*/
QDPTime_t getClockTime();
void initProfile(const std::string& file, const std::string& caller, int line);
void closeProfile();
void printProfile();
int setProfileLevel(int n);
int setProgramProfileLevel(int n);
int getProfileLevel();
int getProgramProfileLevel();
void pushProfileInfo(int level, const std::string& file, const std::string& caller, int line);
void popProfileInfo();


//--------------------------------------------------------------------------------------
// Selectively turn on profiling
//--------------------------------------------------------------------------------------

#if ! defined(QDP_USE_PROFILING)   
// No profiling
#define QDP_PUSH_PROFILE(a)
#define QDP_POP_PROFILE()

#else   // Profiling enabled

#define QDP_PUSH_PROFILE(a) pushProfileInfo(a, __FILE__, __func__, __LINE__)
#define QDP_POP_PROFILE()  popProfileInfo()


//-----------------------------------------------------------------------------
// Support of printing
//-----------------------------------------------------------------------------

struct QDPProfile_t;
void registerProfile(QDPProfile_t* qp);


//! Profiling object
/*!
 * Hold profiling state
 */
struct QDPProfile_t
{
  QDPTime_t     time;
  QDPTime_t     first_time;
  std::string   expr;
  int           count;
  int           num_regs;
  int           local_size;
  int           const_size;
  QDPProfile_t* next;
  bool          first;

  // Start time
  void stime(QDPTime_t t) {
    if (first)
      first_time -= t;
    else
      time -= t;
  }

  // End time
  void etime(QDPTime_t t) {
    if (first)
      first_time += t;
    else
      time += t;
    first=false;
  }

  // End time
  void etime(QDPTime_t t, JitFunction f) {
    if (first) {
      first_time += t;
#if 0
      num_regs = CudaAttributeNumRegs(f);
      local_size = CudaAttributeLocalSize(f);
      const_size = CudaAttributeConstSize(f);
#endif
      first=false;
    } else {
      time += t;
    }
  }

  void print();

  void init();
  QDPProfile_t() {init();}

  //! Profile rhs
  template<class T, class C, class Op, class RHS, class C1>
  QDPProfile_t(const QDPType<T,C>& dest, const Op& op, const QDPExpr<RHS,C1>& rhs)
    {
      init();

      if (getProfileLevel() > 0)
      {
	ostringstream os;
	printExprTree(os, dest, op, rhs);
	expr = os.str();
	registerProfile(this);
      }
    }

  //! Profile  opOuter(rhs)
  template<class T, class C, class Op, class OpOuter, class RHS, class C1>
  QDPProfile_t(const QDPType<T,C>& dest, const Op& op, const OpOuter& opOuter, const QDPExpr<RHS,C1>& rhs)
    {
      init();

      if (getProfileLevel() > 0)
      {
	typedef UnaryNode<OpOuter, typename CreateLeaf<QDPExpr<RHS,C1> >::Leaf_t> Tree_t;
	typedef typename UnaryReturn<C1,OpOuter>::Type_t Container_t;

	ostringstream os;
	printExprTree(os, dest, op, 
		      MakeReturn<Tree_t,Container_t>::make(Tree_t(
			CreateLeaf<QDPExpr<RHS,C1> >::make(rhs))));
	expr = os.str();
	registerProfile(this);
      }
    }

  //! Profile  opOuter(rhs)
  template<class T, class C, class Op, class OpOuter, class T1, class C1>
  QDPProfile_t(const QDPType<T,C>& dest, const Op& op, const OpOuter& opOuter, const QDPType<T1,C1>& rhs)
    {
      init();

      if (getProfileLevel() > 0)
      {
	typedef UnaryNode<OpOuter, typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
	typedef typename UnaryReturn<C1,OpOuter>::Type_t Container_t;

	ostringstream os;
	printExprTree(os, dest, op, 
		      MakeReturn<Tree_t,Container_t>::make(Tree_t(
			CreateLeaf<QDPType<T1,C1> >::make(rhs))));
	expr = os.str();
	registerProfile(this);
      }
    }
};


struct QDPProfileInfo_t
{
  int           level;
  std::string   file;
  std::string   caller;
  int           line;

  QDPProfileInfo_t() {line=level=0;}

  QDPProfileInfo_t(const QDPProfileInfo_t& a) :
    level(a.level), file(a.file), caller(a.caller), line(a.line) {}

  QDPProfileInfo_t(int _level, const std::string& _file, const std::string& _caller, int _line) :
    level(_level), file(_file), caller(_caller), line(_line) {}
};


struct QDPProfileHead_t
{
  QDPProfileInfo_t  info;
  QDPProfile_t*     start;
  QDPProfile_t*     end;

  QDPProfileHead_t() {start=0; end=0;}
  QDPProfileHead_t(const QDPProfileInfo_t& a) : info(a), start(0), end(0) {}
  QDPProfileHead_t(const QDPProfileHead_t& a) : info(a.info), start(a.start), end(a.end) {}
};




#endif  // ! defined(QDP_USE_PROFILING)

} // namespace QDP

#endif  // QDP_PROFILE_INCLUDE
