#ifndef QDP_SPECIALIZATIONS_H
#define QDP_SPECIALIZATIONS_H


namespace QDP {

  
inline int 
toInt(const Integer& s) 
{
  return toInt(s.elem());
}


inline float
toFloat(const Real32& s) 
{
  return toFloat(s.elem());
}


inline double
toDouble(const Real64& s) 
{
  return toDouble(s.elem());
}


inline bool
toBool(const Boolean& s) 
{
  return toBool(s.elem());
}

#ifndef QDP_NO_LIBXML2
// XML readers
template<>
void read(XMLReader& xml, const string& s, multi1d<Integer>& d);
template<>
void read(XMLReader& xml, const string& s, multi1d<Real32>& d);
template<>
void read(XMLReader& xml, const string& s, multi1d<Real64>& d);
template<>
void read(XMLReader& xml, const string& s, multi1d<Boolean>& d);

template<>
void read(XMLReader& xml, const string& s, std::vector<Integer>& d);
template<>
void read(XMLReader& xml, const string& s, std::vector<Real32>& d);
template<>
void read(XMLReader& xml, const string& s, std::vector<Real64>& d);
template<>
void read(XMLReader& xml, const string& s, std::vector<Boolean>& d);

template<>
void read(XMLReader& xml, const string& s, std::list<Integer>& d);
template<>
void read(XMLReader& xml, const string& s, std::list<Real32>& d);
template<>
void read(XMLReader& xml, const string& s, std::list<Real64>& d);
template<>
void read(XMLReader& xml, const string& s, std::list<Boolean>& d);

// XML writers
template<>
void write(XMLWriter& xml, const string& s, const multi1d<Integer>& d);
template<>
void write(XMLWriter& xml, const string& s, const multi1d<Real32>& d);
template<>
void write(XMLWriter& xml, const string& s, const multi1d<Real64>& d);
template<>
void write(XMLWriter& xml, const string& s, const multi1d<Boolean>& d);

template<>
void write(XMLWriter& xml, const string& s, const std::vector<Integer>& d);
template<>
void write(XMLWriter& xml, const string& s, const std::vector<Real32>& d);
template<>
void write(XMLWriter& xml, const string& s, const std::vector<Real64>& d);
template<>
void write(XMLWriter& xml, const string& s, const std::vector<Boolean>& d);

template<>
void write(XMLWriter& xml, const string& s, const std::list<Integer>& d);
template<>
void write(XMLWriter& xml, const string& s, const std::list<Real32>& d);
template<>
void write(XMLWriter& xml, const string& s, const std::list<Real64>& d);
template<>
void write(XMLWriter& xml, const string& s, const std::list<Boolean>& d);

#endif


  

template<>
struct SimpleScalar<float>
{
  typedef Real32   Type_t;
};


template<>
struct SimpleScalar<int>
{
  typedef Integer   Type_t;
};


template<>
struct SimpleScalar<double>
{
  typedef Real64   Type_t;
};


template<>
struct SimpleScalar<bool>
{
  typedef Boolean   Type_t;
};



template<>
struct InternalScalar<float>
{
  typedef float  Type_t;
};

template<>
struct InternalScalar<int>
{
  typedef int   Type_t;
};

template<>
struct InternalScalar<double>
{
  typedef double   Type_t;
};

template<>
struct InternalScalar<bool>
{
  typedef bool  Type_t;
};




template<>
struct PrimitiveScalar<float>
{
  typedef float  Type_t;
};

template<>
struct PrimitiveScalar<int>
{
  typedef int   Type_t;
};

template<>
struct PrimitiveScalar<double>
{
  typedef double   Type_t;
};

template<>
struct PrimitiveScalar<bool>
{
  typedef bool  Type_t;
};



template<>
struct LatticeScalar<float>
{
  typedef float  Type_t;
};

template<>
struct LatticeScalar<int>
{
  typedef int   Type_t;
};

template<>
struct LatticeScalar<double>
{
  typedef double   Type_t;
};

template<>
struct LatticeScalar<bool>
{
  typedef bool  Type_t;
};



template<>
struct RealScalar<int> {
  typedef REAL32  Type_t;
};

template<>
struct RealScalar<float> {
  typedef REAL32  Type_t;
};

template<>
struct RealScalar<double> {
  typedef REAL64  Type_t;
};



} // namespace QDP

#endif
