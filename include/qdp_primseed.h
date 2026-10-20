#ifndef QDP_PRIMSEED_H
#define QDP_PRIMSEED_H


namespace QDP {


template <class T> class PSeed: public BaseView<T>
{
public:
  typedef T Sub_t;
  enum { ThisSize = 4 };

  PSeed( Shape shape , typename WordType<T>::Type_t *base_ ): BaseView<T>(shape,base_) {}

  PSeed(const PSeed& a): BaseView<T>(a) {}

  PSeed operator=(const PSeed& rhs)
  {
    for (int i = 0 ; i < 4 ; ++i )
      elem(i) = rhs.elem(i);
    return *this;
  }

  T elem(int i) const { return BaseView<T>::elem(i); }
};



  template<class T>
  class WordSize< PSeed<T> >
  {
  public:
    static int value( const Shape& ret )
    {
      return 4 * WordSize<T>::value( ret );
    }
  };



  template<class T>
  struct GetLimit<PSeed<T>,0>
  {
    static int limit(const Shape& shape)
    {
      return 4;
    }
  };







template<class T> 
struct JITType<PSeed<T> >
{
  typedef PSeedJIT<typename JITType<T>::Type_t>  Type_t;
};



template<class T>
inline
istream& operator>>(istream& s, PSeed<T>& d)
{
  for(int i=0; i < 4; ++i)
    s >> d.elem(i);

  return s;
}


template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, PSeed<T>& d)
{
  for(int i=0; i < 4; ++i)
    s >> d.elem(i);

  return s;
}


template<class T>
inline
ostream& operator<<(ostream& s, const PSeed<T>& d)
{
  s << d.elem(0) << " " << d.elem(1) << " " << d.elem(2) << " " << d.elem(3) << "\n";
  return s;
}


template<class T>
inline
StandardOutputStream& operator<<(StandardOutputStream& s, const PSeed<T>& d)
{
  s << d.elem(0) << " " << d.elem(1) << " " << d.elem(2) << " " << d.elem(3) << "\n";
  return s;
}


template<class T>
inline
TextReader& operator>>(TextReader& txt, PSeed<T>& d)
{
  for(int i=0; i < 4; ++i)
    txt >> d.elem(i);

  return txt;
}


template<class T>
inline
TextWriter& operator<<(TextWriter& txt, const PSeed<T>& d)
{
  for(int i=0; i < 4; ++i)
    txt << d.elem(i) << "\n";

  return txt;
}

#ifndef QDP_NO_LIBXML2
//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, PSeed<T> d)
{
  xml.openTag("Seed");

  // Copy into another array first
  for(int i=0; i < 4; ++i)
  {
    xml.openTag("elem");
    xml << d.elem(i);
    xml.closeTag();
  }

  xml.closeTag();  // Seed
  return xml;
}


//! XML input
template<class T>
void read(XMLReader& xml, const string& path, PSeed<T> d)
{
  XMLReader arraytop(xml, path + "/Seed");

  std::ostringstream error_message;
  
  std::string elem_base_query = "elem";
	
  int array_size = 0;
  try {
    array_size = arraytop.count(elem_base_query);
  }
  catch( const std::string& e) { 
    error_message << "Exception occurred while counting " << elem_base_query 
		  << " during array read " << std::endl;
    throw error_message.str();
  }

  assert( 4 == array_size );

  std::vector<unsigned char> mem( jit_sizeof(d) / PSeed<T>::ThisSize );

  T tmp( d.shape() , (typename WordType<T>::Type_t*)mem.data() );
    
  for(int i=0; i < array_size; i++) 
    {
      std::ostringstream element_xpath;

      // Create the query for the element 
      element_xpath << elem_base_query << "[" << (i+1) << "]";

      // recursively try and read the element.
      try 
	{
	  read(arraytop, element_xpath.str(), tmp );
	  d.elem( i ) = tmp;
	} 
      catch (const std::string& e) 
	{
	  error_message << "Failed to match element " << i
			<< " of array with query " << element_xpath.str()
			<< std::endl
			<< "Query returned error: " << e;
	  throw error_message.str();
	}
    }
}

#endif


template<class T1>
struct WordType<PSeed<T1> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};


template<class T1>
struct ScalarType<PSeed<T1> >
{
  typedef PSeed< typename ScalarType<T1>::Type_t > Type_t;
};

  

template<class T1>
struct SinglePrecType<PSeed<T1> >
{
  typedef PSeed< typename SinglePrecType<T1>::Type_t > Type_t;
};

template<class T1>
struct DoublePrecType<PSeed<T1> >
{
  typedef PSeed< typename DoublePrecType<T1>::Type_t > Type_t;
};


template<class T>
struct InternalScalar<PSeed<T> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};


template<class T>
struct PrimitiveScalar<PSeed<T> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


template<class T>
struct LatticeScalar<PSeed<T> > {
  typedef PSeed<typename LatticeScalar<T>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSeed<T1>, PSeed<T2>, OpEQ> {
  typedef PScalar<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSeed<T1>, PSeed<T2>, OpNE> {
  typedef PScalar<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};



template<class T1, class T2>
struct BinaryReturn<PSeed<T1>, PSeed<T2>, OpMultiply> {
  typedef PSeed<typename BinaryReturn<T1, T2, OpMultiply>::Type_t>  Type_t;
};



template<class T1, class T2>
struct BinaryReturn<PSeed<T1>, PSeed<T2>, OpBitwiseOr> {
  typedef PSeed<typename BinaryReturn<T1, T2, OpBitwiseOr>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSeed<T1>, PScalar<T2>, OpBitwiseOr> {
  typedef PSeed<typename BinaryReturn<T1, T2, OpBitwiseOr>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<PSeed<T1>, PScalar<T2>, OpLeftShift> {
  typedef PSeed<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};



template<class T>
struct UnaryReturn<PSeed<T>, FnSeedToFloat> {
  typedef PScalar<typename UnaryReturn<T, FnSeedToFloat>::Type_t>  Type_t;
};



template<class T> 
inline void 
zero_rep(PSeed<T>& dest) 
{
  for(int i=0; i < 4; ++i)
    zero_rep(dest.elem(i));
}




} // namespace QDP

#endif
