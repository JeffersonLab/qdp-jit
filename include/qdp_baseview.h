#ifndef QDP_BASEVIEW_H
#define QDP_BASEVIEW_H

namespace QDP {


  //template<class T, template<class> class C>
  template<class T>
  class BaseView
  {
    typename WordType<T>::Type_t *base = nullptr;
    int words_T; // number of words in 'T'
    Shape shape_;
    
  public:

    ~BaseView()
    {
    }
    
    
    BaseView( const BaseView& rhs ): base(rhs.base), words_T(rhs.words_T), shape_(rhs.shape_)
    {
    }
    
    BaseView( Shape shape , typename WordType<T>::Type_t *base_ ): base(base_), shape_(shape)
    {
      words_T = WordSize<T> ::value(shape_);
    }
    
    Shape shape() const { return shape_; }

    
    T elem(int i) const
    {
      assert(base);
      T F( shape_ , base + i * words_T );
      return F;
    }
  };


  template<class T>
  class BaseView< Word<T> >
  {
    T* base = nullptr;
  public:

    Shape shape() const { return Shape(); }

    BaseView( const BaseView& rhs ): base(rhs.base)
    {
    }
    

    
    BaseView( Shape shape , typename WordType<T>::Type_t *base_ ): base(base_)
    {
    }

    
    Word<T> elem(int i) const
    {
      assert(base);
      Word<T> F( base + i );
      return F;
    }
  };

  
  
}

#endif
