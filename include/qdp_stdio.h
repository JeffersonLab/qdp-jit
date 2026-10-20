#ifndef QDP_STDIO_H
#define QDP_STDIO_H

#include <istream>
#include <ostream>
#include <string>

namespace QDP {


class StandardInputStream
{
public:
  StandardInputStream();
  ~StandardInputStream();
  void init(std::istream *is);

  bool fail();

  StandardInputStream& operator>>(std::string& input);
  StandardInputStream& operator>>(char& input);
  StandardInputStream& operator>>(int& input);
  StandardInputStream& operator>>(unsigned int& input);
  StandardInputStream& operator>>(short int& input);
  StandardInputStream& operator>>(unsigned short int& input);
  StandardInputStream& operator>>(long int& input);
  StandardInputStream& operator>>(unsigned long int& input);
  StandardInputStream& operator>>(long long int& input);
  StandardInputStream& operator>>(float& input);
  StandardInputStream& operator>>(double& input);
  StandardInputStream& operator>>(long double& input);
  StandardInputStream& operator>>(bool& input);


private:
  //! Hide copy constructor and =
  StandardInputStream(const StandardInputStream&) {}
  void operator=(const StandardInputStream&) {}

protected:
  template<typename T>
  StandardInputStream& readPrimitive(T& output);

  std::istream& getIstream() {return *is;}

  bool is_open() {return open;}

private:
  std::istream* is;
  bool open;
};


template<class T>
inline
StandardInputStream& operator>>(StandardInputStream& s, multi1d<T>& d)
{
  for(int i=0; i < d.size(); ++i)
    s >> d[i];

  return s;
}



class StandardOutputStream
{
public:
  StandardOutputStream();
  ~StandardOutputStream();
  void init(std::ostream *os);
  void flush();
  bool fail();

  StandardOutputStream& operator<<(std::ostream& (*op)(std::ostream&));

  StandardOutputStream& operator<<(const std::string& output);
  StandardOutputStream& operator<<(const char* output);
  StandardOutputStream& operator<<(char output);
  StandardOutputStream& operator<<(int output);
  StandardOutputStream& operator<<(unsigned int output);
  StandardOutputStream& operator<<(short int output);
  StandardOutputStream& operator<<(unsigned short int output);
  StandardOutputStream& operator<<(long int output);
  StandardOutputStream& operator<<(unsigned long int output);
  StandardOutputStream& operator<<(long long int output);
  StandardOutputStream& operator<<(float output);
  StandardOutputStream& operator<<(double output);
  StandardOutputStream& operator<<(long double output);
  StandardOutputStream& operator<<(bool output);


private:
  StandardOutputStream(const StandardOutputStream&) {}
  void operator=(const StandardOutputStream&) {}

protected:
  template<typename T>
  StandardOutputStream& writePrimitive(T output);

  std::ostream& getOstream() {return *os;}

  bool is_open() {return open;}

private:
  std::ostream* os;
  bool open;
};


  StandardOutputStream& operator<<(StandardOutputStream& s, const Shape& v);


namespace QDPIO
{
  extern StandardInputStream  cin;
  extern StandardOutputStream cout;
  extern StandardOutputStream cerr;
}



} // namespace QDP

#endif
