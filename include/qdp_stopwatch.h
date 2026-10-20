#ifndef QDP_STOPWATCH_H
#define QDP_STOPWATCH_H


#include<sys/time.h>

namespace QDP {

class StopWatch 
{
public:
  StopWatch( bool device_sync = true );
  ~StopWatch();
  void reset();
  void start();
  void stop();
  double getTimeInMicroseconds();
  double getTimeInSeconds();

private:
  long sec;
  long usec;
  bool startedP;
  bool stoppedP;
  bool device_sync;
  
  struct timeval t_start;
  struct timeval t_end;
};

} // namespace QDP  

#endif
