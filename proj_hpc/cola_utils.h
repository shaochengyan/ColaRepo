#ifndef COLA_UTILS
#define COLA_UTILS
#include <chrono>
#include <iostream>
#include <string>

#define FOR_N_TIMES(N) for (int i = 0; i < (N); ++i)
#define FOR_IN_TIMES(i, N) for (int i = 0; i < (N); ++i)

namespace cola {

class Timer {
   private:
    std::chrono::_V2::system_clock::time_point now;

   public:
    Timer() {}
    void start();

    int64_t end_ms(std::string info, bool is_print);
};

}  // namespace cola

#endif COLA_UTILS