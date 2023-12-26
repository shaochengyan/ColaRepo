#include <chrono>
#include <iostream>
#include <string>

#include "cola_utils.h"


void cola::Timer::start() { this->now = std::chrono::system_clock::now(); }
int64_t cola::Timer::end_ms(std::string info, bool is_print) {
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - this->now);
    std::chrono::milliseconds milliseconds_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    // output ms
    int64_t ms_count = milliseconds_duration.count();
    if (is_print)
        std::cout << info << ms_count << " (ms)" << std::endl;

    return ms_count;
}

