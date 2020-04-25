#ifndef CRCHAM_EVALUATOR_HPP
#define CRCHAM_EVALUATOR_HPP

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace crcham {

class WeightsEvaluator {
private:
    const uint64_t d_polynomial; 
    const size_t d_polylen; 
    const size_t d_message; 
    const size_t d_errors; 
    const size_t d_evaluations; 
    size_t d_weight;
    std::chrono::milliseconds d_elapsed;

public:
    WeightsEvaluator(uint64_t polynomial, size_t message_bits, size_t error_bits);

    template<bool GPU>
    void run();

    size_t evaluations() const;
    size_t weight() const;
    std::chrono::milliseconds elapsed() const;
};

}

#endif
