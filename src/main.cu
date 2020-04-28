#include <cassert>
#include <iostream>

#include <crcham/evaluator.hpp>
#include <crcham/math.hpp>

#include <cxxopts.hpp>

int main(int argc, char** argv)
{
    cxxopts::Options options("weights", "Calculate the hamming weights of CRC polynomials");
    options.add_options()("poly",
        "CRC in Koopman notation (implicit +1)", cxxopts::value<uint64_t>());
    options.add_options()("message",
        "Message length in bits", cxxopts::value<size_t>());
    options.add_options()("errors",
        "Number of bit errors in the message", cxxopts::value<size_t>());
    options.add_options()("c,cpu",
        "Run without using a GPU", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("h,help",
        "Print help message");
    options.parse_positional({ "poly", "message", "errors" });
    auto result = options.parse(argc, argv);

    if (result.count("help") > 0) {
        options.positional_help("[poly] [message] [errors]")
            .show_positional_help();
        std::cout << options.help();
        return EXIT_SUCCESS;
    }

    uint64_t polynomial;
    try {
        polynomial = result["poly"].as<uint64_t>();
    } catch (std::exception& ex) {
        std::cerr << "Unable to interpret polynomial: "
                  << ex.what() << std::endl;
        std::cerr << "Consider using https://users.ece.cmu.edu/~koopman/crc/crc32.html to select parameters." << std::endl;
        return EXIT_SUCCESS;
    }

    size_t message_bits;
    try {
        message_bits = result["message"].as<size_t>();
    } catch (std::exception& ex) {
        std::cerr << "Unable to interpret message length: "
                  << ex.what() << std::endl;
        std::cerr << "Consider using https://users.ece.cmu.edu/~koopman/crc/crc32.html to select parameters." << std::endl;
        return EXIT_SUCCESS;
    }

    size_t error_bits;
    try {
        error_bits = result["errors"].as<size_t>();
    } catch (std::exception& ex) {
        std::cerr << "Unable to interpret error count: "
                  << ex.what() << std::endl;
        std::cerr << "Consider using https://users.ece.cmu.edu/~koopman/crc/crc32.html to select parameters." << std::endl;
        return EXIT_SUCCESS;
    }

    crcham::WeightsEvaluator evaluator(polynomial, message_bits, error_bits);
    std::cout << "Evaluating " << evaluator.evaluations() << " "
              << error_bits << "-bit error combinations." << std::endl;

    if (result["cpu"].as<bool>()) {
        evaluator.run<false>();
    }
    else {
        evaluator.run<true>();
    }

    float elapsed_seconds = evaluator.elapsed().count() / 1000.f;
    std::cout << "Completed in " << elapsed_seconds
              << " seconds (" << (evaluator.evaluations() / elapsed_seconds)
              << "/s)." << std::endl;
    std::cout << "Hamming weight is " << evaluator.weight() << "." << std::endl;

    return EXIT_SUCCESS;
}
