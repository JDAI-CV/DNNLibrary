//
// Created by daquexian on 7/24/18.
//

#ifndef PROJECT_OPERAND_HELPER_H
#define PROJECT_OPERAND_HELPER_H

#include <iostream>
#include <tuple>

/*
#define ADD_OPERANDS(...)   \
template <typename... Args> \
std::tuple<Args...> t(__VA_ARGS__)

template <typename Tuple, size_t N>
void addOperand(const Tuple &tuple) {
    addOperand<Tuple, N-1>(tuple);
}

template <typename Tuple>
void addOperand<Tuple, 0>(const Tuple &tuple) {
};

template <typename... Args>
class DefineOperands {
    std::tuple<Args...> t;

    explicit DefineOperands(Args&... args): t(args) {

    }
};
 */

template <typename... Ts>
void addOperand2(const std::tuple<Ts...> &tuple) {
    std::apply([] (const auto &item) {
        std::cout << item << std::endl;
    }, tuple);
}

template<typename... Ts>
void print_tuple (const std::tuple<Ts...> &tuple)
{
    std::apply ([] (const auto &... item)
                {
                    ((std::cout << item << '\n'), ...);
                },
                tuple);
}
#endif //PROJECT_OPERAND_HELPER_H
