#include <iostream>

// Phase 0 smoke target: proves configure -> build -> run works with C++20.
// Replaced by real entrypoints from Phase 1 onward.
int main() {
    std::cout << "simple-nn-visualisation smoke test\n";
    std::cout << "__cplusplus = " << __cplusplus << '\n';

    static_assert(__cplusplus >= 202002L, "C++20 is required");
    return 0;
}
