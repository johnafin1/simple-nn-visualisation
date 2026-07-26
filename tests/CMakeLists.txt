include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

add_executable(nn_tests
    test_main.cpp
    test_linalg.cpp
    test_activations.cpp
    test_loss.cpp
    test_init.cpp
    test_tensor.cpp
    test_dense_layer.cpp
    test_activation_layers.cpp
    test_core_loss.cpp
    test_network.cpp
    test_json.cpp
    test_optimizer.cpp
    test_dataset.cpp
    test_trainer.cpp)
target_link_libraries(nn_tests PRIVATE nn nn_warnings doctest::doctest)

add_test(NAME nn_tests COMMAND nn_tests)
