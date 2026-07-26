# The neural-network library. Implementations (.cpp) live in src/{helpers,classes,api};
# their declarations (.hpp) live in src/includes/... and are exposed via nn_headers.
add_library(nn STATIC
    helpers/linalg.cpp
    helpers/activations.cpp
    helpers/loss.cpp
    helpers/init.cpp
    classes/tensor.cpp
    classes/dense_layer.cpp
    classes/activation_layers.cpp
    classes/loss.cpp
    classes/network.cpp
    classes/json_line.cpp
    classes/jsonl_sink.cpp
    classes/optimizer.cpp
    api/dataset.cpp
    api/trainer.cpp)
target_link_libraries(nn
    PUBLIC nn_headers
    PRIVATE nn_warnings)

add_subdirectory(app)
