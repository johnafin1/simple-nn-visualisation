# The neural-network library. Implementations (.cpp) live in src/{helpers,classes,api};
# their declarations (.hpp) live in src/includes/... and are exposed via nn_headers.
add_library(nn STATIC
    helpers/linalg.cpp)
target_link_libraries(nn
    PUBLIC nn_headers
    PRIVATE nn_warnings)

add_subdirectory(app)
