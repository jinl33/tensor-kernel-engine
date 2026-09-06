add_executable(kernel_bench src/main.cpp src/engine.cpp)
target_include_directories(kernel_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(kernel_bench PRIVATE Threads::Threads)
