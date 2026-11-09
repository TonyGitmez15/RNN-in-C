cmake_minimum_required(VERSION 3.10)

# Nombre del proyecto
project(RNN_C_PROJECT)

# Versión de C
set(CMAKE_C_STANDARD 99)

# 1. Configurar el tipo de construcción a Debug (Para ver errores exactos en VS Code)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 2. Banderas de compilación según el compilador (MSVC vs GCC/Clang)
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")
endif()

# 3. Archivos src para la compilación (Creación del TARGET)
add_executable(rnn_app
    main.c
    src/linalg.c
    src/init.c
    src/rnn.c
    src/sequence.c)

# 4. Enlazar la carpeta include AL TARGET
target_include_directories(rnn_app PRIVATE include)

# 5. Enlazar la librería matemática estándar ('m') AL TARGET para poder usar tanh, exp, etc.
if(NOT MSVC) 
    target_link_libraries(rnn_app m)
endif()