#ifndef INIT_H
#define INIT_H
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "linalg.h"
#include "config.h"
/**
 * @brief Definición de funciones para inicializar vectores y matrices dentro de la arquitectura
 * según se necesite.
 * @file init.h
 */
void gen_zero(float *array, size_t num_elementos);
float gen_float(size_t inputs, size_t outputs);
float init_XavierGlorot(size_t no_input, size_t n_outputs);
void matrix_fill_zeros(Matrix *M);
void matrix_fill_randomXavier(Matrix *M, size_t inputs, size_t outputs);
void matrix_copy(Matrix *dest, const Matrix *src);
#endif