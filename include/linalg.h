#ifndef LINALG_H
#define LINALG_H
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
/**
 * @brief Definición de estructura para operar matrices y sus posteriores operaciones
 * a través de funciones que manejen el typedef struct.
 * @file linealg.h
 */
typedef struct matrix{
    size_t row;
    size_t col;
    float* data;
}Matrix;

bool matrix_init(Matrix *m, size_t rows, size_t cols);
void matrix_free(Matrix *m);
bool matrix_sum(const Matrix *a, const Matrix *b, Matrix *out);
bool matrix_mult(const Matrix *a, const Matrix *b, Matrix *out);
#endif