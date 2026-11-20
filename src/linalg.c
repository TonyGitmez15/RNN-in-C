#include "linalg.h"
#include "logs.h"
#include <string.h>
//Reservar (Construir) espacio de memoria para el puntero de la estructura Matrix.
bool matrix_init(Matrix *m, size_t rows, size_t cols){
    m->data = malloc(rows * cols *sizeof(float));   //Reservar espacio del tamaño de la matriz por el tipo de dato.
    if(m->data == NULL) return false;   //Revisar si se pudo reservar la memoria.
    m->row = rows;
    m->col = cols;
    return true;
}
//Liberar manualmente el espacio de memoria para el puntero a la estructura Matrix.
void matrix_free(Matrix *m){
    if(m->data != NULL){
        free(m->data);
        m->data = NULL;
    }
    m->col = 0;
    m->row = 0;
}
//Funcion que revisa la suma correcta de 2 estructuras tipo Matrix con resultado en una tercera,
//se pasas el tamaño n de la matriz out, las 3 deben de ser del mismo tamaño para que retorne true.
bool matrix_sum(const Matrix *a, const Matrix *b, Matrix *out){
    //Manejando las matrices a y b como arreglos unidimensionales, corroboramos tengan misma longitud.
    if(a == NULL || b == NULL || out == NULL) return false;
    size_t n = out->col * out->row;
    if(a->col != b->col || a->row != b->row || a->col * a->row != n) return false;
    if(out->data == NULL || a->data == NULL || b->data == NULL) return false;
    //Sabiendo que los datos tipo float se representan con 4 bytes en la arquitectura actual.
    for(size_t i = 0; i < n; i++){
        out->data[i] = a->data[i] + b->data[i];
    }
    return true;
}
//Funcion que revisa la correcta multiplicacion de 2 matrices, tal que a y b satisfagan la condiion
// p*n y n*q para que sean multiplicables.
bool matrix_mult(const Matrix *a, const Matrix *b, Matrix *out){
    if(a == NULL || b == NULL || out == NULL){
        LOG_ERROR("Error al inicalizar matrices en multiplicacion, A = %p, B = %p y C = %p", a, b, out);
        return false; //Corroborar structs inizializados.
    } 
    if(a->col != b->row || a->row != out->row || b->col != out->col){
        LOG_ERROR("Error en las dimensiones de matrices A[%zu, %zu], B[%zu, %zu] y OUT[%zu, %zu]", 
            a->row, a->col, b->row, b->col, out->row, out->col);
        return false;  //Corroborar dimensiones de a y b para producto punto.
    }
    if(out->data == NULL || a->data == NULL || b->data == NULL) return false;   //Corroborar datos en strcuts.
    size_t index_out = 0;
    for(size_t i = 0; i < a->row; i++){
        for(size_t j = 0; j < b->col; j++){
            out->data[index_out] = 0.0f;    //inicializar indice out en cero antes de calcular producto punto.
            for(size_t k = 0; k < a->col; k++){
                size_t index_a = (i * a->col) + k;    //Indice para extraer el valor de la columna de a.
                size_t index_b = (k * b->col) + j;    //Indice para extraer el valor de la columna de b.
                out->data[index_out] += a->data[index_a] * b->data[index_b];
            }
            index_out++;    //Indice para colocar el valor obtenido del producto punto de a y b.
        }
    }
    return true;
}