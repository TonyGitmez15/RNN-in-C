#ifndef SEQUENCE_H
#define SEQUENCE_H
#include "linalg.h"
/**
 * @brief Apartado para definir firmas que se encarguen de manejar las secuencias de tiempo de entrada X y estados ocultos H.
 */
//Declarar alias para mejor lectura del código y diferenciar al trabajar con matrices o vectores, pues
//su definición parte del mismo struct Matrix.
typedef Matrix vector;
//Estructura para manipular secuencias de manera general controlando su tipo de dato, la cantidad de instantes de 
//tiempo que maneja y la longitud de cada uno de los datos en cada instante de tiempo t.
typedef struct sequence_t{
    vector data_seq;       //Vector que permite guardar arreglos de T filas por D columnas en arreglos de 1D.
    size_t t_max;       //Define la cantidad de T instantes de tiempo.
    size_t dim_data;    //Dimension D de cada dato para cada instante t.
}Sequence_t;
bool sequence_init(Sequence_t *seq_t, size_t t, size_t dim);
void sequence_free(Sequence_t *seq_t);
bool sequence_NULL(Sequence_t *X, Sequence_t *H);
#endif