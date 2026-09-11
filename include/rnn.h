#ifndef RNN_H
#define RNN_H
#include "linalg.h"

/**
 * @brief Se definen los componentes de vectores, matrices e hiperparametros para diseñar
 * la red neuronal recurrente.
 * @file rnn.h
 */
//Declarar alias para mejor lectura del código y diferenciar al trabajar con matrices o vectores, pues
//su definición parte del mismo struct Matrix.
typedef Matrix vector;
//Se declaran matrices y vectores como miembros de la estructura, no se inicializan aparte, por ello no
//declaran como punteros, evitando declararlos en otra parte de la memoria. Asi se reservara tambien el
//espacio en memoria para las variables de control row, col y data de cada matriz/vector.
typedef struct rnn_parameters{
    Matrix W_hy;    //Matriz pesos de salida.
    vector h_t_1;  //Vector de estado oculto previo.
    vector b_h;    //Vector de sego oculto.
    vector b_y;    //Vector de sesgo de salida.
    Matrix W_xh;   //Matriz de pesos de entrada.
    Matrix W_hh;   //Matriz de pesos recurrente.
}RNN_P;

typedef struct rnn_states{
    vector x_t;    //Vector de entrada actual.
    vector h_t;    //Vector de estado oculto actual.
    vector y_t;    //Vector de salida predicha.
}RNN_S;

#endif