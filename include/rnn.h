#ifndef RNN_H
#define RNN_H
#include "linalg.h"
#include "sequence.h"

/**
 * @brief Se definen los componentes de vectores, matrices e hiperparametros para diseñar
 * la red neuronal recurrente.
 * @file rnn.h
 */
//Se declaran matrices y vectores como miembros de la estructura, no se inicializan aparte, por ello no
//declaran como punteros, evitando declararlos en otra parte de la memoria. Asi se reservara tambien el
//espacio en memoria para las variables de control row, col y data de cada matriz/vector.
//Parametros que se modificaran durante el entrenamiento y al final quedaran estaticos para inferencias.
typedef struct rnn_parameters{
    Matrix W_hy;    //Matriz pesos de salida.
    vector b_h;    //Vector de sego oculto.
    vector b_y;    //Vector de sesgo de salida.
    Matrix W_xh;   //Matriz de pesos de entrada.
    Matrix W_hh;   //Matriz de pesos recurrente.
}RNN_P;
//Estados que seran usados en entrenamientos e inferiencias, interactuan en ambos casos.
typedef struct rnn_states{
    vector x_t;    //Vector de entrada actual.
    vector h_t;    //Vector de estado oculto actual.
    vector h_t_1;  //Vector de estado oculto previo.
    vector y_t;    //Vector de salida predicha.
}RNN_S;
//Vectores para guardar resultados de multiplicaciones entre vector-matriz o vector-vector.
typedef struct rnn_aux{
    vector aux1;    //Vector auxiliar para multiplicaciones de vectores.
    vector aux2;    //Vector auxiliar para multiplicaciones de vectores.
    vector aux3;    //Vector auxiliar para suma de vectores.
    vector aux_out;
}RNN_A;
//Unificación de estados y parametros para su acceso en entrenamiento o inferencia.
typedef struct rnn{
    RNN_P parameters;
    RNN_S states;
    RNN_A auxs;
}RNN;
//--------------------------------------------------------------------------------------
//Firmas de las funciones de rnn.c
//--------------------------------------------------------------------------------------
bool rnn_init(RNN *net, size_t input_s, size_t hidden_s, size_t output_s);
void rnn_free(RNN *net);
void rnn_forward(RNN *net);
bool rnn_forward_sequence(Sequence_t *X, Sequence_t *H, RNN *net);
bool rnn_h_t(const Matrix *W_xh, const Matrix *W_hh, const vector *h_t1,
             const vector *x_t, const vector *b_h, vector *out_h_t,
             vector *aux1, vector *aux2, vector *aux3);
bool rnn_y_t(const Matrix *W_hy, const vector *h_t, const vector *b_y, vector *y_t,
             vector *aux1);
void activationF_Elliott(Matrix *m);
#endif