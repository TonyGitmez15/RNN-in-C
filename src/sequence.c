#include "sequence.h"
#include "rnn.h"
#include "logs.h"
bool sequence_init(Sequence_t *seq_t, size_t t, size_t dim){
    seq_t->t_max = t;
    seq_t->dim_data = dim;
    //Reservar espacio de memoria para arreglo interno de seq_t.
    return matrix_init(&(seq_t->data_seq),t,dim);
}
//Funcion para libererar espacio de memoria de secuencias.
bool sequence_free(Sequence_t *seq_t)
{
    seq_t->dim_data = 0;
    seq_t->t_max = 0;
    free(&(seq_t->data_seq));
    if(&(seq_t->data_seq) == NULL){
        LOG_INFO("Sequencuia %p liberada.", seq_t);
        return true;
    }
    LOG_ERROR("Error al liberar secuencia %p.", seq_t);
    return false;
}
//Funcion para inicializar secuencia de entrada X y secuencia de estados ocultos H.
bool rnn_forward_sequence(Sequence_t *X, Sequence_t *H, RNN *net){
    //Validar punteros nulos.
    if(X == NULL){
        LOG_ERROR("Error con secuencia %p = NULL.",X);
        return false;
    }
    if( H == NULL){
        LOG_ERROR("Error con secuencia %p = NULL.",H);
        return false;
    }
    if(net == NULL){
        LOG_ERROR("Error con red %p = NULL.",net);
        return false;
    }
    //Validar dimensiones de las secuecnias X y H.
    if(X->t_max != H->t_max){
        LOG_ERROR("Tamaño de secuencias diferentes de %p y %p", X, H);
        return false;
    }
    //Validar dimensiones desde entrada en las secuencias de entrada y estado oculto con las matrices de pesos.
    if(X->dim_data != net->parameters.W_xh.col)
    {
        LOG_ERROR("Error en dimensiones de secuencia %p y numero de columnas en W_xh.", X);
        return false;
    }
    if(H->dim_data != net->parameters.W_hh.col){
        LOG_ERROR("Error en dimensiones de secuencia %p y numero de columnas en W_hh", H);
        return false;
    }
    //Resguardar puntero original de los datos de la secuencia de estados ocultos h_t.
    float *h_t_backup = net->states.h_t.data;
    //Inicializamos las dimensiones de entrada del vector de entrada x(t).
    matrix_init(&(net->states.x_t), X->dim_data, 1);

    //Bucle para recorrer todos los instantes de tiempo t a T-1 en X.
    for(size_t t = 0; t < X->t_max; t++){
        //Primero redireccionar la entrada x_t. Pasamos el dato de la secuencia X en la posición de 
        //tiempo actual t multiplicada por la longitud de cada dato en la secuencia X; pasandolo a 
        //data de x_t de la red. Esto actualiza la entrada x_t en la red para cada instante t.
        net->states.x_t.data = &(X->data_seq.data[t * X->dim_data]);
        //Hacemos lo mismo para el estado oculto h_t. Se pasa el dato de H en el instante de tiempo
        //t por la longitud de cada dato; pasandose a data de h_t de la red. Esto actualiza la entrada h_t en la red para cada instante t.
        net->states.h_t.data = &(H->data_seq.data[t * H->dim_data]);
        //Con x_t y h_t actualizados ejecutamos las operaciones de algebra lineal de la RNN.
        rnn_forward(net);
    }
    //Regresamos el puntero del estado oculto h_t a la dirección original.
    net->states.h_t.data = h_t_backup;
    LOG_INFO("Secuencia de %zu pasos de timepo t procesados correctamente.", X->t_max);
    return true;
}