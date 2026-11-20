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
        LOG_INFO("Sequencia %p liberada.", seq_t);
        return true;
    }
    LOG_ERROR("Error al liberar secuencia %p.", seq_t);
    return false;
}
//Funcion para inicializar secuencia de entrada X y secuencia de estados ocultos H.
bool rnn_forward_sequence(Sequence_t *X, Sequence_t *H, RNN *net){
    //Validar punteros nulos.
    if(X == NULL){
        LOG_ERROR("Error con secuencia X = NULL.");
        return false;
    }
    if( H == NULL){
        LOG_ERROR("Error con secuencia H = NULL.");
        return false;
    }
    if(net == NULL){
        LOG_ERROR("Error con red net = NULL.");
        return false;
    }
    //Validar dimensiones de las secuecnias X y H.
    if(X->t_max != H->t_max){
        LOG_ERROR("Tamaño de secuencias diferentes de X y H");
        return false;
    }
    //Resguardar puntero original de los datos de la secuencia de estados ocultos h_t.
    float *h_t_backup = net->states.h_t.data;
    float *x_t_backup = net->states.x_t.data;
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
        if(!rnn_forward(net)){
            LOG_ERROR("Fallo en calculo de red.");
            return false;
        }
    }
    //Regresamos el puntero del estado oculto h_t a la dirección original.
    net->states.h_t.data = h_t_backup;
    net->states.x_t.data = x_t_backup;
    LOG_INFO("Secuencia de %zu pasos de timepo t procesados correctamente.", X->t_max);
    return true;
}