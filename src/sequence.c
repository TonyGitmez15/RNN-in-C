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
void sequence_free(Sequence_t *seq_t)
{
    if(seq_t == NULL || seq_t->data_seq.data == NULL){
        LOG_INFO("Puntero %p de secuencia ya esta libre.", seq_t->data_seq.data);
        return;
    }
    LOG_INFO("Intentando liberar memoria en puntero: %p", (void*)seq_t->data_seq.data);
    free(seq_t->data_seq.data);
    seq_t->data_seq.data = NULL;
    seq_t->dim_data = 0;
    seq_t->t_max = 0;
}
//Funcion para liberar X y H.
bool sequence_NULL(Sequence_t *X, Sequence_t *H){
    Sequence_t* list_sequence[] = {
        &(X->data_seq),
        &(H->data_seq)
    };
    //Tamañano de la lista de arriba.
    size_t num_elementos = sizeof(list_sequence)/sizeof(list_sequence[0]);
    for(size_t i = 0; i < num_elementos; i++){
        sequence_free(list_sequence[i]);
    }
    return true;
}