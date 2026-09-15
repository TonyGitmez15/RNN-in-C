#include "rnn.h"
#include "init.h"
#include "logs.h"
//Funcion de inicialización de red donde se reserva memoria para cada matriz y vector;
//además se inicializan con valores aleatrios o ceros.
bool rnn_init(RNN *net, size_t input_s, size_t hidden_s, size_t output_s){
    Matrix* list_matrix[] = {
        &(net->parameters.W_hh),
        &(net->parameters.W_hy),
        &(net->parameters.W_xh)
    };
    Matrix* list_vector[] = {
        &(net->parameters.b_h),
        &(net->parameters.b_y)
    };
    if(!matrix_init(list_matrix[0], hidden_s, hidden_s)){ //W_hh
        LOG_ERROR("Fallo al inicializar matriz de pesos ocultos W_hh.");
        return false;
    }
    matrix_fill_randomXavier(list_matrix[0], hidden_s, hidden_s);    //Asignar valores aleatorios de inicio a los pesos.
    if(!matrix_init(list_matrix[1], output_s, hidden_s)){ //W_hy
        matrix_free(list_matrix[0]);
        LOG_ERROR("Fallo al inicializar matriz de pesos de salida W_hy.");
        return false;
    }
    matrix_fill_randomXavier(list_matrix[1], output_s, hidden_s);
    if(!matrix_init(list_matrix[2], hidden_s, input_s)){ //W_xh
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        LOG_ERROR("Fallo al inicializar matriz de pesos de entrada W_xh.");
        return false;
    }
    matrix_fill_randomXavier(list_matrix[2], hidden_s, input_s);

    if(!matrix_init(list_vector[0], hidden_s, 1)){//b_h
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        LOG_ERROR("Fallo al inicializar vector de sesgos oculto b_h.");
        return false;
    }
    matrix_fill_zeros(list_vector[0]);    //Asignar valores en ceros.
    if(!matrix_init(list_vector[1], output_s, 1)){//b_y
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        LOG_ERROR("Fallo al inicializar vector de sesgos de salida b_y.");
        return false;
    }
    matrix_fill_zeros(list_vector[1]);
    LOG_INFO("Procesamiento exitoso de %zu inputs, %zu hidden, %zu outputs en rnn.", input_s, hidden_s, output_s);
    return true;
}
//-----------------------------------------------------------------------------------------------------------------
//Función para liberar espacio en memoria de la rnn.
void rnn_free(RNN *net){
    //Lista de punteros a cada matriz y vector para ser recorrida en ciclo for.
    Matrix* list_matrix[] = {
        &(net->parameters.b_h),
        &(net->parameters.b_y),
        &(net->parameters.W_hh),
        &(net->parameters.W_hy),
        &(net->parameters.W_xh),
        &(net->states.h_t),
        &(net->states.h_t_1),
        &(net->states.x_t),
        &(net->states.y_t)
    };
    //Tamañano de la lista de arriba.
    size_t num_elementos = sizeof(list_matrix)/sizeof(list_matrix[0]);
    for(size_t i = 0; i < num_elementos; i++){
        matrix_free(list_matrix[i]);
    }
}