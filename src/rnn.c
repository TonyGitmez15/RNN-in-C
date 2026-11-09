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
    if(!matrix_init(list_matrix[1], hidden_s, output_s)){ //W_hy
        matrix_free(list_matrix[0]);
        LOG_ERROR("Fallo al inicializar matriz de pesos de salida W_hy.");
        return false;
    }
    matrix_fill_randomXavier(list_matrix[1], hidden_s, output_s);
    if(!matrix_init(list_matrix[2], input_s, hidden_s)){ //W_xh
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        LOG_ERROR("Fallo al inicializar matriz de pesos de entrada W_xh.");
        return false;
    }
    matrix_fill_randomXavier(list_matrix[2], input_s, hidden_s);

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
//--------------------------------------------------------------------------
//Funcion de activacion Elliot f(x) = x / (1 + |x|)
void activationF_Elliott(Matrix *m) {
    if (m == NULL || m->data == NULL){
        LOG_ERROR("Error en función de activación Elliot en matriz %p.", (void*)&m);
        return;
    }
    size_t total_elementos = m->row * m->col;
    for (size_t i = 0; i < total_elementos; i++) {
        // fabsf() pertenece a <math.h> para precisión float (32 bits)
        m->data[i] = m->data[i] / (1.0f + fabsf(m->data[i]));
    }
}
//--------------------------------------------------------------------------
//Funcion para obtener estado oculto h_t = act(W_xh*x_t + W_hh*h_t1 + b_h).
bool rnn_h_t(const Matrix *W_xh, const Matrix *W_hh, const vector *h_t1,
             const vector *x_t, const vector *b_h, vector *out_h_t,
             vector *aux1, vector *aux2, vector *aux3){
    if(!matrix_mult(W_xh, x_t, aux1)){
        LOG_ERROR("Error al multiplicar matriz %p con %p.", (void*)&W_xh, (void*)&x_t);
        LOG_INFO("Dimensiones de %p: [%zu, %zu], dimensiones de %p: [%zu, %zu]", (void*)&W_xh, W_xh->col, W_xh->row, (void*)&x_t, x_t->col, x_t->row);
        return false;
    }
    if(!matrix_mult(W_hh, h_t1, aux2)){
        LOG_ERROR("Error al multiplicar matriz %p con %p.", (void*)&W_hh, (void*)&h_t1);
        return false;
    }
    if(!matrix_sum(aux1,aux2, aux3)){
        LOG_ERROR("Error al sumar multiplicacion de matrices: %p * %p + %p * %p.", (void*)&W_xh, (void*)&x_t, (void*)&W_hh, (void*)&h_t1);
        return false;
    }
    if(!matrix_sum(aux3, b_h, out_h_t)){
        LOG_ERROR("Error al sumar %p + %p.", (void*)&aux3, (void*)&b_h);
        return false;
    }
    activationF_Elliott(out_h_t);
    return true;
}
//Funcion que calulca el vector de salida y_t = W_hy*h_t + b_y
bool rnn_y_t(const Matrix *W_hy, const vector *h_t, const vector *b_y, vector *y_t,
             vector *a){
    if(!matrix_mult(W_hy, h_t, a)){
        LOG_ERROR("Fallo en multiplicación matriz %p con vector %p.", (void*)&W_hy, (void*)&h_t);
        return false;
    }
    if(!matrix_sum(a, b_y, y_t)){
        LOG_ERROR("Fallo en suma de %p con vector %p.", (void*)&a, (void*)&b_y);
        return false;
    }
    return true;
}
//--------------------------------------------------------------------------
//Funcion para obtener los estados de salida h_t y y_t, tambien actualiza el valor de h_{t-1} con h_{t}.
void rnn_forward(RNN *net){
    if(!rnn_h_t(&(net->parameters.W_xh), &(net->parameters.W_hh),
                &(net->states.h_t_1), &(net->states.x_t),
                &(net->parameters.b_h), &(net->states.h_t),
                &(net->auxs.aux1), &(net->auxs.aux2), &(net->auxs.aux3))){
        LOG_ERROR("Error al calcular h_t.");
        return;
    }
    if(!rnn_y_t(&(net->parameters.W_hy), &(net->states.h_t), &(net->parameters.b_y),
                &(net->states.y_t), &(net->auxs.aux_out))){
        LOG_ERROR("Error al calcular y_t.");
        return;
    }
    matrix_copy(&(net->states.h_t_1), &(net->states.h_t));
    LOG_INFO("Actualizacion de estado oculto para t-1 excitoso y calculo de y_t.");
}
//Funcion para inicializar en memoria secuencias con t valores y cada valor t de dimension dim.
