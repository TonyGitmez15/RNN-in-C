#include "rnn.h"
#include "init.h"
#include "logs.h"
#include "linalg.h"
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
    Matrix* list_states[] = {
        &(net->states.x_t),
        &(net->states.y_t),
        &(net->states.h_t),
        &(net->states.h_t_1)
    };
    Matrix* list_aux[] = {
        &(net->auxs.aux1),
        &(net->auxs.aux2),
        &(net->auxs.aux3),
        &(net->auxs.aux_out)
    };
    //Inicializacion de matriz W_hh con dimensiones [h_s, h_s].
    if(!matrix_init(list_matrix[0], hidden_s, hidden_s)){ 
        LOG_ERROR("Fallo al inicializar matriz de pesos ocultos W_hh.");
        return false;
    }
    matrix_fill_randomXavier(list_matrix[0], hidden_s, hidden_s);    //Asignar valores aleatorios de inicio a los pesos.
    //Inicializacion de matriz W_hy con dimensiones [h_s, o_s].
    if(!matrix_init(list_matrix[1], hidden_s, output_s)){ 
        matrix_free(list_matrix[0]);
        LOG_ERROR("Fallo al inicializar matriz de pesos de salida W_hy.");
        return false;
    }
    matrix_fill_randomXavier(list_matrix[1], hidden_s, output_s);
    //Inicializacion de matriz W_xh con dimensiones [i_s, h_s].
    if(!matrix_init(list_matrix[2], input_s, hidden_s)){ 
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        LOG_ERROR("Fallo al inicializar matriz de pesos de entrada W_xh.");
        return false;
    }
    matrix_fill_randomXavier(list_matrix[2], input_s, hidden_s);
    //Inicializacion de vector de sesgos b_h, dimensiones [1, h_s].
    if(!matrix_init(list_vector[0], 1, hidden_s)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        LOG_ERROR("Fallo al inicializar vector de sesgos oculto b_h.");
        return false;
    }
    matrix_fill_zeros(list_vector[0]);    //Asignar valores en ceros.
    //Inicializacion de vector de sesgos b_y, dimensiones [o_s, 1].
    if(!matrix_init(list_vector[1], output_s, 1)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        LOG_ERROR("Fallo al inicializar vector de sesgos de salida b_y.");
        return false;
    }
    matrix_fill_zeros(list_vector[1]);
    //Inicializacion de x_t.
    if(!window_init(list_states[0], input_s, 1)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        LOG_ERROR("Fallo al inicializar vector de caracteristicas de entrada por instante de tiempo x_t.");
        return false;
    }
    //Inicializacion de y_t.
    if(!matrix_init(list_states[1], output_s, 1)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        matrix_free(list_states[0]);
        LOG_ERROR("Fallo al inicializar vector de prediccion por instante de tiempo y_t.");
        return false;
    }
    //Inicializacion de h_t.
    if(!window_init(list_states[2], 1, hidden_s)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        matrix_free(list_states[1]);
        LOG_ERROR("Fallo al inicializar vector de nuevo estado oculto por instante de tiempo h_t.");
        return false;
    }
    //Inicializacion de h_t_1.
    if(!matrix_init(list_states[3], 1, hidden_s)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        matrix_free(list_states[1]);
        LOG_ERROR("Fallo al inicializar vector de estado oculto previo por instante de tiempo h_t_1.");
        return false;
    }
    //Inicializacion aux1,guarda x_t*W_xh, dimensiones [i_s, h_s], x_t{1,1}*W_xh{1,4}=aux1{1,4}.
    if(!matrix_init(list_aux[0], input_s, hidden_s)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        matrix_free(list_states[1]);
        matrix_free(list_states[3]);
        LOG_ERROR("Fallo al inicializar matriz aux1.");
        return false;
    }
    //Inicialización aux2, guarda W_hh*h_t1, dimensiones [i_s, h_s].
    if(!matrix_init(list_aux[1], input_s, hidden_s)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        matrix_free(list_states[1]);
        matrix_free(list_states[3]);
        matrix_free(list_aux[0]);
        LOG_ERROR("Fallo al inicializar matriz aux2.");
    }
    //Inicializacion aux3, guarda x_t*W_xh + W_hh*h_t1, dimensiones [i_s, h_s].
    if(!matrix_init(list_aux[2], input_s, hidden_s)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        matrix_free(list_states[1]);
        matrix_free(list_states[3]);
        matrix_free(list_aux[0]);
        matrix_free(list_aux[1]);
        LOG_ERROR("Fallo al inicializar matriz aux3.");
    }
    //Inicializacion aux_out, guarda y_t = h_t * W_hy, dimensiones [h_s, o_s].
    if(!matrix_init(list_aux[3], input_s, output_s)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        matrix_free(list_states[1]);
        matrix_free(list_states[3]);
        matrix_free(list_aux[0]);
        matrix_free(list_aux[1]);
        matrix_free(list_aux[2]);
        LOG_ERROR("Fallo al inicializar matriz aux_out.");
    }
    //Inicializacion a, guarda h_t * W_hy, dimensiones [h_s, i_s]
    /*if(!matrix_init(list_aux[4],hidden_s, output_s)){
        matrix_free(list_matrix[0]);
        matrix_free(list_matrix[1]);
        matrix_free(list_matrix[2]);
        matrix_free(list_vector[0]);
        matrix_free(list_vector[1]);
        matrix_free(list_states[1]);
        matrix_free(list_states[3]);
        matrix_free(list_aux[0]);
        matrix_free(list_aux[1]);
        matrix_free(list_aux[2]);
        matrix_free(list_aux[3]);
        LOG_ERROR("Fallo al inicializar a.");
    }*/
    LOG_INFO("Procesamiento exitoso de %zu inputs, %zu hidden, %zu outputs en rnn.", input_s, hidden_s, output_s);
    LOG_INFO("Inicializacion de matriz W_xh con dimensiones [%zu, %zu]", input_s, hidden_s);
    LOG_INFO("Inicializacion de matriz W_hh con dimensiones [%zu, %zu]", hidden_s, hidden_s);
    LOG_INFO("Inicializacion de matriz W_hy con dimensiones [%zu, %zu]", hidden_s, output_s);
    LOG_INFO("Inicializacion de vector aux1 con dimensiones [%zu, %zu]", input_s, hidden_s);
    LOG_INFO("Inicializacion de vector aux2 con dimensiones [%zu, %zu]", input_s, hidden_s);
    LOG_INFO("Inicializacion de vector aux3 con dimensiones [%zu, %zu]", input_s, hidden_s);
    LOG_INFO("Inicializacion de vector aux_out con dimensiones [%zu, %zu]", input_s, hidden_s);
    LOG_INFO("Inicializacion de vector b_h con dimensiones [%zu, 1]", hidden_s);
    LOG_INFO("Inicializacion de vector b_y con dimensiones [%zu, 1]", output_s);
    LOG_INFO("Inicializacion de vector x_t con dimensiones [%zu, 1]", input_s);
    LOG_INFO("Inicializacion de vector y_t con dimensiones [%zu, 1]", output_s);
    LOG_INFO("Inicializacion de vector h_t con dimensiones [%zu, 1]", hidden_s);
    LOG_INFO("Inicializacion de vector h_t_1 con dimensiones [%zu, 1]", hidden_s);
    return true;
}
//----------------------------------------------------------------------------------------------------------------
//Función para liberar espacio en memoria de la rnn.
bool rnn_free(RNN *net){
    //Lista de punteros a cada matriz y vector para ser recorrida en ciclo for.
    Matrix* list_matrix[] = {
        &(net->parameters.b_h),
        &(net->parameters.b_y),
        &(net->parameters.W_hh),
        &(net->parameters.W_hy),
        &(net->parameters.W_xh),
        //&(net->states.h_t),
        &(net->states.h_t_1),
        //&(net->states.x_t),
        &(net->states.y_t),
        &(net->auxs.aux1),
        &(net->auxs.aux2),
        &(net->auxs.aux3),
        &(net->auxs.aux_out)
    };
    //Tamañano de la lista de arriba.
    size_t num_elementos = sizeof(list_matrix)/sizeof(list_matrix[0]);
    for(size_t i = 0; i < num_elementos; i++){
        matrix_free(list_matrix[i]);
    }
    return true;
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
    if(!matrix_mult(x_t, W_xh, aux1)){
        LOG_ERROR("Error al multiplicar vector x_t con matriz W_xh. Dimensiones de x_t: [%zu, %zu], dimensiones de W_xh: [%zu, %zu]",
            x_t->row, x_t->col, W_xh->row, W_xh->col);
        return false;
    }
    if(!matrix_mult(h_t1, W_hh, aux2)){
        LOG_ERROR("Error al multiplicar matriz W_hh con vector h_t1. Dimensiones de W_hh: [%zu, %zu], dimensiones de h_t1: [%zu, %zu]",
            W_hh->row, W_hh->col, h_t1->row, h_t1->col);
        return false;
    }
    if(!matrix_sum(aux1,aux2, aux3)){
        LOG_ERROR("Error al sumar multiplicacion W_hh*h_t1 + x_t*W_xh. Dimensiones de W_hh*h_t1: [%zu, %zu], dimensiones de x_t*W_xh: [%zu, %zu]",
            aux1->row, aux1->col, aux2->row, aux2->col);
        return false;
    }
    if(!matrix_sum(aux3, b_h, out_h_t)){
        LOG_ERROR("Error al sumar aux3 + b_h. Dimensiones de aux3: [%zu, %zu], dimensiones de b_h: [%zu, %zu]",
            aux3->row, aux3->col, b_h->row, b_h->col);
        return false;
    }
    activationF_Elliott(out_h_t);
    return true;
}
//Funcion que calulca el vector de salida y_t = W_hy*h_t + b_y
bool rnn_y_t(const Matrix *W_hy, const vector *h_t, const vector *b_y, vector *y_t,
             vector *a){
    if(!matrix_mult(h_t, W_hy, a)){
        LOG_ERROR("Fallo en multiplicación a = b_y*W_hy. Dimensiones de b_y: [%zu, %zu], dimensiones de W_hy: [%zu, %zu]",
            h_t->row, h_t->col, W_hy->row, W_hy->col);
        return false;
    }
    if(!matrix_sum(a, b_y, y_t)){
        LOG_ERROR("Fallo en suma y_t = a + b_y. Dimensiones de a: [%zu, %zu], dimensiones de b_y: [%zu, %zu]",
            a->row, a->col, b_y->row, b_y->col);
        return false;
    }
    return true;
}
//--------------------------------------------------------------------------
//Funcion para obtener los estados de salida h_t y y_t, tambien actualiza el valor de h_{t-1} con h_{t}.
bool rnn_forward(RNN *net){
    if(!rnn_h_t(&(net->parameters.W_xh), &(net->parameters.W_hh),
                &(net->states.h_t_1), &(net->states.x_t),
                &(net->parameters.b_h), &(net->states.h_t),
                &(net->auxs.aux1), &(net->auxs.aux2), &(net->auxs.aux3))){
        LOG_ERROR("Error al calcular h_t.");
        return false;
    }
    if(!rnn_y_t(&(net->parameters.W_hy), &(net->states.h_t), &(net->parameters.b_y),
                &(net->states.y_t), &(net->auxs.aux_out))){
        LOG_ERROR("Error al calcular y_t.");
        return false;
    }
    if(!matrix_copy(&(net->states.h_t_1), &(net->states.h_t))){
        LOG_ERROR("Fallo en el paso de h_t a h_t1.");
        return false;
    }
    LOG_INFO("Actualizacion de estado oculto para t-1 excitoso y calculo de y_t.");
    return true;
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
    //Regresamos el puntero de h_t y x_t a su valor previo a las asignaciones de X y H, este valor
    //se asigno en window_init(), desde rnn.c dentro de la funcion rnn_init().
    net->states.h_t.data = NULL;
    net->states.x_t.data = NULL;
    LOG_INFO("Secuencia de %zu pasos de timepo t procesados correctamente.", X->t_max);
    return true;
}
