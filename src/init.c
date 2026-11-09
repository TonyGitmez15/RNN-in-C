#include "init.h"
//-----------------------------------------------------------------------------------------------------------------------------
//Funcion para generar de manera aleatoria un numero tipo float.
float gen_float(size_t inputs, size_t outputs){
    float range = init_XavierGlorot(inputs, outputs);
    //Obtener un decimal entre 0.0 y 1.0.
    float norm = (float)rand() / (float)RAND_MAX;
    //Escalar valores entre -1.0 y 1.0.
    return -range + (2 * norm * range); //Equivale a: min + norm *(max - min)
}
//-----------------------------------------------------------------------------------------------------------------------------
//Función para poner en ceros un arreglo tipo float.
void gen_zero(float *array, size_t num_elementos){
    memset(array, 0, num_elementos * sizeof(float));
}
//-----------------------------------------------------------------------------------------------------------------------------
//Funcion para determinar de manera dinamica el rango aleatorio en base al número de
//entradas y salidas de la red.
float init_XavierGlorot(size_t no_inputs, size_t n_outputs){
    return sqrtf(6.0f) / sqrtf((float)no_inputs + (float)n_outputs);
}
//-----------------------------------------------------------------------------------------------------------------------------
//Funcion que toma una matriz M y la rellena con ceros.
void matrix_fill_zeros(Matrix *M){
    gen_zero(M->data, M->col * M->row);
}
//-----------------------------------------------------------------------------------------------------------------------------
//Funcion para rellenar datos de manera aleatoria una matriz.
void matrix_fill_randomXavier(Matrix *M, size_t inputs, size_t outputs){
    //Corroborar structs inizializados.
    if(M == NULL){
        printf("%s","Error al inicializar struct de matriz M. Error: init.c en función matrix_fill_randomXavier" );
        return;
    }
    if(M->data == NULL){
        printf("%s","Error al inicializar datos de la matriz M. Error init.c en función matrix_fill_randomXavier");
        return;
    } 
    for(size_t i = 0; i < M->row; i++){
        for(size_t j = 0; j < M->col; j++){
            size_t index = (i * M->col) + j;
            M->data[index] = gen_float(inputs, outputs);
        }
    }
}