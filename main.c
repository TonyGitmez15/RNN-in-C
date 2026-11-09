/**
 * @brief Main de proyecto RNN, se entrenara con senos y cosenos discretos para predecir seceuncias periodicas.
 * @file main.c
 */
#include <stdio.h>
#include <math.h>

#include "rnn.h"
#include "sequence.h"
#include "logs.h"
#include "config.h"

int main(void){
    LOG_INFO("Inicialización de Forwars Pass en secuencia...");
    //1. Inicializar la red, definir dimensiones de matrices y vectores.
    RNN net;
    if(!rnn_init(&net, CONFIG_INPUT_SIZE, CONFIG_HIDDEN_SIZE, CONFIG_OUTPUT_SIZE)){
        LOG_ERROR("Fallo al inicializar la estructura de %p", (void*)&net);
        return -1;
    }
    //2. Inicializar secuencias X y H, se reserva memoria.
    Sequence_t X, H;
    if(!sequence_init(&X, CONFIG_TIME_STEPS, CONFIG_INPUT_SIZE)){
        LOG_ERROR("Fallo al inicializar la secuencia de entrada %p", (void*)&X);
        rnn_free(&net);
        return -1;
    }
    if(!sequence_init(&H, CONFIG_TIME_STEPS, CONFIG_HIDDEN_SIZE)){
        LOG_ERROR("Fallo al inicializar la secuencia de estados ocultos %p", (void*)&H);
        rnn_free(&net);
        sequence_free(&X);
        return -1;
    }
    //3. Muestreo de señales senoidales para llenar la secuencia X. x[t] = sin(t * CONFIG_DELTA_T).
    for(size_t t = 0; t < CONFIG_TIME_STEPS; t++){
        X.data_seq.data[t] = sinf((float)t * CONFIG_DELTA_T);
    } 
    //4. Ejecutar Forward Pass, procesamiento de la secuencia en un solo bloque.
    if(!rnn_forward_sequence(&X, &H, &net)){
        LOG_ERROR("Error al pasar secuencia de entrada.");
    }
    else{
        LOG_INFO("Secuencia procesada con excito.");
        printf("___________________ Resultados del FORWARD PASS ___________________");
    }
    printf(" t | Input x[t] | Hidden h[t][0] | Output y[t][0]\n");
    printf("---------------------------------------------------");
    for(size_t t = 0; t < CONFIG_TIME_STEPS; t++){
        float x_val = X.data_seq.data[t * CONFIG_INPUT_SIZE];
        float h_val = H.data_seq.data[t * CONFIG_HIDDEN_SIZE]; // Primera neurona oculta
        float y_val = net.states.y_t.data[0]; // Salida actual
        printf("%2zu | %10.4f | %20.4f | %14.4f\\n", t, x_val, h_val, y_val); }
    printf("------------------------------------------------------\n\n");
    //5. Revisión de resultados y liberación de memoria.
    sequence_free(&X);
    sequence_free(&H);
    rnn_free(&net);

    LOG_INFO("Memoria liberada. Prueba finalizada con éxito.");
    return 0;
}