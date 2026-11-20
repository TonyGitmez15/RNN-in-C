#ifndef CONFIG_H
#define CONFIG_H
/**
 * @brief Archivo con valores globales de configuración para hiperparametros de funciones y enums de errores.
 * @file config.h
 */
//Geometria de la red neuronal.
#define CONFIG_INPUT_SIZE 1     //Dimension de entrada, numero de caracteristicas en cada vector x(t).
#define CONFIG_OUTPUT_SIZE 1    //Dimension de salida, numero de valores a predecir en y_hat(t).
#define CONFIG_HIDDEN_SIZE 4    //Dimension oculta, nuemero de neuronas (cantidad de memoria) de la capa recurrente.
//Parametros de la secuencia temporal.
#define CONFIG_TIME_STEPS 10
#define CONFIG_DELTA_T 0.1f
//Parametros de entrenamiento.
#define CONFIG_LEARNING_RATE 0.1f
#endif