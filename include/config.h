#ifndef CONFIG_H
#define CONFIG_H
/**
 * @brief Archivo con valores globales de configuración para hiperparametros de funciones y enums de errores.
 * @file config.h
 */
//Geometria de la red neuronal.
#define CONFIG_INPUT_SIZE 1
#define CONFIG_OUTPUT_SIZE 1
#define CONFIG_HIDDEN_SIZE 4
//Parametros de la secuencia temporal.
#define CONFIG_TIME_STEPS 10
#define CONFIG_DELTA_T 0.1f
//Parametros de entrenamiento.
#define CONFIG_LEARNING_RATE 0.1f
#endif