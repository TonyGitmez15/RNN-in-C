#ifndef CONFIG_H
#define CONFIG_H
/**
 * @brief Archivo con valores globales de configuración para hiperparametros de funciones y enums de errores.
 * @file config.h
 */

 typedef enum error{
    ERROR_FILL_XAVIER,
    ERROR_FILL_ZEROS
 }ERROR;

#define NUM_NEURONS_INPUT 10
#define NUM_NEURONS_OUTPUT 10
#define MAX_NORM 1.0f
#define MIN_NORM -1.0f

#endif