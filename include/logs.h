#ifndef LOGS_H
#define LOGS_H
#include <stdio.h>

typedef enum{
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR
}LogLevel;

#define LOG_ERROR(fmt, ...)\
    fprintf(stderr, "[ERROR] [%s:%d -> %s()]: " fmt "\n", \
            __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)\
        fprintf(stdout, "[INFO] [%s:%d]: " fmt "\n", \
            __FILE__, __LINE__, ##__VA_ARGS__)
#endif