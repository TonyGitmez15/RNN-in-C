@echo off
:: Configurar consola UTF-8 para visualizar caracteres de logs correctamente.
chcp 65001 > nul
echo ========================================================================
echo                        COMPILACION Y PRUEBA DE RNN
echo ========================================================================
:: 1. Generar carpeta de CMake para compilación si no existe.
if not exist build (
    echo [*] Generando archivos de construccion de CMake...
    cmake -B build
    if %ERRORLEVEL% neq 0 (
        echo [!] Error en la configuracion del CMake.
        pause
        exit /b %ERRORLEVEL%
    )
)

:: 2. Compilar el proyecto en modo Debug.
echo [*] Compilando el ejecutable rnn_app...
cmake --build build --config Debug

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] ERROR: Fallo en la compilacion, revisar mensajes anteriores.
    echo.
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo ========================================================================
echo                      EJECUTANDO PRUEBA (rnn_app.exe)
echo ========================================================================
echo.

:: 3. Ejecutar el binario generado.
.\build\Debug\rnn_app.exe

echo.
echo ========================================================================
echo                          EJECUCION FINALIZADA
echo ========================================================================
echo.