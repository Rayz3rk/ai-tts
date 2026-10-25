@echo off
setlocal enabledelayedexpansion


:: Check if we're running in SysWOW64
if /i "%PROCESSOR_ARCHITECTURE%"=="x86" (
    if defined PROCESSOR_ARCHITEW6432 (
        :: We're running in 32-bit mode on a 64-bit system
        :: Re-launch using 64-bit cmd.exe
        %SystemRoot%\Sysnative\cmd.exe /c "%~dpnx0" %*
        exit /b
    )
)

:: At this point, we're running in 64-bit mode
:: Check if the script is being run directly or through another cmd instance
if /i "%~dp0"=="%SystemRoot%\SysWOW64\" (
    :: We're running from SysWOW64, re-launch using System32 cmd.exe
    %SystemRoot%\System32\cmd.exe /c "%~dpnx0" %*
    exit /b
)



:: NOTE (v4.0): admin elevation is no longer required - deleting installer_files
:: needs no special rights. Only the optional system-package removal (choco) does.

echo =========================================================================
echo.
echo   ABUS Uninstaller [Version 4.0 - uv]
echo   contact: abus.aikorea@gmail.com
echo.
echo =========================================================================
echo.

:: If we've reached here, we're running in the correct environment
:: Your actual batch file commands start here
echo Running in 64-bit mode from System32
echo Current directory: %CD%
echo Command line: %*


cd /D "%~dp0"


:: "uninstall.bat silent" skips the prompts (terminate=yes, remove system packages=no)
set SILENT=0
if /i "%1"=="silent" set SILENT=1


:: quit task
if "%SILENT%"=="0" (
    choice /C YN /N /T 10 /D Y /M "Terminate all running python.exe. Do you want to continue (Y/N)?"
    if errorlevel 2 (
        echo.
        echo Quit Uninstallation.
        pause
        exit 0
    )
)
taskkill /f /im python.exe /t
echo.


if "%SILENT%"=="0" (
    echo.
    choice /C NY /N /T 10 /D N /M "Would you like to remove system packages (not recommended, requires administrator) (N/Y)?"
    if errorlevel 2 (
        echo.
        net session >nul 2>&1
        if errorlevel 1 (
            echo Removing system packages requires administrator rights.
            echo Please run uninstall.bat again as administrator to remove them.
        ) else (
            echo Removing system packages ...
            choco uninstall -y git.install
            choco uninstall -y ffmpeg
        )
    )
)


:: remove folder
if exist "%~dp0\installer_files\" (
    echo Deleting installer_files folder ...
    echo Please wait a moment

    rmdir /s /q "%~dp0\installer_files"
) 
echo.
echo ABUS uninstall.bat finished.
echo Note: the application folder itself was not deleted.
echo To completely remove Voice-Pro, delete this entire folder.
if "%SILENT%"=="0" pause