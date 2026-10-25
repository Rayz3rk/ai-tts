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



:: run as admin (needed for choco installs and the LongPathsEnabled registry key)
net session >nul 2>&1
if errorlevel 1 (
    if "%1"=="am_admin" (
        echo This script requires administrator rights.
        echo Please right-click configure.bat and select "Run as administrator".
        pause
        exit /b 1
    )
    :: try self-elevation via PowerShell; on machines where PowerShell is blocked
    :: by group policy this fails, so fall through with a clear message
    powershell -NoProfile start -verb runas '%0' am_admin >nul 2>&1
    if errorlevel 1 (
        echo This script requires administrator rights, and automatic elevation failed.
        echo Please right-click configure.bat and select "Run as administrator".
        pause
    )
    exit /b
)

echo =========================================================================
echo.
echo   ABUS Configure [Version 4.0 - uv]
echo   contact: abus.aikorea@gmail.com
echo.
echo =========================================================================
echo.

:: If we've reached here, we're running in the correct environment
:: Your actual batch file commands start here
echo Running in 64-bit mode from System32
echo Current directory: %CD%
echo Command line: %*


:: check LongPathsEnabled
echo Enabling long paths support...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
if %ERRORLEVEL% == 0 (
    echo Long paths have been enabled successfully.
) else (
    echo Failed to enable long paths.
    pause
    exit /b 1
)



cd /D "%~dp0"
SET "CHOCPATH=%SYSTEMROOT%\System32\WindowsPowerShell\v1.0\powershell.exe"

:: Install choco
echo Checking Chocolatey installation...
where choco >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Chocolatey...
    %CHOCPATH% -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Chocolatey.
        pause
        exit /b 1
    )
    :: Add Chocolatey to PATH for current session
    if exist "%ALLUSERSPROFILE%\chocolatey\bin" (
        set "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
    )
    :: Verify choco is now available
    where choco >nul 2>&1
    if %errorlevel% neq 0 (
        echo WARNING: Chocolatey installed but choco command not found in PATH.
        echo You may need to restart the command prompt.
    )
) else (
    echo Chocolatey is already installed.
)

:: NOTE (v4.0, uv-based install):
:: - CUDA Toolkit is no longer required: PyTorch cu128 wheels bundle the CUDA
::   runtime and cuDNN. Only an NVIDIA driver (>= 525, recommended >= 570) is needed.
:: - Visual Studio Build Tools are no longer required: every Python dependency
::   now ships prebuilt wheels.

:: check NVIDIA GPU (informational only)
set IS_NVIDIA_GPU=0
set "registry_path=HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000"
set "search_key=DriverDesc"
for /f "tokens=2*" %%a in ('reg query "%registry_path%" /v "%search_key%" 2^>nul ^| findstr /i /c:"%search_key%"') do (
    set "value=%%b"
)

if not "!value!"=="" (
    echo DriverDesc is "!value!"
    set "substring=nvidia"
    echo "!value!" | findstr /I /C:"!substring!" >nul 2>&1
    if !errorlevel! equ 0 (
        echo NVIDIA GPU detected. CUDA Toolkit installation is NOT required:
        echo PyTorch wheels bundle the CUDA runtime. Just keep your NVIDIA driver up to date.
        set IS_NVIDIA_GPU=1
    )
)

echo.
echo =========================================================================
echo Installing required packages...
echo =========================================================================
echo.

echo Installing git.install...
choco install -y git.install
if %errorlevel% neq 0 (
    echo WARNING: git.install installation failed.
)

echo.
echo Installing ffmpeg...
choco install -y ffmpeg
if %errorlevel% neq 0 (
    echo WARNING: ffmpeg installation failed.
)

echo.
echo =========================================================================
echo ABUS configure.bat finished.
echo =========================================================================
echo.
echo Note: LongPathsEnabled requires a system reboot to take effect.
echo.

:: Ask user about reboot
set /p REBOOT_NOW="Do you want to reboot now? (Y/N): "
if /i "!REBOOT_NOW!"=="Y" (
    echo.
    echo System will be rebooted in 30 seconds...
    echo Press Ctrl+C to cancel.
    echo.
    for /l %%i in (30,-1,1) do (
        cls
        echo.
        echo =========================================================================
        echo ABUS Configure finished.
        echo =========================================================================
        echo.
        echo System will be rebooted in %%i seconds.
        echo Press Ctrl+C to cancel.
        timeout /t 1 /nobreak >nul 2>&1
    )
    shutdown /r /t 0
) else (
    echo.
    echo Please reboot your system manually when ready.
    echo LongPathsEnabled and CUDA changes require a reboot to take effect.
    pause
)