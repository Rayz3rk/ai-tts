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



echo =========================================================================
echo.
echo   ABUS Updater [Version 4.0 - uv]
echo   contact: abus.aikorea@gmail.com
echo.
echo =========================================================================
echo.

cd /D "%~dp0"
set PATH=%PATH%;%SystemRoot%\system32


:: fix failed install when installing to a separate drive
set TMP=%cd%\installer_files
set TEMP=%cd%\installer_files


:: config
set INSTALL_DIR=%cd%\installer_files
set INSTALL_ENV_DIR=%cd%\installer_files\env
set UV_VERSION=0.11.28
set UV_DIR=%INSTALL_DIR%\uv
set UV_EXE=%UV_DIR%\uv.exe

:: keep everything project-local (no global uv/python/cache pollution)
set UV_PYTHON_INSTALL_DIR=%INSTALL_DIR%\python
set UV_CACHE_DIR=%INSTALL_DIR%\uv-cache
set UV_PROJECT_ENVIRONMENT=%INSTALL_ENV_DIR%

:: environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=


:: (if necessary) download uv into a contained folder
:: NOTE: extraction must use the built-in bsdtar with its full path. A bare `tar` may
:: resolve to GNU tar from Git which cannot read zip files, and PowerShell
:: Expand-Archive may be blocked by group policy on corporate machines.
if not exist "%UV_EXE%" (
	echo Downloading uv %UV_VERSION% to %UV_DIR%
	mkdir "%UV_DIR%" 2>nul
	call curl -Lk "https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip" -o "%INSTALL_DIR%\uv.zip" || ( echo. && echo uv failed to download. && goto end )
	"%SystemRoot%\System32\tar.exe" -xf "%INSTALL_DIR%\uv.zip" -C "%UV_DIR%" || ( echo. && echo uv failed to extract. && goto end )
	del "%INSTALL_DIR%\uv.zip"
)


:: figure out the GPU choice: GPU_CHOICE env var > saved choice > NVIDIA registry autodetect
set SAVED_CHOICE_FILE=%INSTALL_DIR%\gpu_choice.txt
if not defined GPU_CHOICE (
	if exist "%SAVED_CHOICE_FILE%" (
		set /p GPU_CHOICE=<"%SAVED_CHOICE_FILE%"
	)
)
if not defined GPU_CHOICE (
	set GPU_CHOICE=C
	set "registry_path=HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000"
	for /f "tokens=2*" %%a in ('reg query "!registry_path!" /v "DriverDesc" 2^>nul ^| findstr /i /c:"DriverDesc"') do (
		set "value=%%b"
	)
	if not "!value!"=="" (
		echo "!value!" | findstr /I /C:"nvidia" >nul 2>&1 && set GPU_CHOICE=G
		echo "!value!" | findstr /I /C:"tesla" >nul 2>&1 && set GPU_CHOICE=G
	)
)
echo !GPU_CHOICE!>"%SAVED_CHOICE_FILE%"

if /i "!GPU_CHOICE!"=="G" (
	set SYNC_EXTRA=gpu
) else (
	set SYNC_EXTRA=cpu
)
echo Updating environment (extra: !SYNC_EXTRA!) from uv.lock ...


:: update the environment to exactly match the committed lockfile
call "%UV_EXE%" sync --frozen --extra !SYNC_EXTRA! || ( echo. && echo Environment update failed. && goto end )

echo.
echo Update finished successfully.

:end
pause
