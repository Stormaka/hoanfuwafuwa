@echo off
chcp 65001 >nul
title Unity Car Hand Tracking - Launcher
color 0B

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║       UNITY CAR HAND TRACKING - AUTO LAUNCHER               ║
echo  ║       Unity 2022.3.0f1  +  Python MediaPipe                 ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: ── Đường dẫn ────────────────────────────────────────────────────────────────
set PROJECT_DIR=%~dp0Unity-Hand-Tracking-Computer-Vision-main
set UNITY_EXE=C:\Program Files\Unity\Hub\Editor\6000.4.4f1\Editor\Unity.exe
set PYTHON_SCRIPT=%PROJECT_DIR%\hand_tracking.py

:: ── Kiểm tra Unity Editor ────────────────────────────────────────────────────
echo [1/3] Kiem tra Unity Editor...
if not exist "%UNITY_EXE%" (
    echo [WARNING] Khong tim thay Unity 6000.4.4f1
    echo           Tim kiem phien ban khac...
    for /d %%i in ("C:\Program Files\Unity\Hub\Editor\*") do (
        if exist "%%i\Editor\Unity.exe" (
            set UNITY_EXE=%%i\Editor\Unity.exe
            echo [OK] Tim thay Unity: %%i
        )
    )
)

if not exist "%UNITY_EXE%" (
    echo [ERROR] Khong tim thay Unity Editor!
    echo         Vui long cai Unity Hub tai: https://unity.com/download
    pause
    exit /b 1
)

echo [OK] Unity: %UNITY_EXE%
echo.

:: ── Kiểm tra Project ─────────────────────────────────────────────────────────
echo [2/3] Kiem tra Unity Project...
if not exist "%PROJECT_DIR%\ProjectSettings\ProjectVersion.txt" (
    echo [ERROR] Khong tim thay Unity project tai: %PROJECT_DIR%
    pause
    exit /b 1
)
echo [OK] Project: %PROJECT_DIR%
echo.

:: ── Khởi động Unity ──────────────────────────────────────────────────────────
echo [3/3] Khoi dong Unity Editor...
echo       (Qua trinh nay co the mat 1-2 phut de compile scripts)
echo.
start "" "%UNITY_EXE%" -projectPath "%PROJECT_DIR%"

:: Chờ Unity khởi động
echo Dang cho Unity khoi dong (30 giay)...
timeout /t 30 /nobreak >nul

:: ── Khởi động Python ─────────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║  Chu y: Trong Unity, nhan Play truoc khi chay Python!       ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
echo Nhan phim bat ky de khoi dong Python Hand Tracking...
pause >nul

echo.
echo Dang khoi dong Python Hand Tracking...
python "%PYTHON_SCRIPT%"

echo.
echo [INFO] Chuong trinh da ket thuc.
pause
