@echo off
title Invoice Intelligence Server
cd /d "D:\CSproject\Cloud\invoice-intelligence"

echo ==========================================
echo Starting Invoice Intelligence Backend...
echo ==========================================

:: فتح الداشبورد في المتصفح تلقائيا
start "" "ai\Dashboard\index.html"

:: تشغيل السيرفر بالبايثون الخاص بالـ venv مباشرة
"D:\CSproject\Cloud\invoice-intelligence\venv\Scripts\python.exe" -m uvicorn inference_api.main:app --host 127.0.0.1 --port 8000 --reload

:: لو حصل أي إيرور الشاشة مش هتقفل وهتقولك السبب
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server failed to start! Check the error above.
    pause
)
pause