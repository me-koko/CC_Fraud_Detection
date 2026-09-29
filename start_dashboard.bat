@echo off
title CyberGuard AI - Fraud Detection System
cd /d "D:\CC_Fraud_Detection"
echo ========================================================
echo   Starting CyberGuard AI - Fraud Detection System
echo ========================================================
echo.
echo Opening dashboard in your browser...
start http://127.0.0.1:5000
echo.
python app.py
pause
