@echo off
REM Startup script for Face Recognition Attendance System GUI
REM Uses Python 3.10 virtual environment

echo ========================================
echo Face Recognition Attendance System
echo ========================================
echo.
echo Starting GUI application...
echo.

REM Activate virtual environment and run GUI
.\venv310\Scripts\python.exe gui\attendance_gui.py

pause
