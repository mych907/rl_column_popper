@echo off
setlocal EnableExtensions EnableDelayedExpansion
set CMD=%1
if "%CMD%"=="" goto usage

if /I "%CMD%"=="demo" (
  python scripts\gym_demo.py
  goto end
)

if /I "%CMD%"=="train" (
  rem First arg after 'train' is timesteps; default if missing
  set TIMESTEPS=%~2
  if "!TIMESTEPS!"=="" set TIMESTEPS=300000
  rem Pass through any remaining args after timesteps
  set "EXTRA=%3 %4 %5 %6 %7 %8 %9"
  python scripts\train_agent.py --timesteps !TIMESTEPS! !EXTRA!
  goto end
)

if /I "%CMD%"=="watch" (
  set MODEL=%~2
  if "!MODEL!"=="" set MODEL=models\ppo_column_popper.zip
  python scripts\watch_agent_curses.py --model "!MODEL!"
  goto end
)

if /I "%CMD%"=="play" (
  column_popper --mode=play
  goto end
)

:usage
echo Usage:
echo   run.bat demo
echo   run.bat train [timesteps] [extra args]
echo   run.bat watch [model_path]
echo   run.bat play [--initial-fall X --fall-curve T:I,...]
exit /b 1

:end
endlocal & exit /b 0
