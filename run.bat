@echo off
set CMD=%1
if "%CMD%"=="" goto usage

if /I "%CMD%"=="demo" (
  python scripts\gym_demo.py
  goto end
)

if /I "%CMD%"=="train" (
  set TIMESTEPS=%~2
  if "%TIMESTEPS%"=="" (
    set TIMESTEPS=300000
    python scripts\train_agent.py --timesteps %TIMESTEPS%
    goto end
  )
  shift
  shift
  python scripts\train_agent.py --timesteps %TIMESTEPS% %*
  goto end
)

if /I "%CMD%"=="watch" (
  set MODEL=%2
  if "%MODEL%"=="" set MODEL=models\ppo_column_popper.zip
  python scripts\watch_agent_curses.py --model "%MODEL%"
  goto end
)

if /I "%CMD%"=="play" (
  column_popper --mode=play
  goto end
)

:usage
echo Usage:
echo   run.bat demo
echo   run.bat train [timesteps]
echo   run.bat watch [model_path]
echo   run.bat play [--initial-fall X --fall-curve T:I,...]
exit /b 1

:end
exit /b 0
