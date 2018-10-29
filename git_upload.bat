@echo off
title github upload
set /p msg=Enter commit msg:
git add -A
git commit -m "%msg%"
git push origin master
pause