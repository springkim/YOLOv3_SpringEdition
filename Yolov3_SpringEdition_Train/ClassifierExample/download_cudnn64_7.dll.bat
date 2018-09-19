@echo off
title download_cudnn64_7.dll
set cudir=%CD%
cd %TEMP%
nvcc -V | findstr /C:"release 8.0" > cuda_ver.txt
set /p "ver80="<"cuda_ver.txt"
if not "%ver80%" equ "" (
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda8.0/cudnn64_7.dll','cudnn64_7.dll')"
)
nvcc -V | findstr /C:"release 9.0" > cuda_ver.txt
set /p "ver90="<"cuda_ver.txt"
if not "%ver90%" equ "" (
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda9.0/cudnn64_7.dll','cudnn64_7.dll')"
)
DEL cuda_ver.txt
move cudnn64_7.dll "%cudir%"
::nvcc -V | findstr /C:"release 9.1" > cuda_ver.txt
::set /p "ver91="<"cuda_ver.txt"
::if not "%ver91%" equ "" (
::	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.dropbox.com/s/l0zsm9g78ciexzh/cudnn64_7.dll?dl=1','cudnn64_7.dll')"
::)
pause