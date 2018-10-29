@echo off
nvcc -V | findstr /C:"release 8.0" > cuda_ver.txt
set /p "ver80="<"cuda_ver.txt"
if not "%ver80%" equ "" (
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda8.0/cudnn.h','3rdparty\include\cudnn.h')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda8.0/cudnn.lib','3rdparty\lib\cudnn.lib')"
)
nvcc -V | findstr /C:"release 9.0" > cuda_ver.txt
set /p "ver90="<"cuda_ver.txt"
if not "%ver90%" equ "" (
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda9.0/cudnn.h','3rdparty\include\cudnn.h')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda9.0/cudnn.lib','3rdparty\lib\cudnn.lib')"
)
nvcc -V | findstr /C:"release 10.0" > cuda_ver.txt
set /p "ver100="<"cuda_ver.txt"
if not "%ver100%" equ "" (
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.3.1_cuda10.0/cudnn.h','3rdparty\include\cudnn.h')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.3.1_cuda10.0/cudnn.lib','3rdparty\lib\cudnn.lib')"
)
DEL cuda_ver.txt
md build
cd build
if exist "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\MSBuild.exe" (
	cmake .. -G "Visual Studio 12 2013 Win64"
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\MSBuild.exe" (
	cmake .. -G "Visual Studio 14 2015 Win64"
)
if exist 'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\amd64\MSBuild.exe' (
	cmake .. -G "Visual Studio 15 2017 Win64"
)
cmake --build . --config Release --target ALL_BUILD
pause