@echo off
mkdir build\Release
nvcc -V | findstr /C:"release 8.0" > cuda_ver.txt
set /p "ver80="<"cuda_ver.txt"
if not "%ver80%" equ "" (
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda8.0/cudnn.h','3rdparty\include\cudnn.h')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda8.0/cudnn.lib','3rdparty\lib\cudnn.lib')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda8.0/cudnn64_7.dll','build\Release\cudnn64_7.dll')"
)
nvcc -V | findstr /C:"release 9.0" > cuda_ver.txt
set /p "ver90="<"cuda_ver.txt"
if not "%ver90%" equ "" (
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda9.0/cudnn.h','3rdparty\include\cudnn.h')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda9.0/cudnn.lib','3rdparty\lib\cudnn.lib')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.1cuda9.0/cudnn64_7.dll','build\Release\cudnn64_7.dll')"
)
nvcc -V | findstr /C:"release 10.0" > cuda_ver.txt
set /p "ver100="<"cuda_ver.txt"
if not "%ver100%" equ "" (
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.5cuda10.0/cudnn.h','3rdparty\include\cudnn.h')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.5cuda10.0/cudnn.lib','3rdparty\lib\cudnn.lib')"
	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/cudnn7.5cuda10.0/cudnn64_7.dll','build\Release\cudnn64_7.dll')"
)
DEL cuda_ver.txt
md build
cd build
cmake .. -G "Visual Studio 12 2013 Win64"
cmake --build . --config Release --target ALL_BUILD
pause