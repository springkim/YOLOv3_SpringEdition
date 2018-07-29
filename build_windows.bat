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
::nvcc -V | findstr /C:"release 9.1" > cuda_ver.txt
::set /p "ver91="<"cuda_ver.txt"
::if not "%ver91%" equ "" (
::	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.dropbox.com/s/0487mtsb4dao5xb/cudnn.h?dl=1','3rdparty\include\cudnn.h')"
::	powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.dropbox.com/s/l64r6jbkw0fe5p4/cudnn.lib?dl=1','3rdparty\lib\cudnn.lib')"
::)
DEL cuda_ver.txt
md build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
cmake --build . --config Release --target ALL_BUILD
pause