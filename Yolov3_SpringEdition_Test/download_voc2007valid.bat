@echo off
title download_voc2007valid
powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/image/voc2007valid.zip','%TEMP%\voc2007valid.zip')"
powershell -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%TEMP%\voc2007valid.zip', '.'); }"
IF EXIST "%TEMP%\voc2007valid.zip" (
	DEL "%TEMP%\voc2007valid.zip"
)
