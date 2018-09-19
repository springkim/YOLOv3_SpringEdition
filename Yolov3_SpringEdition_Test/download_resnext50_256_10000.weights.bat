@echo off
title download_resnext50_256_10000.weights
powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/image/resnext50_256_10000.weights','resnext50_256_10000.weights')"
pause