@echo off
title download_yolov3_darknet53_spp.weights
powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/image/yolov3_darknet53_spp.weights','yolov3_darknet53_spp.weights')"
pause