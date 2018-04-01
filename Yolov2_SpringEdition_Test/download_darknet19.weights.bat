@echo off
powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.dropbox.com/s/x241flu1lwy2h2x/darknet19.weights?dl=1','darknet19.weights')"
pause