powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/image/stl10valid.zip','%TEMP%\stl10valid.zip')"
powershell -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%TEMP%\stl10valid.zip', '.'); }"
IF EXIST "%TEMP%\stl10valid.zip" (
	DEL "%TEMP%\stl10valid.zip"
)