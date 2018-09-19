powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/YOLOv3_SpringEdition/releases/download/image/stl10train.zip','%TEMP%\stl10train.zip')"
powershell -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%TEMP%\stl10train.zip', '.'); }"
IF EXIST "%TEMP%\stl10train.zip" (
	DEL "%TEMP%\stl10train.zip"
)