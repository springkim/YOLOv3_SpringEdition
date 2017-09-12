IF NOT EXIST "install" (
	MKDIR "install"
)
IF EXIST "x64\Release" (
	XCOPY /Y "x64\Release\*.dll" "install\"
)
IF EXIST "x64\Debug" (
	XCOPY /Y "x64\Debug\*.dll" "install\"
)
XCOPY /Y "prj_YOLOv2_dll\install\*.*" "install\"