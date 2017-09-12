IF EXIST "x64" (
	RMDIR /S /Q "x64"
)
IF EXIST "network\yolo.weights" (
	DEL "network\yolo.weights"
)