mkdir build
cd build
if [ -f "/usr/local/cuda/lib64/libcudnn.so" ]; then
	#Default cuda path
    cmake .. -DCUDNN=/usr/local/cuda/lib64/libcudnn.so
	make
elif [ -f "/usr/bin/libcudnn.so" ]; then
	#global path
    cmake .. -DCUDNN=/usr/bin/libcudnn.so
	make
elif [ -f "/usr/lib/aarch64-linux-gnu/libcudnn.so" ]; then
	#jetson cuda path(jetpack)
    cmake .. -DCUDNN=/usr/lib/aarch64-linux-gnu/libcudnn.so
	make
else
	echo "there is no cudnn in your PC";
fi
