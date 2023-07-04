mkdir build
cp cmake/config.cmake build
cd build

OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    echo "Macos: $OS"
    sed -i '' 's|set(USE_LLVM OFF)|set(USE_LLVM /opt/homebrew/opt/llvm@15/bin/llvm-config)|g' config.cmake
    #sed -i '' 's|set(USE_PAPI OFF)|set(USE_PAPI ON)|g' config.cmake
elif [ "$OS" = "Linux" ]; then
    echo "Linux: $OS"
    sed -i 's|set(USE_LLVM OFF)|set(USE_LLVM /lib/llvm-15/bin/llvm-config)|g' config.cmake
    sed -i 's|set(USE_PAPI OFF)|set(USE_PAPI ON)|g' config.cmake
else
    echo "Unknown operating system."
fi

if [ "$OS" = "Darwin" ]; then
    cmake -DCMAKE_INSTALL_PREFIX=/Users/yannic/Desktop/work/MY/tvm-scripts/Install ..
elif [ "$OS" = "Linux" ]; then
    cmake -DCMAKE_INSTALL_PREFIX=/home/yannic/work/tvm-scripts/Install ..
else
    echo "Not set install path ..."
fi

make -j10
make install
cd ..

if [ "$OS" = "Darwin" ]; then
    echo "export TVM_HOME=/Users/yannic/Desktop/work/MY/tvm-scripts/tvm"
    echo "export PYTHONPATH=\$TVM_HOME/python:\${PYTHONPATH}"
elif [ "$OS" = "Linux" ]; then
    echo "export TVM_HOME=/home/yannic/work/tvm-scripts/tvm"
    echo "export PYTHONPATH=\$TVM_HOME/python:\${PYTHONPATH}"
else
    echo "You need export TVM_HOME and PYTHONPATH"
fi
