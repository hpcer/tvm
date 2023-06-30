mkdir build
cp cmake/config.cmake build
cd build

sed -i '' 's|set(USE_LLVM OFF)|set(USE_LLVM /opt/homebrew/opt/llvm@15/bin/llvm-config)|g' config.cmake
sed -i '' 's|set(USE_PAPI OFF)|set(USE_PAPI ON)|g' config.cmake


cmake -DCMAKE_INSTALL_PREFIX=/Users/yannic/Desktop/work/MY/tvm-scripts/Install ..
make -j10
make install
cd ..

echo "export TVM_HOME=/Users/yannic/Desktop/work/MY/tvm-scripts/tvm"
echo "export PYTHONPATH=\$TVM_HOME/python:\${PYTHONPATH}"