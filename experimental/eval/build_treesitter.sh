mkdir ts_package;
cd ts_package;
# Download the tree-sitter package
git clone https://github.com/tree-sitter/tree-sitter-python.git;
git clone https://github.com/tree-sitter/tree-sitter-java.git;
git clone https://github.com/tree-sitter/tree-sitter-c-sharp.git;
git clone https://github.com/tree-sitter/tree-sitter-typescript.git;
cd ..;
# Build tree-sitter
python build_ts_lib.py