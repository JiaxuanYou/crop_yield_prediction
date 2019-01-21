#!
# convert_2to3.sh

for py_file in $(find . -name '*.py'); do
  echo $py_file
  # 2to3 -w $py_file
done
