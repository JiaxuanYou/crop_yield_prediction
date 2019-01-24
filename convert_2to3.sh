#!
# convert_2to3.sh
# RUN:
#   nohup bash convert_2to3.sh &

# create the py3 compatible files and save backups of py2 files
for py_file in $(find . -name '*.py'); do
  # echo $py_file
  2to3 -w $py_file
done

# move the .bak backups into a new python2 dir
for dir in */; do
  cd $dir
  mkdir python2
  mv *.bak python2
  cd ..
done

echo "***** Converted to Python3! *****"
