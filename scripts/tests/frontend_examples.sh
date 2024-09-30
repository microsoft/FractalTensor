#!/bin/bash
set -e

example_dir='examples'
for path1 in $example_dir/*; do
    if test -d $file1; then
        file=$(echo $path1 | awk -F "/" '{print $NF}')
        path="$path1/$file.py"

        if [ -f "$path" ]; then
            echo -e "\n====== Running $file ======\n"
            python3 $path 2>&1 | tee -a $logfile
            echo -e "\n====== Finished running $file ======\n\n"
        fi
    fi
done

files=$(ls *.gv 2>/dev/null | wc -l)
if [ "$files" != "0" ]; then
    rm *.gv
fi

files=$(ls *.pdf 2>/dev/null | wc -l)
if [ "$files" != "0" ]; then
    rm *.pdf
fi
