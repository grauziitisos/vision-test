#!/bin/bash

for f in ./*.py; do
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	echo "Running $f >>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    python3 "$f"
    
    if [ $? -eq 0 ]; then
        echo "<<<<<<<<<<<<<<<<<<<<<< finished $f"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!! Error with $f"
    fi

done
