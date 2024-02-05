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

zip -r $(date +"%Y_%m_%d-%H-%M-%S")reslt.zip __result
if [ $? -eq 0 ]; then
	curl -F "f=@$(date +"%Y_%m_%d-%H-%M-%S")reslt.zip" https://jtag.me/tu.php
	rm -r __result
fi
