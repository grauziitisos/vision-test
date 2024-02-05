#!/bin/bash
if [ $# -eq 0 ]; then
    echo "$(date +"%Y_%m_%d-%H-%M-%S") jāpadod procesa id, kuram mērīt atmiņas lietojumu: $0 <procesa_id>"
    exit 1
fi

pid=$1

of="$(date +"%Y_%m_%d-%H-%M-%S")_memory_$pid.txt"

if ! ps -p $pid > /dev/null; then
    echo "$(date +"%Y_%m_%d-%H-%M-%S") ar $pid nav tāds process šobrīd..."
    exit 1
fi

while true; do
	echo "$(date +"%Y_%m_%d-%H-%M-%S") mem: $(ps -o rss= -p $pid) KB, whole sys: $(free -m | awk 'NR==2 {print $3}') MB" >> $of
	sleep 1
done
