#!/bin/sh
while true; do
	#clear
	rocm-smi --showmeminfo vram | grep Used
	sleep 0.01
done
