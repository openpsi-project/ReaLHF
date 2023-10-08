#!/bin/bash

trap 'echo "Ignoring error"' ERR

for ((i=0; i<7; i++)); do
    echo "Running opt-2x8-params-r3-$i"
    python3 -m apps.main start -e opt-2x8-params-r3-$i -f 20231008-01
done

# python3 -m apps.main start -e starcoder-2x8-16b -f 20231006-01