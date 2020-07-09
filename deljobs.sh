#!/bin/bash
for j in $(kubectl get jobs -o custom-columns=:.metadata.name -n beomjoon)
do
    #if [[ "$j" == *"one-arm"* ]]; then
      echo $j
      kubectl delete jobs/$j -n beomjoon
    #fi
done
