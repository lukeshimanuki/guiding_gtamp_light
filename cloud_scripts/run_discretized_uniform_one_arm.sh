#!/bin/bash
NUMBATCHES=1
for ((PIDX=20000;PIDX<=20100;PIDX++)) do
    for ((PLANSEED=0;PLANSEED<5;PLANSEED++)) do
          cat cloud_scripts/run_discretized_uniform_one_arm.yaml |
          sed "s/NAME/guiding-gtamp-discretized-uniform-$PIDX-$PLANSEED/" |
          sed "s/PIDX/$PIDX/" |
          sed "s/PLANSEED/$PLANSEED/" |  kubectl apply -f - -n beomjoon;
      done
  done

