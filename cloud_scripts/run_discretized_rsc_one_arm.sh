#!/bin/bash
NUMBATCHES=1
for ((PIDX=20000;PIDX<=20000;PIDX++)) do
    for ((PLANSEED=0;PLANSEED<1;PLANSEED++)) do
          cat cloud_scripts/run_discretized_rsc_one_arm.yaml |
          sed "s/NAME/guiding-gtamp-discretized-rsc-$PIDX-$PLANSEED/" |
          sed "s/PIDX/$PIDX/" |
          sed "s/PLANSEED/$PLANSEED/" |  kubectl apply -f - -n beomjoon;
      done
  done

