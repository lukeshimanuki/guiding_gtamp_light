#!/bin/bash
NUMBATCHES=1
for ((BATCHIDX=0;BATCHIDX<1;BATCHIDX++)) do
    for ((PIDX=60000;PIDX<=60100;PIDX++)) do
        for ((PLANSEED=0;PLANSEED<=4;PLANSEED++)) do
              cat ./cloud_scripts/run_learned_two_arm.yaml | sed "s/NAME/guiding-gtamp-learn-$BATCHIDX-$PIDX-$PLANSEED/" | sed "s/PIDX/$PIDX/" | sed "s/PLANSEED/$PLANSEED/" |  kubectl apply -f - -n beomjoon;
          done
      done
  done

