#!/bin/bash
NUMBATCHES=1
for ((BATCHIDX=0;BATCHIDX<1;BATCHIDX++)) do
    for ((PIDX=60099;PIDX<=60099;PIDX++)) do
        for ((PLANSEED=2;PLANSEED<=2;PLANSEED++)) do
              cat liscloud_run_sahs_learn.yaml |
              sed "s/NAME/guiding-gtamp-learn-$BATCHIDX-$PIDX-$PLANSEED/" |
              sed "s/PIDX/$PIDX/" |
              sed "s/PLANSEED/$PLANSEED/" |  kubectl apply -f - -n beomjoon;
          done
      done
  done

