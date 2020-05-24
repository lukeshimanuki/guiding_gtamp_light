#!/bin/bash
NUMBATCHES=1
for ((BATCHIDX=0;BATCHIDX<1;BATCHIDX++)) do
    for ((PIDX=20000;PIDX<20100;PIDX++)) do
        for ((PLANSEED=0;PLANSEED<5;PLANSEED++)) do
              kubectl delete job.batch/rsc-one-arm-mover-$PIDX-$PLANSEED -n beomjoon
          done
      done
  done

