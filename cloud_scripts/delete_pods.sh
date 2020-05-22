#!/bin/bash
NUMBATCHES=1
for ((BATCHIDX=0;BATCHIDX<1;BATCHIDX++)) do
    for ((PIDX=60000;PIDX<60100;PIDX++)) do
        for ((PLANSEED=0;PLANSEED<5;PLANSEED++)) do
              kubectl delete job.batch/guiding-gtamp-learn-$BATCHIDX-$PIDX-$PLANSEED -n beomjoon
          done
      done
  done

