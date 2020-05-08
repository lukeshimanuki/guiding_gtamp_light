#!/bin/bash

for ((BATCHIDX=0;BATCHIDX<$NUMBATCHES;BATCHIDX++)); do cat liscloud_template.yaml | sed "s/COMMIT/$COMMIT/" | sed "s/DOMAINS/$DOMAINS/" | sed "s/PIDX_RANGE/{$((${BASEPIDX} + ${BATCHIDX}*${BATCHSIZE}))..$((${BASEPIDX} + (${BATCHIDX}+1)*${BATCHSIZE} - 1))}/" | sed "s/NUMOBJ_RANGE/$NUMOBJ_RANGE/" | sed "s/GPUS/$GPUS/" | sed "s/SOLVER/$SOLVER/" | sed "s/SEEDS/$SEEDS/" | sed "s/NAME/guiding-gtamp-$BATCHIDX/" | sed "s/OPENSTACKSECRETSREF/$OPENSTACKSECRETSREF/" |  kubectl $ACTION -f - -n $NAMESPACE; done

