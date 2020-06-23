#!/bin/bash

# get directory
# from https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
	DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
	SOURCE="$(readlink "$SOURCE")"
	[[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

for ((BATCHIDX=0;BATCHIDX<$NUMBATCHES;BATCHIDX++)); do cat ${DIR}/${SOLVER}.yaml | sed "s/COMMIT/$COMMIT/g" | sed "s/DOMAINS/$DOMAINS/g" | sed "s/PIDX_RANGE/{$((${BASEPIDX} + ${BATCHIDX}*${BATCHSIZE}))..$((${BASEPIDX} + (${BATCHIDX}+1)*${BATCHSIZE} - 1))}/g" | sed "s/NUMOBJ_RANGE/$NUMOBJ_RANGE/g" | sed "s/SEEDS/$SEEDS/g" | sed "s/NAME/guiding-gtamp-$BATCHIDX/g" | sed "s/OPENSTACKSECRETSREF/$OPENSTACKSECRETSREF/g" |  sed "s/TIMELIMIT/$TIMELIMIT/g" | kubectl $ACTION -f - -n $NAMESPACE; done

