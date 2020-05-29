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

for ((BATCHIDX=0;BATCHIDX<$NUMBATCHES;BATCHIDX++)); do cat ${DIR}/${SOLVER}.yaml | sed "s/COMMIT/$COMMIT/" | sed "s/DOMAINS/$DOMAINS/" | sed "s/PIDX_RANGE/{$((${BASEPIDX} + ${BATCHIDX}*${BATCHSIZE}))..$((${BASEPIDX} + (${BATCHIDX}+1)*${BATCHSIZE} - 1))}/" | sed "s/NUMOBJ_RANGE/$NUMOBJ_RANGE/" | sed "s/SEEDS/$SEEDS/" | sed "s/NAME/guiding-gtamp-$BATCHIDX/" | sed "s/OPENSTACKSECRETSREF/$OPENSTACKSECRETSREF/" |  kubectl $ACTION -f - -n $NAMESPACE; done

