#!/bin/sh

#$ -S /bin/sh
#$ -l ncpus=4 s_vmem=32G -l mem_req=32G
#$ -pe def_stat 4
#$ -N QSUB_TEST

if [ -n "$PBS_O_WORKDIR" ]; then
    echo PBS_O_WORKDIR = $PBS_O_WORKDIR
    DIR=${PBS_O_WORKDIR}/
else
    echo "dirname $0 = `dirname $0`"
    DIR=`dirname $0`/
fi

if [ -n "$NCPUS" ]; then
    echo NCPUS = $NCPUS
fi

${DIR}target/release/parallel-execution "$@"
