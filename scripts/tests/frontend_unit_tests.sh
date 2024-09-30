#!/bin/bash
set -e
logfile="unittest.log"

if [ -f "$logfile" ]; then
    rm $logfile
fi

for ut in $(find kaleido/frontend/tests/test_*py); do
    echo "runing $ut"
    python3 $ut 2>&1 | tee -a $logfile
done

for ut in $(find kaleido/frontend/operations/tests/test_*py); do
    python3 $ut 2>&1 | tee -a $logfile
done

for ut in $(find kaleido/parser/tests/test_*.py); do
    python3 $ut 2>&1 | tee -a $logfile
done
