#!/bin/bash
#
# Compute results after downloading with ./dal_download_results.sh
#
results="cv-results"
datasets=(hh{101..130})
folds=({0..2})

for target in "${datasets[@]}"; do
    for fold in "${folds[@]}"; do
        for f in "$results/$target-$fold"-{source,target}-{training,validation}.csv; do
            # Skip if it doesn't exist
            [[ ! -f $f ]] && continue
            # Skip if the results weren't available (i.e. tensorboard gives 500 error)
            grep -q "500 Internal Server Error" "$f" && continue

            #echo -en "$f\t"
            tail -n 1 "$f" | cut -d',' -f3
        done
    done
done | awk '
{
    i=(NR-1)%4
    if (i==0)      { sum1+=$1; sumsq1+=($1)^2; }
    else if (i==1) { sum2+=$1; sumsq2+=($1)^2; }
    else if (i==2) { sum3+=$1; sumsq3+=($1)^2; }
    else if (i==3) { sum4+=$1; sumsq4+=($1)^2; }
} END {
    n=NR/4
    std1=sqrt((sumsq1-sum1^2/n)/n)
    std2=sqrt((sumsq2-sum2^2/n)/n)
    std3=sqrt((sumsq3-sum3^2/n)/n)
    std4=sqrt((sumsq4-sum4^2/n)/n)
    print "Averages over " n " homes (and each home is an average of 3-fold CV)"
    print "Train A* \t Avg: " sum1/n "\t Std: " std1
    print "Test A  \t Avg: " sum2/n "\t Std: " std2
    print "Train B* \t Avg: " sum3/n "\t Std: " std3
    print "Test B  \t Avg: " sum4/n "\t Std: " std4
    print ""
    print "* only on batch of 1024 examples for training sets (different than AL)"
}'
