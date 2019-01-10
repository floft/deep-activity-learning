#!/bin/bash
#
# Run AL outputting to text file
#
[[ ! -e al_results.txt ]] && python3 al.py | tee al_results.txt

#
# Get averages, average each of the set of 4 numbers
#
grep -Eo 'Avg: [0-9\.]+' al_results.txt | sed 's/Avg: //' | awk '
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
    print "Train A \t Avg: " sum1/n "\t Std: " std1
    print "Test A  \t Avg: " sum2/n "\t Std: " std2
    print "Train B \t Avg: " sum3/n "\t Std: " std3
    print "Test B  \t Avg: " sum4/n "\t Std: " std4
}'
