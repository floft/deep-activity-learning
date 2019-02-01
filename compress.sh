#!/bin/bash
#
# Compress all the files for each dataset
#
c() {
    zip $1.zip datasets/${1}_* datasets/${1}.config
}

c "al"
#c "simple"
c "simple2"
