#!/bin/bash
#
# Translate home sensor and activity names to be consistent
#
if [[ -z $1 ]]; then
    echo "Usage: time ./translate.sh ../raw/*/*.txt"
    exit 1
fi

translate_str_end() {
    # Generate sed replacements, e.g. "s# Maintenance# Other_Activity#g" and
    # then the next on another line... (spaces only on left and end of line at
    # right, since line ends on right)
    awk '{print "s# " $1 "$# " $2 "#"}' "$1"
}

translate_str_middle() {
    # Generate sed replacements, e.g. "s# M001 # OutsideDoor #g" and then the next
    # on another line... (middle of line, so spaces on both sides)
    awk '{print "s# " $1 " # " $2 " #"}' "$1"
}

ignore() {
    grep -v "Ignore" | grep -v "Other_Activity"
}

# Only do once since this is the same for them all
activity="translate/act.translate"
activity_replacements="$(translate_str_end "$activity")"

process() {
    i="$1"
    echo "Processing $i"

    # e.g. if ../raw/rw/rw101.txt this will make a "rw" directory
    dir=$(basename -- $(dirname -- "$i"))
    mkdir -p "$dir"

    # e.g. if ../raw/rw/rw101.txt this will get "rw101"
    filename=$(basename -- "$i")
    filename="${filename%.*}"

    # e.g. if ../raw/rw/rw101.txt this will output to a file rw/rw101.hdf5
    input="$i"
    output="${dir}/${filename}.txt"
    translate="translate/${filename}.translate"

    # If translate file doesn't exist, use the generic one for the type of
    # house (e.g. no rw103, so just use rw.translate)
    if [[ ! -f $translate ]]; then
        translate="translate/${filename:0:2}.translate"
    fi

    # Make both sets of replacements and then drop all Ignore and
    # Other_Activity lines
    translate_replacements="$(translate_str_middle "$translate")"

    # Ignore both before and after since most of the files are ignored, this
    # makes it much faster
    ignore < "$input" | sed "$translate_replacements" | \
        sed "$activity_replacements" | ignore > "$output"
}

. /scripts/threading
thread_init

for i; do
    process "$i" &
    thread_wait
done

thread_finish
