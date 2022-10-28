#!/bin/bash
#Declare a string array # "02"  "05"   "08"  "11"
#    "06"  "07"  "09"   "12") "03"  "04"  
# LanguageArray=("06"  "07"  "09"  "10"  "12")
LanguageArray=("02"  "05"   "08"  "11")
 
# Print array values in  lines
echo "Print every element in new line"
for val1 in ${LanguageArray[*]}; do
     echo $val1
     mkdir /ais/hal9000/datasets/reddit/stance_analysis/2014_${val1}_files
     split -l 500000 /ais/hal9000/datasets/reddit/stance_analysis/RC_2014-${val1}_counts.json /ais/hal9000/datasets/reddit/stance_analysis/2014_${val1}_files/
done