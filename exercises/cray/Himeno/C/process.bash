loglist=$(ls $1/*/*.log)

for log in $loglist
do
    awk '/Gosa/ {gosa[1]=$3};/Gosa/ {gosa[2]=$3};/MFLOPS/ {mflops=$3};
         END {print "MFLOPS: "mflops"   Gosa values: "gosa[1]"   "gosa[2]}' \
	     $log
done
