source ../../Tools/XK_setup.bash

module load perftools

for version in 01 02 03
do

    make clean
    make VERSION=${version}
    pat_build -u himeno_v${version}.x
    bash submit.bash himeno_v${version}.x+pat

    make clean
    make VERSION=${version} ACC_MODEL=auto_async_none
    pat_build -u himeno_v${version}.x
    bash submit.bash himeno_v${version}.x+pat

done
