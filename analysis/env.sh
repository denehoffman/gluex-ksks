export GLUEX_TOP=/home/gluex/gluex_top2
export BUILD_SCRIPTS=/home/gluex/gluex_top2/build_scripts
source $GLUEX_TOP/gluex_env_local.sh "${0##*/}"/version.xml
export JANA_CALIB_CONTEXT="default"
export CCDB_CONNECTION=sqlite:////home/gluex/gluexdb/ccdb_2024_10_25.sqlite
export RCDB_CONNECTION=sqlite:////home/gluex/gluexdb/rcdb_2024_10_25.sqlite
