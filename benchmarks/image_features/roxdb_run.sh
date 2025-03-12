#!/usr/bin/env bash

DB_PATH="/tmp/roxdb"


BUILD_PATH="./benchmarks/image_features"
ADD="${BUILD_PATH}/roxdb_add"
SEARCH="${BUILD_PATH}/roxdb_search"

BASE_VEC=$HOME/image_features/image_features_100k.h5
QUERY_VEC=$HOME/image_features/image_features_query_1k.h5

# export OMP_NUM_THREADS=16
# rm -rf $DB_PATH
# echo "Adding to database"
# $ADD $DB_PATH $BASE_VEC

export OMP_NUM_THREADS=32
echo "Searching"
$SEARCH $DB_PATH $QUERY_VEC  --evaluate