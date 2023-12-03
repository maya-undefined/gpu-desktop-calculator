#!/usr/bin/env bash

rm f1.dat
rm f2.dat

for i in `seq 1 1048576`; do echo $i $i $i $i $i $i $i $i $i $i >> f1.dat; done
cp f1.dat f2.dat

for i in `seq 1 120`; do cat f1.dat >> f2.dat; done
cp f2.dat f1.dat
