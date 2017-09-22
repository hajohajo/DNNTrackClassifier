#!/bin/bash

MAINDIR=$PWD
SCRIPTDIR_TRAIN=$PWD/scripts_train
SCRIPTDIR_TEST=$PWD/scripts_test
OUTPUTDIR_TRAIN=$PWD/outputRoots_trackingNtuples_Train
OUTPUTDIR_TEST=$PWD/outputRoots_trackingNtuples_Test
LOGDIR_TRAIN=$MAINDIR/logs_train
LOGDIR_TEST=$MAINDIR/logs_test
CMSDIR=$CMSSW_BASE/src

ALGOS=(initialStep lowPtQuadStep highPtTripletStep lowPtTripletStep detachedQuadStep detachedTripletStep pixelPairStep mixedTripletStep pixelLessStep tobTecStep jetCoreRegionalStep)

filelist_train=trainFileList.txt
filelist_test=testFileList.txt

condorFile_train=$SCRIPTDIR_TRAIN/submitJobsNtuple.sh
condorFile_test=$SCRIPTDIR_TEST/submitJobsNtuple.sh


mkdir -p $SCRIPTDIR_TRAIN
mkdir -p $SCRIPTDIR_TEST

mkdir -p $OUTPUTDIR_TRAIN
mkdir -p $OUTPUTDIR_TEST

mkdir -p $LOGDIR_TRAIN
mkdir -p $LOGDIR_TEST

if [ -e $condorFile_train ]
then
    	rm -rf $condorFile_train
fi
touch $condorFile_train
chmod a+x $condorFile_train

if [ -e $condorFile_test ]
then
    	rm -rf $condorFile_test
fi
touch $condorFile_test
chmod a+x $condorFile_test

#filelist=testFileList.txt
#filelist=trainFileList.txt

#Create batch job submission scripts for each ntuple for training
i=0
while read line in filelist_train
        do
	for algo in "${ALGOS[@]}"
        do
          	runScript=$SCRIPTDIR_TRAIN/runJobsStep3_${algo}_${i}.sh
                if [ -e $runScript ]
                        then
                        rm -rf $runScript
                fi
                touch $runScript
                chmod a+x $runScript

                echo "#BSUB -o $LOGDIR_TRAIN/runJobsStep3_${algo}_${i}.out" >> $runScript
                echo "#BSUB -e $LOGDIR_TRAIN/runJobsStep3_${algo}_${i}.err" >> $runScript
                echo "#BSUB -L /bin/bash" >> $runScript
                echo "cd $CMSDIR" >> $runScript
                echo "export SCRAM_ARCH=$SCRAM_ARCH" >> $runScript
                echo 'eval `scramv1 runtime -sh`' >> $runScript
                echo "cd $MAINDIR" >> $runScript


                echo "cmsRun step3_MVA_forJobs.py $line $i $algo $OUTPUTDIR_TRAIN" >> $runScript
                echo "bsub -R \" pool>30000\" -q 1nd < $runScript" >> $condorFile_train
        done
	let i++
done < $filelist_train

#Create batch job submission scripts for each ntuple for testing
i=0
while read line in filelist_test
        do
	for algo in "${ALGOS[@]}"
        do
          	runScript=$SCRIPTDIR_TEST/runJobsStep3_${algo}_${i}.sh
                if [ -e $runScript ]
                        then
                        rm -rf $runScript
                fi
                touch $runScript
                chmod a+x $runScript

                echo "#BSUB -o $LOGDIR_TEST/runJobsStep3_${algo}_${i}.out" >> $runScript
                echo "#BSUB -e $LOGDIR_TEST/runJobsStep3_${algo}_${i}.err" >> $runScript
                echo "#BSUB -L /bin/bash" >> $runScript
                echo "cd $CMSDIR" >> $runScript
                echo "export SCRAM_ARCH=$SCRAM_ARCH" >> $runScript
                echo 'eval `scramv1 runtime -sh`' >> $runScript
                echo "cd $MAINDIR" >> $runScript


                echo "cmsRun step3_MVA_forJobs.py $line $i $algo $OUTPUTDIR_TEST" >> $runScript
                echo "bsub -R \" pool>30000\" -q 1nd < $runScript" >> $condorFile_test
        done
	let i++
done < $filelist_test

