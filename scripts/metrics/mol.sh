#!/bin/bash

RESULTS_DIR=`realpath $1`
echo "Evaluating results from $RESULTS_DIR"

CBGBENCH_REPO=/home/zhangpd/Off_Target/UniMoMo_try2/evaluation/CBGBench
DATA_DIR=/home/zhangpd/Off_Target/UniMoMo_try2/datasets/molecule/CrossDocked/crossdocked_pocket10

DATA_DIR=`realpath $DATA_DIR`
echo "CrossDocked2020 located at $DATA_DIR"

cd ${CBGBENCH_REPO}/evaluate_scripts


# chemistry
python evaluate_chem_folder.py --base_result_path ${RESULTS_DIR} --base_pdb_path ${DATA_DIR}

echo "Evaluation of chemistry for ${RESULTS_DIR} completed."
echo "--------------------------------------------------"

python cal_chem_results.py --root_directory ${RESULTS_DIR} | tee ${RESULTS_DIR}/chem_results.txt

echo "Calculation of chemistry for ${RESULTS_DIR} completed."
echo "--------------------------------------------------"




# interaction
python evaluate_interact_folder.py --base_result_path ${RESULTS_DIR} --base_pdb_path ${DATA_DIR}

echo "Evaluation of interaction for ${RESULTS_DIR} completed."
echo "--------------------------------------------------"

python cal_intera_results.py --base_result_path ${RESULTS_DIR} | tee ${RESULTS_DIR}/interaction_results.txt

echo "Calculation of interaction for ${RESULTS_DIR} completed."
echo "--------------------------------------------------"


# substructure
python evaluate_substruct_folder.py --base_result_path ${RESULTS_DIR} --base_pdb_path ${DATA_DIR}

echo "Evaluation of substructure for ${RESULTS_DIR} completed."
echo "--------------------------------------------------"

python cal_sub_results.py --base_result_path ${RESULTS_DIR} | tee ${RESULTS_DIR}/substruct_results.txt

echo "Calculation of substructure for ${RESULTS_DIR} completed."
echo "--------------------------------------------------"


# geometry
python evaluate_geom_folder.py --base_result_path ${RESULTS_DIR} --base_pdb_path ${DATA_DIR}

echo "Evaluation of geometry for ${RESULTS_DIR} completed."
echo "--------------------------------------------------"

python cal_geom_results.py --base_result_path ${RESULTS_DIR} | tee ${RESULTS_DIR}/geom_results.txt

echo "Calculation of geometry for ${RESULTS_DIR} completed." 
echo "--------------------------------------------------"
