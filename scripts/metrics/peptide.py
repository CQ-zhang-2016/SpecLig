#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Tuple
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import statistics
import warnings
import difflib
import tempfile

warnings.filterwarnings("ignore")

import ray
import numpy as np
from scipy.stats import spearmanr

from data.bioparse.parser.pdb_to_complex import pdb_to_complex as pdb_to_complex_raw
from data.bioparse import VOCAB, Block, const
from data.bioparse.utils import recur_index
from evaluation import diversity
from evaluation.rmsd import compute_rmsd
from evaluation.dockq import dockq
from utils.random_seed import setup_seed
from evaluation.seq import align_sequences
# from evaluation.openmm_relaxer import ForceFieldMinimizer
# from evaluation.energy import pyrosetta_interface_energy
from evaluation.clash import eval_pdb_clash
from evaluation.dihedrals import jsd_angle_profile, dihedral_distribution
from utils.logger import print_log


def pdb_to_complex(pdb_path, selected_chains):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_file:
        temp_pdb_path = tmp_file.name
        
        # Open the input PDB and write lines excluding CONECT records
        with open(pdb_path, 'r') as infile, open(temp_pdb_path, 'w') as outfile:
            for line in infile:
                if not line.startswith("CONECT") and not line.startswith('HETATM'):
                    outfile.write(line)
    res = pdb_to_complex_raw(temp_pdb_path, selected_chains)

    # Delete the temporary file
    os.remove(temp_pdb_path)

    return res


def _get_ref_pdb(_id, root_dir):
    return os.path.join(root_dir, 'references', f'{_id}_ref.pdb')


def _get_gen_pdb(_id, number, root_dir, use_rosetta=False):
    suffix = '_rosetta' if use_rosetta else ''
    return os.path.join(root_dir, 'candidates', _id, f'{number}{suffix}.pdb')


def _mean(vals):
    return sum(vals) / len(vals)


def subprocess_run_dG(pdb, rec_chains, lig_chains, relax, relax_first=None, relax_last=None, relax_save_pdb=None):

    args = {
        'pdb_path': os.path.abspath(pdb),
        'receptor_chains': rec_chains,
        'ligand_chains': lig_chains,
        'relax': relax,
        'relax_save_pdb': None if relax_save_pdb is None else os.path.abspath(relax_save_pdb)
    }
    if relax_first is not None and relax_last is not None:
        args['relax_opt'] = {
            'flexible_residue_first': relax_first,
            'flexible_residue_last': relax_last
        }
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation')
    cmd = f"cd {root_dir}; python energy.py '{json.dumps(args)}'"
    p = os.popen(cmd)
    text = p.read()
    p.close()

    dG = text.strip().split('\n')[-1]
    try: dG = float(dG)
    except ValueError:
        print_log(f'pdb {json.dumps(args)} calculating rosetta energy failed, reason: \n{text}', level='WARN')
        dG = 0

    return dG


def robust_dG(pdb, rec_chains, lig_chains, gen_block_idx=None, n=3, relax=True, relax_save_pdb=None):
    energies = []
    def _to_rosetta_res_id(idx):
        res_id = [idx[0], idx[1][0]]
        if idx[1][1] != '': res_id.append(idx[1][1])
        return res_id
    
    if gen_block_idx is not None:
        res_first = _to_rosetta_res_id(gen_block_idx[0])
        res_last = _to_rosetta_res_id(gen_block_idx[-1])
    else: res_first, res_last = None, None

    min_dG = None

    for _ in range(n):
        # energies.append(pyrosetta_interface_energy(pdb, rec_chains, lig_chains, relax=relax))
        energies.append(subprocess_run_dG(pdb, rec_chains, lig_chains, relax, res_first, res_last, relax_save_pdb))

        # save pose with the best energy
        if relax_save_pdb is not None:
            if (min_dG is None or energies[-1] <= min_dG) and os.path.exists(relax_save_pdb): # update best pose
                with open(relax_save_pdb, 'r') as fin: min_pose_text = fin.read()
                min_dG = energies[-1]
            elif min_dG is not None and energies[-1] > min_dG: # recover original pose
                with open(relax_save_pdb, 'w') as fout: fout.write(min_pose_text)

    return _mean(energies)


def extract_paired_coords(gen_blocks: List[Block], ref_blocks: List[Block], ca_only: bool):
    # Full atom RMSD
    gen_x, ref_x = [], []
    for gen_block, ref_block in zip(gen_blocks, ref_blocks):
        allow_atoms = { atom.name for atom in ref_block }
        if ca_only: allow_atoms = { 'CA': 1 } if 'CA' in allow_atoms else {}
        elif gen_block.name != ref_block.name: # only backbone atoms
            allow_atoms = { atom: 1 for atom in const.backbone_atoms if atom in allow_atoms }
    
        gen_atoms, ref_atoms = {}, {}
        for atom in gen_block: gen_atoms[atom.name] = atom.get_coord()
        for atom in ref_block: ref_atoms[atom.name] = atom.get_coord()
        for ref_atom in allow_atoms:
            if ref_atom in gen_atoms:
                ref_x.append(ref_atoms[ref_atom])
                gen_x.append(gen_atoms[ref_atom])
    return np.array(gen_x), np.array(ref_x)


@dataclass
class Task:
    root_dir: str
    id: str
    number: int
    target_chains_ids: List[str]
    ligand_chains_ids: List[str]
    gen_block_idx: List[tuple]
    pmetric: float
    is_antibody: bool
    ref_metrics: dict = None
    metrics: dict = None
    seq: str = None
    ca_coords: List[tuple] = None


@ray.remote(num_cpus=1) # dG requires larger RAM
def run_ref_metrics(task: Task):

    if task.is_antibody:
        ref_pdb = _get_ref_pdb(os.path.dirname(task.id), task.root_dir)
    else:
        ref_pdb = _get_ref_pdb(task.id, task.root_dir)
    
    task.ref_metrics = {}
    # set reference dG
    ref_dG_relax = robust_dG(
        ref_pdb, task.target_chains_ids,
        task.ligand_chains_ids,
        task.gen_block_idx if task.is_antibody else None,
        relax_save_pdb=ref_pdb.rstrip('.pdb') + '_rosetta.pdb')
    task.ref_metrics['ref_dG'] = round(ref_dG_relax, 2)

    # ref clash
    clash_inner, clash_outer = eval_pdb_clash(ref_pdb, task.target_chains_ids, task.ligand_chains_ids)
    task.ref_metrics['ref_Clash_inner'] = round(clash_inner, 4)
    task.ref_metrics['ref_Clash_outer'] = round(clash_outer, 4)

    return task


@ray.remote(num_cpus=1)
def run_basic_metrics(task: Task):

    task.metrics = {}

    # get complex
    gen_pdb = _get_gen_pdb(task.id, task.number, task.root_dir)
    try:
        gen_cplx = pdb_to_complex(gen_pdb, task.target_chains_ids + task.ligand_chains_ids)
    except Exception as e:
        print_log(f'{gen_pdb} read in complex error! {task}', level='ERROR')
        task.metrics = defaultdict(lambda: float('nan'))
        return task
    if task.is_antibody:
        ref_pdb = _get_ref_pdb(os.path.dirname(task.id), task.root_dir)
    else:
        ref_pdb = _get_ref_pdb(task.id, task.root_dir)
    ref_cplx = pdb_to_complex(ref_pdb, task.target_chains_ids + task.ligand_chains_ids)

    # get generated blocks
    gen_block_idx = [(idx[0], tuple(idx[1])) for idx in task.gen_block_idx]
    try:
        gen_blocks = [recur_index(gen_cplx, block_id) for block_id in gen_block_idx]
    except Exception:
        print_log(f'{gen_pdb} generation part missing', level='ERROR')
    ref_blocks = [recur_index(ref_cplx, block_id) for block_id in gen_block_idx]
    
    # get sequence
    gen_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in gen_blocks])
    ref_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in ref_blocks]) 

    # set sequence
    task.seq = gen_seq

    # aar
    assert len(gen_seq) == len(ref_seq)
    _, seq_id = align_sequences(gen_seq, ref_seq)
    task.metrics['AAR'] = round(seq_id, 4)

    # CA coordinates
    gen_ca_x, ref_ca_x = extract_paired_coords(gen_blocks, ref_blocks, ca_only=True)

    # set CA coordinates
    task.ca_coords = gen_ca_x

    # CA RMSD
    c_rmsd = compute_rmsd(gen_ca_x, ref_ca_x, need_align=False)
    task.metrics['C_RMSD(CA)'] = round(c_rmsd, 2)
    l_rmsd = compute_rmsd(gen_ca_x, ref_ca_x, need_align=True)
    task.metrics['L_RMSD(CA)'] = round(l_rmsd, 2)
    
    # DockQ
    try:
        dockq_score = round(dockq(gen_pdb, ref_pdb, task.target_chains_ids, task.ligand_chains_ids), 3)
    except (ValueError, KeyError):
        dockq_score = 0
    task.metrics['DockQ'] = dockq_score

    # clash
    try:
        clash_inner, clash_outer = eval_pdb_clash(gen_pdb, task.target_chains_ids, task.ligand_chains_ids)
    except Exception:
        clash_inner, clash_outer = float('nan'), float('nan')
    task.metrics['Clash_inner'] = round(clash_inner, 4)
    task.metrics['Clash_outer'] = round(clash_outer, 4)

    return task


@ray.remote(num_cpus=1)
def run_dG(task: Task):

    gen_pdb = _get_gen_pdb(task.id, task.number, task.root_dir)

    # Rosetta dG relax
    dG = robust_dG(
        gen_pdb, task.target_chains_ids, task.ligand_chains_ids,
        task.gen_block_idx if task.is_antibody else None,
        relax_save_pdb=gen_pdb.rstrip('.pdb') + '_rosetta.pdb')
    dG = round(dG, 2)
    task.metrics['dG'] = dG
    task.metrics['ddG'] = dG - task.ref_metrics['ref_dG']
    task.metrics['ddG <= 0 (IMP)'] = dG <= task.ref_metrics['ref_dG']

    return task


@ray.remote(num_cpus=1)
def aggregate_metrics(tasks: List[Task]):
    pmets = [task.pmetric for task in tasks]
    indexes = list(range(len(tasks)))

    aggr_results = {}

    # aggregation
    def nan_filter(values):
        return [val for val in values if not np.isnan(val)]

    for name in tasks[0].metrics:
        vals = [task.metrics[name] for task in tasks]
        vals_not_none = nan_filter(vals)
        if len(vals_not_none) < 3: corr = 0 # at least 3 not being nan
        else: corr = spearmanr(vals, pmets, nan_policy='omit').statistic
        if np.isnan(corr): corr = 0
        corr = round(corr, 3)
        if len(vals_not_none) == 0: vals_not_none = [float('nan')] # all failed
        aggr_res = {
            'max': max(vals_not_none),
            'min': min(vals_not_none),
            'mean': sum(vals_not_none) / len(vals_not_none),
            'random': vals_not_none[0],
            'max*': vals[(max if corr > 0 else min)(indexes, key=lambda i: pmets[i])],
            'min*': vals[(min if corr > 0 else max)(indexes, key=lambda i: pmets[i])],
            'pmet_corr': corr,
            'individual': vals,
            'individual_pmet': pmets
        }
        aggr_results[name] = aggr_res

    if len(tasks) > 1:
        seqs = [task.seq for task in tasks if (task.seq is not None and task.ca_coords is not None)]
        ca_coords = [task.ca_coords for task in tasks if (task.seq is not None and task.ca_coords is not None)]
        if len(seqs) > 1:
            seq_div, struct_div, co_div, _ = diversity.diversity(seqs, np.array(ca_coords))
        else:
            seq_div, struct_div, co_div = 0, 0, 0
        aggr_results['Sequence Diversity'] = round(seq_div, 3)
        aggr_results['Struct Diversity'] = round(struct_div, 3)
        aggr_results['Codesign Diversity'] = round(co_div, 3)
    
    return aggr_results


@ray.remote
def pipeline_eval(input: Tuple[List[dict], bool]):
    items, is_antibody = input
    item = items[0]
    ref_task = Task(
        root_dir = item['root_dir'],
        id = item['id'],
        number = item['n'],
        target_chains_ids = item['target_chains_ids'],
        ligand_chains_ids = item['ligand_chains_ids'],
        gen_block_idx = item['gen_block_idx'],
        pmetric = round(item['pmetric'], 4),
        is_antibody = is_antibody
    )

    ref_funcs = [
        run_ref_metrics,
    ]
    for fn in ref_funcs:
        ref_task = fn.remote(ref_task)
    ref_task = ray.get(ref_task)

    # pipeline tasks
    funcs = [
        run_basic_metrics,
        run_dG,
    ]

    # get all tasks to run
    finished_tasks = []
    for item in items:
        task = Task(
            root_dir = item['root_dir'],
            id = item['id'],
            number = item['n'],
            target_chains_ids = item['target_chains_ids'],
            ligand_chains_ids = item['ligand_chains_ids'],
            gen_block_idx = item['gen_block_idx'],
            pmetric = round(item['pmetric'], 4),
            is_antibody = is_antibody,
            ref_metrics = ref_task.ref_metrics
        )
        for fn in funcs:
            task = fn.remote(task)
        finished_tasks.append(ray.get(task))

    aggr_res = aggregate_metrics.remote(finished_tasks)

    return (ref_task.id, ref_task.ref_metrics, ray.get(aggr_res))


def main(args):
    root_dir = os.path.dirname(args.results)
    # load results
    with open(args.results, 'r') as fin:
        lines = fin.read().strip().split('\n')
    id2items = {}
    for line in lines:
        item = json.loads(line)
        item['root_dir'] = root_dir
        _id = item['id']
        if _id not in id2items: id2items[_id] = []
        id2items[_id].append(item)
    ids = list(id2items.keys())
    
    eval_results_path = os.path.join(os.path.dirname(args.results), f'eval_report_{args.log_suffix}.jsonl')

    fout = open(eval_results_path, 'w')

    ray.init(num_cpus=args.num_workers)
    futures = [pipeline_eval.remote((item, args.antibody)) for item in id2items.values()]
    ref_metrics, metrics = [], []
    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            _id, ref, gen = ray.get(done_id)
            ref_metrics.append(ref)
            metrics.append(gen)
            
            record = { 'id': _id }
            record.update(ref)
            record.update(gen)

            fout.write(json.dumps(record) + '\n')
            fout.flush()

            print_log(f'Finished {len(metrics)}/{len(id2items)}, {record}')

    fout.close()
    ray.shutdown()
    
    log_file = open(os.path.join(os.path.dirname(args.results), f'eval_final_{args.log_suffix}.log'), 'w')
    def print_and_log(s, **kwargs):
        print(s, **kwargs)
        log_file.write(s + '\n')

    # calculate reference
    for name in ref_metrics[0]:
        vals = [m[name] for m in ref_metrics]
        print_and_log(f'reference {name}: mean {sum(vals) / len(vals)}, median {statistics.median(vals)}')

    # individual level results
    print_and_log('Point-wise evaluation results:')
    for name in metrics[0]:
        vals = [item[name] for item in metrics]
        if isinstance(vals[0], dict):
            if (('RMSD' in name) or ('dG' in name) or ('ddG' in name)) and '<=' not in name:
                aggr = 'min'
            elif 'Clash' in name: # clash ratio. Average performance should be measured
                aggr = 'mean'
            else:
                aggr = 'max'
            all_vals = [val[aggr] for val in vals]
            not_nan_idx = [i for i in range(len(all_vals)) if not np.isnan(all_vals[i])]
            aggr_vals = [val for val in all_vals if not np.isnan(val)]
            if '>=' in name or '<=' in name:  # percentage
                print_and_log(f'{name}: {sum(aggr_vals) / len(aggr_vals)}')
            else:
                if 'RMSD' in name:  # use median as conventions
                    print_and_log(f'{name} (median): {statistics.median(aggr_vals)}')
                else:
                    print_and_log(f'{name} (mean): {sum(aggr_vals) / len(aggr_vals)}')
                if len(not_nan_idx) == 0:
                    print_log('all values are nan!', level='WARN')
                    lowest_i, highest_i = 0, 0
                else:
                    lowest_i = min(not_nan_idx, key=lambda i: all_vals[i])
                    highest_i = max(not_nan_idx, key=lambda i: all_vals[i])
                print_and_log(f'\tlowest: {all_vals[lowest_i]}, id: {ids[lowest_i]}', end='')
                print_and_log(f'\thighest: {all_vals[highest_i]}, id: {ids[highest_i]}')
                corrs = [val['pmet_corr'] for val in vals if val['pmet_corr'] != 0]
                if len(corrs) == 0: corrs = [0]
                print_and_log(f'\tcorrelation with flow matching likelihood: {sum(corrs) / len(corrs)}')
            
        else:
            print_and_log(f'{name} (mean): {sum(vals) / len(vals)}')
            lowest_i = min([i for i in range(len(vals))], key=lambda i: vals[i])
            highest_i = max([i for i in range(len(vals))], key=lambda i: vals[i])
            print_and_log(f'\tlowest: {vals[lowest_i]}, id: {ids[lowest_i]}')
            print_and_log(f'\thighest: {vals[highest_i]}, id: {ids[highest_i]}')
    
    # get distribution of dihedral angles
    print_and_log('\n')
    pdbs, selected_residues = [], []
    for _id in id2items:
        for item in id2items[_id]:
            pdb = _get_gen_pdb(item['id'], item['n'], root_dir)
            pdbs.append(pdb)
            selected_residues.append(item['gen_block_idx'])
    print_and_log('JSD of dihedral angles:')
    dist = dihedral_distribution(pdbs, all_selected_residues=selected_residues, num_cpus=args.num_workers)
    profile = jsd_angle_profile(dist, 'antibody' if args.antibody else 'peptide')
    for key in profile:
        if key not in ['backbone_overall', 'sidechain_overall']: print_and_log(f'{key}: {profile[key]}')
    print_and_log('\n')
    print_and_log(f'backbone_overall (JSD_bb): {profile["backbone_overall"]}')
    print_and_log(f'sidechain_overall (JSD_sc): {profile["sidechain_overall"]}')
    print_and_log('\n')

    log_file.close()


def parse():
    parser = argparse.ArgumentParser(description='calculate metrics')
    parser.add_argument('--results', type=str, required=True, help='Path to test set')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use')
    parser.add_argument('--antibody', action='store_true', help='Special reference id and only relax CDR regions')
    parser.add_argument('--log_suffix', type=str, default='', help='Suffix of the log file (eval_final.log)')

    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(0)
    main(parse())
