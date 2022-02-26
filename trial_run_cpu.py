import os
import time
import tqdm
import filecmp
import logging

import numpy as np
import pandas as pd
from argparse import ArgumentParser


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def calc_dist_cpu(vegt, cond):
    # start_time = time.time()
    min_values = []
    
    for x in vegt:
        dist = np.min(np.sqrt(np.sum(np.square(x - cond),axis=1)))
        min_values.append(dist)

    min_dist = np.expand_dims(np.array(min_values), axis=1)
    all_data = np.append(vegt, min_dist, axis=1)
    
    # epoch_mins, epoch_secs = epoch_time(start_time, time.time())
    # logging.info(f'{vegt.shape},{cond.shape},{epoch_mins}m {epoch_secs}s') 

    np.savetxt('results/all_data_cpu.txt', all_data, delimiter=',', fmt='%f')
    return all_data

def main(args):
    
    limit = args.no
    
    path = "data/"
    all_files = os.listdir(path)    
    csv_files = list(filter(lambda f: f.startswith('object') and f.endswith('.csv'), all_files))    
    
    logging.basicConfig(filename="results/"+"cpu_runtime.log", level=logging.DEBUG)
    start_time_ = time.time()
    
    for filename in tqdm.tqdm(csv_files[:limit]):
        
        df = pd.read_csv(path+filename, names = ['x' ,'y', 'z', 'atom'])

        #separate vegetation and conductor
        points_vegt = df.loc[(df['atom'] == 'C') | (df['atom'] == 'H')]
        points_cond = df.loc[df['atom'] == 'O']

        vegt = points_vegt[['x','y','z']].to_numpy()
        cond = points_cond[['x','y','z']].to_numpy()

        with open("results/cpu_"+str(limit)+"_metadata.txt", "a") as file_object:
            line = "{},{},{},{}\n".format(filename, df.shape[0], vegt.shape[0], cond.shape[0])
            file_object.write(line)    

        vegt_clip = vegt
        cond_clip = cond

        calc_dist_cpu(vegt_clip, cond_clip)
    
    epoch_mins, epoch_secs = epoch_time(start_time_, time.time())
    logging.info(f'Files Processed: {limit} in Runtime: {epoch_mins}m {epoch_secs}s') 
    logging.shutdown()

    return

if __name__ == "__main__":
    parser = ArgumentParser()

    subparser = parser.add_subparsers(dest='command')
    calculate = subparser.add_parser('calculate')
    validate = subparser.add_parser('validate')

    validate.add_argument("--file1", type=str,
                        help="file 1 to compare ", required=True)       
    validate.add_argument("--file2", type=str,
                        help="file 2 to compare ", required=True)                                                                                         
    calculate.add_argument("--no", default=100, type=int,
                        help="number of files to process", required=True)


    args = parser.parse_args()

    if  args.command == 'calculate':
        main(args)
    elif args.command == 'validate':
        print("Files contain same values: ",filecmp.cmp(args.file1, args.file2))    

