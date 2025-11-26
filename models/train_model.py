import os, argparse, time

from scripts.model_pipeline import run_train
from scripts.analysis_results import plot_training_curves, plot_roc

def get_parsed_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--fasta_path', type=str, dest='fasta_path', 
                        required=True, help='The path to the fasta dataset.')
    parser.add_argument('-l', '--label_path', type=str, dest='label_path', 
                        required=True, help='The path to the label (feature) file.')
    parser.add_argument('-o', '--output_dir', type=str, dest='output_dir', 
                        default='vgp_model_data_tpase', help='Directory to save model data.')
    parser.add_argument('-b', '--batch_size', type=int, dest='batch_size',
                        default=8, help='Batch size for data preparation.')
    parser.add_argument('-w', '--num_workers', type=int, dest='num_workers',
                        default=4, help='Number of workers for data loading.')
    parser.add_argument('-e', '--epochs', type=int, dest='epochs',
                        default=100, help='Number of training epochs.')
    parser.add_argument('-d', '--device', type=str, dest='device',
                        default=None, help='Device used for torch.')
    parser.add_argument('-p', '--patience', type=int, dest='patience',
                        default=20, help='Patience for early stopping.')
    parser.add_argument('-s', '--subset_size', type=int, dest='subset_size',
                        default=None, help='Size of subset to use for training. If None, use full dataset.')
    parser.add_argument('-t', '--trial', type=bool, dest='trial', 
                        default=False, help='If set, runs a trial training without saving the result.')
    parser.add_argument('-a', '--analysis', type=bool, dest='analysis', 
                        default=True, help='If set, plot curves to analysis the simulation. ')
    
    args = parser.parse_args()
    return args

def main():
    
    args = get_parsed_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    results = run_train(
        fasta_path=args.fasta_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        device=args.device,
        patience=args.patience,
        subset_size=args.subset_size,
        trial=args.trial
    )
    
    if (not args.trial) and args.analysis: 
        history = results["history"]
        plot_training_curves(history)
        
        roc_data = results["roc"]
        plot_roc(roc_data["labels"], roc_data["scores"])
    
    return results

if __name__ == '__main__':
    
    st = time.time()
    main()
    et = time.time()
    print(f"Total time elapsed: {et - st:.2f} seconds.")
    
    
