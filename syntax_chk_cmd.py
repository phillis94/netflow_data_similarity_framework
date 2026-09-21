import argparse
from src import syntax_check


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load_path", required=True)
    parser.add_argument("-s", '--store_path', required=True)
    parser.add_argument("-i", '--ds_id', required=True)
    parser.add_argument("-v", '--vocab_dir', default=None,
                        help="directory holding CIDDS ip_vocab.json / port_vocab.json. Given, "
                             "the check switches to the CIDDS bucketed alphabet and counts "
                             "semantic rather than format errors (see src/cidds.py). Omit for "
                             "the Queensland datasets.")


    args = parser.parse_args()
    load_path = args.load_path
    store_path = args.store_path
    dataset_id = args.ds_id

    syntax_check.remove_syntax_erros_files(load_path, store_path, dataset_id, args.vocab_dir)
