import pandas as pd
def calc_metrics(load_path, syntax_path, store_path):
    """calculate the data and domain dissimlarity seperatly as tabular value, best applied to the aggregated table.

    Args:
        load_path (str): path to load the data from
        store_path (str): path to store the data + filename
    """

    df_results = pd.read_csv(load_path, header=0)
    print(df_results.columns)

    df_syntax = pd.read_csv(syntax_path, header=0)
    print(df_syntax.columns)

    df_syntax = df_syntax[['Flows', 'model','ds','step']]
    df_syntax['Flows'] = df_syntax['Flows']/10000
    df_syntax.columns = ['error_flows', 'model','ds_1','sample_1']
    #df_syntax['sample_1'] = 'sample-' + df_syntax['sample_1'].astype(str)
    #df_syntax['sample_1'] = 'step-' + df_syntax['sample_1'].astype(str)
    df_results['sample_1'] = df_results['sample_1'].astype(str).str.replace('step-','')
    df_results['sample_1'] = df_results['sample_1'].astype(int)

    print(df_syntax)
    print(df_results)

    #print(df_syntax[['ds_1', 'model', 'sample_1']])
    #print(df_results[['ds_1', 'model', 'sample_1']])

    #df_results.join(df_syntax, on=['model','ds','step'])
    df_results = df_results.merge(df_syntax, how='inner', left_on=['ds_1', 'model', 'sample_1'], right_on=['ds_1', 'model', 'sample_1'])
    print(df_results)
    #quit()

    invert_cols = [ 'if_disc_tr-ds1_tst-ds2_fpr', #calc 1-metric for value where 0-1, 0 is better
                    'if_disc_tr-ds2_tst-ds1_fpr', 
                    'ocsvm_disc_tr-ds1_tst-ds2_fpr', #calc 1-metric for value where 0-1, 0 is better
                    'ocsvm_disc_tr-ds2_tst-ds1_fpr', 
                    'if_task_tr-ds1_tst-ds2_f1-weighted',
                    'if_task_tr-ds2_tst-ds1_f1-weighted',
                    'ocsvm_task_tr-ds1_tst-ds2_f1-weighted',
                    'ocsvm_task_tr-ds2_tst-ds1_f1-weighted',
                    'xgb_task_tr-ds1_tst-ds2_f1-weighted',
                    'xgb_task_tr-ds2_tst-ds1_f1-weighted',
    ]
    
    data_metrics = [ 'JS', 
                    'pearson',
                    'corr_ratio', 
                    'theils_u', 
                    'if_disc_tr-ds1_tst-ds2_fpr', #calc 1-metric for value where 0-1, 0 is better
                    'if_disc_tr-ds2_tst-ds1_fpr', 
                    'ocsvm_disc_tr-ds1_tst-ds2_fpr', #calc 1-metric for value where 0-1, 0 is better
                    'ocsvm_disc_tr-ds2_tst-ds1_fpr',  ]
                
    domain_metrics = [ 'if_task_tr-ds1_tst-ds2_f1-weighted',
                    'if_task_tr-ds2_tst-ds1_f1-weighted',
                    'ocsvm_task_tr-ds1_tst-ds2_f1-weighted',
                    'ocsvm_task_tr-ds2_tst-ds1_f1-weighted',
                    'xgb_task_tr-ds1_tst-ds2_f1-weighted',
                    'xgb_task_tr-ds2_tst-ds1_f1-weighted',
                    'error_flows']
                
    # -1 is the framework's "metric could not be computed" sentinel. It has to be dropped
    # *before* the inversion below, which would otherwise turn it into 2.0 -- a value outside
    # the [0,1] range of every real metric, silently dominating the mean it is averaged into.
    # This is not hypothetical: test_xgboost_task_raw is commented out (xgboost is not even a
    # dependency), so all six xgb_* columns are always -1, and two of them sit in
    # domain_metrics. Left alone they pushed the Domain Dissimilarity Score to 0.76 for both
    # a synthetic run and a real-vs-real run whose real task scores differed by 0.09.
    def usable(cols):
        return [c for c in cols if c in df_results.columns and not (df_results[c] == -1).all()]

    dropped = sorted(set(data_metrics + domain_metrics) - set(usable(data_metrics + domain_metrics)))
    if dropped:
        print('skipping metrics that are entirely the -1 error sentinel:', dropped)
    invert_cols = usable(invert_cols)
    data_metrics = usable(data_metrics)
    domain_metrics = usable(domain_metrics)

    #invert values
    df_results[invert_cols] = 1- df_results[invert_cols]
    df_results['Data Dissimilarity Score'] = df_results[data_metrics].mean(axis=1)
    df_results['Domain Dissimilarity Score'] = df_results[domain_metrics].mean(axis=1)

    df_results['Data Dissimilarity Score_std'] = df_results[data_metrics].std(axis=1)
    df_results['Domain Dissimilarity Score_std'] = df_results[domain_metrics].std(axis=1)
    print(df_results)
    df_results.to_csv(store_path, index=False, header=True)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load_path", required=True)
    parser.add_argument("-c", "--syntax_path", required=True)
    parser.add_argument("-s", '--store_path', required=True)

    args = parser.parse_args()
    load_path = args.load_path
    syntax_chk_path = args.syntax_path
    store_path = args.store_path
    
    calc_metrics(load_path, syntax_chk_path, store_path)