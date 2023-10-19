import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import roc_auc_score, matthews_corrcoef, ndcg_score

COPYPASTE_NAMES_DDG = ['ndcg_-0.5_30', 'prec_-0.5_30', 'spear', 'rmse', 'cls_mcc', 'cls_auroc']
## ndcg_-0.5_30: nDCG considering the top 30 predicted mutations where a mutation is considered stabilizing if ground truth ddg < -0.5

def eval_ddg(
    df: pd.DataFrame,
    preds: dict,
    max_dets: list = [30],
    max_ddg: float = -0.5,
):
    """
    Args:
        df: DataFrame with pdb_id, mut_info, gt ddg
        preds: dict of {pdb_id: {'mutations': [], 'scores': []}}
            mutations formatted f'{cur_aa}{seq_pos}{mut_aa}' (indexing from 1)
        max_dets: max number of predicted mutations to consider
        max_ddg: max ddg value to be positive stabilizing mutation
    Returns:
        metrics: dict of all metrics
        metrics_det: DataFrame of detection metrics
        metrics_det_pdb: DataFrame of detection metrics per pdb
        copypaste: string
        df: merged gt and predictions
    """
    df = _preprocess_gt_pr(df, preds)
    metrics = compute_cls_reg_metrics(-df.scores, df.ddg)

    # det metrics
    metrics_det, metrics_det_pdb = compute_detection_metrics(df, max_dets, max_ddg)
    metrics.update({
        f'{row.variable}_{row.max_ddg}_{row.max_det}': row.value
        for _, row in
        metrics_det.melt(id_vars=['max_ddg', 'max_det']).iterrows()
    })  # melt returns rows of (max_ddg, max_det, variable, value)

    copypaste = ','.join([f'{metrics[name]:.02f}' for name in COPYPASTE_NAMES_DDG])
    return metrics, metrics_det, metrics_det_pdb, copypaste, df

def _preprocess_gt_pr(df, preds):
    """ Clean the GT, then merge the predictions into the GT dataframe. """
    # clean up ground truth
    df = df[~df.mut_info.isna()]
    df = df[~df.ddg.isna()]
    df['mut_info'] = df['mut_info'].str.upper()
    df = df.groupby(['pdb_id', 'mut_info'], as_index=False).median(numeric_only=True)  # multiple ddg for same mutation
    if 'ddg' in df.columns and np.mean(df.ddg < 0) > 0.5:
        print('WARNING: more stabilizing than destabilizing, flipping sign')
        df['ddg'] = -1 * df['ddg']

    # check preds have >0 pdbs in ground truth
    pdbs_eval = list(preds.keys())
    gt_pdb_ids = list(df.pdb_id.unique())
    if len(pdbs_eval) == 0:
        return {}
    assert set(pdbs_eval).issubset(set(gt_pdb_ids)), f'preds for {set(pdbs_eval) - set(gt_pdb_ids)} not in gt'

    # merge preds into ground truth df
    preds_df = []
    for pdb_id, preds_pdb in preds.items():
        preds_pdb_df = pd.DataFrame.from_dict(preds_pdb)
        preds_pdb_df['pdb_id'] = pdb_id
        preds_df.append(preds_pdb_df)
    preds_df = pd.concat(preds_df).rename({'mutations': 'mut_info'}, axis=1)
    preds_df['mut_info'] = preds_df['mut_info'].str.upper()
    df = df.merge(preds_df, on=['pdb_id', 'mut_info'], how='left')
    return df

def compute_detection_metrics(
    df: pd.DataFrame,
    max_dets: list = [30],
    max_ddg: float = -0.5,
):
    # iterate over pdbs, max_dets
    metrics_pdb = []
    for pdb_id in df.pdb_id.unique():
        # extract pdb-specific values
        df_pdb = df[df.pdb_id == pdb_id].sort_values('scores', ascending=False)
        scores = df_pdb.scores.to_numpy()
        ddg = df_pdb.ddg.to_numpy()
        nddg = np.maximum(-ddg, 0)

        df_pdb_stbl = df_pdb[ddg <= max_ddg]
        muts_sorted = df_pdb.mut_info.to_list()
        if len(df_pdb) <= 1 or len(df_pdb_stbl) == 0:
            continue

        for max_det in sorted(max_dets):
            metrics_ = {
                'pdb_id': pdb_id,
                'max_det': max_det,
                'max_ddg': max_ddg,
                'n_tot_muts': len(df_pdb),
                'ndcg': ndcg_score(nddg[None], scores[None], k=max_det),
                **compute_precision(df_pdb_stbl, muts_sorted[:max_det])
            }
            metrics_pdb.append(metrics_)
    metrics_pdb = pd.DataFrame(metrics_pdb)
    assert len(metrics_pdb) > 0, 'no pdbs evaluated'

    # summarize
    summary = metrics_pdb.groupby(['max_ddg', 'max_det'], as_index=False).mean(numeric_only=True)
    counts = metrics_pdb.groupby(['max_ddg', 'max_det'], as_index=False).pdb_id.count()
    summary = pd.merge(counts, summary, on=['max_ddg', 'max_det'])
    return summary, metrics_pdb

def compute_precision(gt_pdb, pr_muts_sorted):
    """
    gt_pdb: DataFrame with pdb_id, mut_info, gt ddg with ONLY ddg < threshold
    pr_muts_sorted: list of mutations sorted by score already filtered to max_det
    """
    assert len(gt_pdb.pdb_id.unique()) == 1, f'more than 1 pdb {gt_pdb.pdb_id.unique()}'
    metrics = {}
    gt_mut = gt_pdb.mut_info.str.upper().to_numpy()
    is_tp = np.array([mut in gt_mut for mut in pr_muts_sorted])
    metrics['is_tp'] = is_tp
    metrics['prec'] = is_tp.mean()
    return metrics

def compute_cls_reg_metrics(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)

    # metrics (cls)
    correct = (np.array(predictions) * np.array(labels)) > 0
    positive = np.array(labels) < 0
    acc = correct.mean()
    recall = (correct * positive).sum() / (positive.sum() + np.spacing(1))
    precision = (correct * positive).sum() / (predictions < 0).sum()
    if len(set(positive)) == 1:
        print('only one class, auroc/mcc are not well defined')
        auroc = -1
        mcc = -1
    else:
        auroc = roc_auc_score(positive, -predictions)
        mcc = matthews_corrcoef(positive, predictions < 0)

    # metrics (reg)
    mae = np.mean(np.abs(labels - predictions))
    rmse = np.sqrt(np.mean((labels - predictions) ** 2))
    pears = stats.pearsonr(labels, predictions)[0]
    spear = stats.spearmanr(labels, predictions)[0]

    return {
        'n_eval_muts': len(labels),
        'cls_mcc': mcc,
        'cls_auroc': auroc,
        'cls_acc': acc,
        'cls_recall': recall,
        'cls_precision': precision,
        'mae': mae,
        'rmse': rmse,
        'pears': pears,
        'spear': spear,
    }
