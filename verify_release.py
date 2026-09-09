"""Verify packaged artifacts and saved predictions; does not retrain models."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                f'File integrity mismatch: {name}')
    for name, expected in manifest['original_artifact_sha256'].items():
        if name != 'reference_results/README_results.md':
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                    f'Original artifact mismatch: {name}')
    protocol = json.loads((ROOT / 'reference_results/protocol.json').read_text(encoding='utf-8'))
    require(hashlib.sha256((ROOT / 'data.xlsx').read_bytes()).hexdigest() == protocol['data_sha256'],
            'Data hash does not match the executed protocol')
    spec = importlib.util.spec_from_file_location('wb_piml_release', ROOT / 'WB-PIML.py')
    analysis = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analysis
    spec.loader.exec_module(analysis)
    data = analysis.load_data(ROOT / 'data.xlsx')
    require((len(data), int(data.is_exact.sum()), int(data.is_runout.sum())) == (223, 158, 65),
            'Unexpected exact/runout counts')
    require((data.source_id.nunique(), data.campaign_id.nunique()) == (18, 17),
            'Unexpected source/campaign counts')
    require((len(analysis.HYBRID_CANDIDATES), len(analysis.CONTROL_CANDIDATES)) == (54, 64),
            'Unexpected primary candidate counts')
    require(Counter(c['model'] for c in analysis.SIMPLE_HYBRID_CANDIDATES) ==
            {'WB-Residual': 18, 'Basquin-Residual': 6}, 'Unexpected simple-trunk candidates')

    with pd.ExcelFile(ROOT / 'reference_results/WB_PIML_results.xlsx') as workbook:
        require(len(workbook.sheet_names) == 57, 'Unexpected worksheet count')
        assignment = pd.read_excel(workbook, 'Outer_Fold_Assignment')
        require(assignment.row_id.is_unique and len(assignment) == 223, 'Invalid fold assignment rows')
        require(assignment.groupby('campaign_id').outer_fold.nunique().eq(1).all(),
                'Campaign crosses primary test folds')
        _, fold_ids = analysis.grouped_folds(data, 5, analysis.SEED)
        saved = assignment.set_index('row_id').outer_fold.reindex(data.row_id).to_numpy()
        np.testing.assert_array_equal(fold_ids, saved)

        predictions = pd.read_excel(workbook, 'Exact_Outer_Pred')
        reported = pd.read_excel(workbook, 'Primary_Campaign_Metrics').set_index('model')
        for model, group in predictions.groupby('model'):
            require(len(group) == 158 and group.row_id.is_unique and group.is_exact.all(),
                    f'Invalid primary prediction rows: {model}')
            recomputed = analysis.sb_metrics([g.reset_index(drop=True) for _, g in group.groupby('campaign_id')])
            for metric, value in recomputed.items():
                np.testing.assert_allclose(value, reported.loc[model, metric], rtol=1e-10, atol=1e-10)

        probability = pd.read_excel(workbook, 'Prob_Predictions')
        losses = analysis.distribution_nll(probability.prob_family, probability.logN,
                    probability.prob_location, probability.prob_scale, probability.is_exact)
        np.testing.assert_allclose(losses, probability.row_nll, rtol=1e-10, atol=1e-10)
        for level in (80, 90):
            tail = (1 - level / 100) / 2
            for label, quantile in [('lower', tail), ('upper', 1-tail)]:
                calculated = analysis.distribution_quantile(probability.prob_family,
                    probability.prob_location, probability.prob_scale, quantile)
                np.testing.assert_allclose(calculated, probability[f'prob_{label}{level}'],
                                           rtol=1e-10, atol=1e-10)
        metrics = pd.read_excel(workbook, 'Prob_Metrics')
        metrics = metrics.loc[metrics.aggregation_level.eq('overall')].set_index(['scenario', 'model'])
        for key, group in probability.groupby(['scenario', 'model']):
            values = analysis._prob_summary(group)
            for metric in ('SB_joint_censored_NLL', 'SB_exact_NLL', 'SB_runout_NLL', 'SB_RMSE'):
                np.testing.assert_allclose(values[metric], metrics.loc[key, metric], rtol=1e-10, atol=1e-10)
        trunks = pd.read_excel(workbook, 'Trunk_Metrics')
        for row in trunks.itertuples():
            for metric in ('SB_RMSE', 'SB_joint_censored_NLL', 'SB_exact_NLL', 'SB_runout_NLL'):
                np.testing.assert_allclose(getattr(row, metric), metrics.loc[(row.scenario,row.model),metric],
                                           rtol=1e-10, atol=1e-10)
    print(f'PASS: {len(manifest["sha256"])} file hashes; 223 data records; 54/64/18/6 candidates; '
          f'5 campaign-disjoint folds; 57 worksheets; {len(reported)} primary model summaries; '
          f'{len(probability)} distribution predictions and {len(metrics)} scenario/model summaries.')
    print('No model training was rerun. These are integrity and saved-result consistency checks.')


if __name__ == '__main__':
    main()
