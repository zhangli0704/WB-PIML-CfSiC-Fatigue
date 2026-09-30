"""Verify source, reference artifacts, saved scores and optional selected refits.

Default checks do not train models. --refit-selected refits the five selected
outer-fold pipelines and their calibration folds, plus the frozen final fit;
it does not repeat the complete candidate search or all sensitivity refits.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import gzip
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
from run import ROOT, canonical_source, load_analysis


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_saved(analysis, data, reference):
    workbook = reference / 'primary/WB_PIML_results.xlsx'
    analysis.single_check_primary(workbook)
    with pd.ExcelFile(workbook) as book:
        assignment = pd.read_excel(book, 'Outer_Fold_Assignment')
        require(assignment.row_id.is_unique and len(assignment) == 223, 'Invalid fold rows')
        require(assignment.groupby('campaign_id').outer_fold.nunique().eq(1).all(), 'Campaign crosses outer folds')
        _, folds = analysis.grouped_folds(data, 5, analysis.SEED)
        np.testing.assert_array_equal(folds, assignment.set_index('row_id').outer_fold.reindex(data.row_id))
        predictions = pd.read_excel(book, 'Exact_Outer_Pred')
        reported = pd.read_excel(book, 'Primary_Campaign_Metrics').set_index('model')
        for model, group in predictions.groupby('model'):
            require(len(group) == 158 and group.row_id.is_unique and group.is_exact.all(), 'Invalid exact predictions: '+model)
            metrics = analysis.sb_metrics([g.reset_index(drop=True) for _, g in group.groupby('campaign_id')])
            for metric, value in metrics.items():
                np.testing.assert_allclose(value, reported.loc[model, metric], rtol=1e-10, atol=1e-10)
        probability = pd.read_excel(book, 'Prob_Predictions')
        losses = analysis.distribution_nll(probability.prob_family, probability.logN, probability.prob_location, probability.prob_scale, probability.is_exact)
        np.testing.assert_allclose(losses, probability.row_nll, rtol=1e-10, atol=1e-10)
        for level in (80, 90):
            tail = (1-level/100)/2
            for label, quantile in [('lower',tail), ('upper',1-tail)]:
                values = analysis.distribution_quantile(probability.prob_family, probability.prob_location, probability.prob_scale, quantile)
                np.testing.assert_allclose(values, probability[f'prob_{label}{level}'], rtol=1e-10, atol=1e-10)
        metrics = pd.read_excel(book, 'Prob_Metrics')
        metrics = metrics.loc[metrics.aggregation_level.eq('overall')].set_index(['scenario','model'])
        for key, group in probability.groupby(['scenario','model']):
            values = analysis._prob_summary(group)
            for metric in ('SB_joint_censored_NLL','SB_exact_NLL','SB_runout_NLL','SB_RMSE'):
                np.testing.assert_allclose(values[metric], metrics.loc[key,metric], rtol=1e-10, atol=1e-10)
        trunks = pd.read_excel(book, 'Trunk_Metrics')
        for row in trunks.itertuples():
            for metric in ('SB_RMSE','SB_joint_censored_NLL','SB_exact_NLL','SB_runout_NLL'):
                np.testing.assert_allclose(getattr(row,metric), metrics.loc[(row.scenario,row.model),metric], rtol=1e-10, atol=1e-10)
        return dict(worksheets=len(book.sheet_names), primary_models=len(reported), probability_rows=len(probability),
                    probability_scenario_models=len(metrics), primary_WB_PIML=reported.loc['WB-PIML'].to_dict(),
                    primary_WB_PIML_joint_NLL=float(metrics.loc[('Primary','WB-PIML'),'SB_joint_censored_NLL']))


def verify_reviewer(analysis, reference):
    # The original audit writes two summaries. Run it on a disposable copy so
    # the published run remains byte-for-byte identical to the supplied run.
    cache = ROOT / '.wb_piml_runtime'
    with tempfile.TemporaryDirectory(prefix='audit_', dir=cache) as folder:
        temporary = Path(folder).resolve()
        require(temporary.parent == cache.resolve(), 'Unsafe audit directory')
        results = temporary / 'reviewer'
        shutil.copytree(reference / 'reviewer', results)
        compressed = results/'strict_holdout/calibration.csv.gz'
        if compressed.exists():
            with gzip.open(compressed,'rb') as reader, (results/'strict_holdout/calibration.csv').open('wb') as writer:
                shutil.copyfileobj(reader,writer)
        with redirect_stdout(io.StringIO()):
            analysis.review_verify_main(['--results',str(results),'--data',str(ROOT/'data.xlsx'),
                                        '--baseline',str(reference/'primary/WB_PIML_results.xlsx')])
        summary = json.loads((results/'verification_summary.json').read_text(encoding='utf-8'))
        original = json.loads((reference/'reviewer/verification_summary.json').read_text(encoding='utf-8'))
        require(summary == original, 'Recomputed reviewer audit differs from the completed run')
    return summary


def verify_frozen(analysis, data, reference, refit):
    import joblib
    manifest = json.loads((reference/'prospective/model_manifest.json').read_text(encoding='utf-8'))
    model = reference/'prospective/frozen_model.joblib'
    require(sha(model) == manifest['model_sha256'], 'Frozen model hash mismatch')
    require(manifest['code_sha256'] == sha(canonical_source()), 'Frozen model source mismatch')
    require(manifest['data_sha256'] == sha(ROOT/'data.xlsx'), 'Frozen model data mismatch')
    bundle = joblib.load(model)
    prediction = analysis.revision_predict_state(bundle['state'], data)
    require(len(prediction) == 223 and np.isfinite(prediction).all(), 'Invalid frozen inference')
    result = dict(records=223, source_and_model_hashes='matched', finite_inference=True, selected_final_fit_rerun=False)
    if refit:
        with analysis.threadpool_limits(limits=1):
            values, coef, _ = analysis.fit_hybrid_policy(data, data, analysis.COMPETING_WB_SPEC,
                manifest['seed']+1, manifest['trees'], **manifest['settings'])
        np.testing.assert_allclose(values, prediction, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(coef, bundle['state']['coefficients'], rtol=1e-10, atol=1e-10)
        result.update(selected_final_fit_rerun=True, max_abs_prediction_difference=float(np.max(np.abs(values-prediction))))
    return result


def verify_selected_refits(analysis, data, reference):
    selected = pd.read_csv(reference/'reviewer/selection_by_fold.csv')
    selected = selected.loc[selected.model.eq('WB-PIML-Anchor')].set_index('outer_fold')
    variant = next(v for v in analysis.review_variants() if v['name'] == 'selected_pipeline')
    report = []
    cache = ROOT/'.wb_piml_runtime'
    with tempfile.TemporaryDirectory(prefix='refit_',dir=cache) as folder:
        require(Path(folder).resolve().parent == cache.resolve(), 'Unsafe refit directory')
        for fold in range(1,6):
            actual = analysis.review_refit_one(data,selected,variant,fold,400,folder,'release-selected-refit-verification')
            expected = json.loads((reference/f'reviewer/checkpoints/selected_pipeline_fold{fold}.json').read_text(encoding='utf-8'))
            differences = {}
            for table, columns in [('predictions',['pred_logN','row_nll','prob_lower90','prob_upper90','scale']),
                                   ('calibration',['mu'])]:
                a = pd.DataFrame(actual[table]).set_index('row_id').sort_index()
                e = pd.DataFrame(expected[table]).set_index('row_id').sort_index()
                pd.testing.assert_index_equal(a.index,e.index)
                np.testing.assert_allclose(a[columns],e[columns],rtol=1e-10,atol=1e-10)
                differences[table] = float(np.max(np.abs(a[columns].to_numpy()-e[columns].to_numpy())))
            np.testing.assert_allclose(actual['coefficients'],expected['coefficients'],rtol=1e-10,atol=1e-10)
            report.append(dict(fold=fold,max_abs_differences=differences))
            print(f'PASS selected pipeline refit: outer fold {fold}/5 (including four calibration fits)',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit-selected',action='store_true')
    parser.add_argument('--report',type=Path,help='Optional JSON report path; reference outputs are never overwritten')
    args = parser.parse_args()
    require(not sys.flags.optimize, 'Do not use Python -O/-OO; audit assertions must remain enabled')
    manifest = json.loads((ROOT/'release_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['sha256'].items():
        require(sha(ROOT/name) == digest, 'Release file hash mismatch: '+name)
    source = canonical_source()
    require(sha(source) == manifest['canonical_analysis_source_sha256'], 'Canonical source mismatch')
    for name,digest in manifest['original_artifact_sha256'].items():
        if name in manifest.get('compressed_original_artifacts',{}):
            archive = ROOT/manifest['compressed_original_artifacts'][name]
            with gzip.open(archive,'rb') as stream:
                actual = hashlib.file_digest(stream,'sha256').hexdigest()
        else:
            actual = sha(ROOT/name)
        require(actual == digest, 'Original artifact mismatch: '+name)
    analysis = load_analysis()
    preflight = analysis.single_preflight(ROOT/'data.xlsx')
    data = analysis.load_data(ROOT/'data.xlsx')
    require((len(data),int(data.is_exact.sum()),int(data.is_runout.sum())) == (223,158,65), 'Unexpected data counts')
    require((data.source_id.nunique(),data.campaign_id.nunique()) == (18,17), 'Unexpected source/campaign counts')
    require((len(analysis.HYBRID_CANDIDATES),len(analysis.CONTROL_CANDIDATES)) == (54,64), 'Unexpected candidates')
    require(Counter(c['model'] for c in analysis.SIMPLE_HYBRID_CANDIDATES) == {'WB-Residual':18,'Basquin-Residual':6}, 'Unexpected simple-trunk candidates')
    reference = ROOT/'reference_results'
    scores = verify_saved(analysis,data,reference)
    print('PASS saved primary and probability scores',flush=True)
    reviewer = verify_reviewer(analysis,reference)
    print('PASS 34 sensitivity variants, 170 checkpoints and strict holdout isolation/arithmetic',flush=True)
    refits = verify_selected_refits(analysis,data,reference) if args.refit_selected else []
    frozen = verify_frozen(analysis,data,reference,args.refit_selected)
    summary = dict(status='passed',canonical_source_sha256=sha(source),source_parts=3,
        original_run_files=manifest['original_run_file_count'],release_file_hashes_checked=len(manifest['sha256']),
        actual_environment=dict(python=preflight['python'],versions=preflight['versions']),
        saved_results=scores,reviewer_audit=reviewer,selected_pipeline_refits=refits,frozen_model=frozen,
        full_nested_search_rerun=False,all_sensitivity_models_retrained=False,
        verification_scope='Source and artifact identity; saved-score arithmetic and partition audits; optional selected-configuration refits. No claim of a new full workflow run.')
    if args.report:
        destination = args.report.resolve()
        require(not destination.is_relative_to(reference.resolve()), 'Do not overwrite reference results with the release audit')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('PASS release verification. Full nested search was NOT rerun.',flush=True)


if __name__ == '__main__':
    main()
