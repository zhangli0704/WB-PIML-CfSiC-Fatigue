"""Compare numeric outputs of a new full run with the archived reference run."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

RTOL = ATOL = 1e-10
KEYS = ('scenario','model','variant','strict_task_id','task_id','outer_fold','fold',
        'inner_fold','row_id','candidate_id','candidate_index','nominal','nominal_level',
        'holdout_group','campaign_id','source_id','architecture','temp_bin','env_class',
        'distribution','metric','parameter','value','evaluation','aggregation_level',
        'effect','comparison','component','strategy','repeat','pool','family')
META = {'run_hash','code_sha256','created_utc','created_at','source_file',
        'data_path','baseline_path','duration_seconds','elapsed_seconds'}

def compare_table(expected, actual, name):
    if expected.shape != actual.shape or list(expected.columns) != list(actual.columns):
        raise AssertionError('Table shape/columns differ: '+name)
    numeric = [c for c in expected.columns if pd.api.types.is_numeric_dtype(expected[c])
               or pd.to_numeric(expected[c],errors='coerce').notna().any()
               or pd.to_numeric(actual[c],errors='coerce').notna().any()]
    if not numeric:
        return 0, 0.0
    a = expected[numeric].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    b = actual[numeric].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    if not np.allclose(a,b,rtol=RTOL,atol=ATOL,equal_nan=True):
        keys = [c for c in KEYS if c in expected.columns]
        if keys:
            expected = expected.sort_values(keys,kind='stable',na_position='last').reset_index(drop=True)
            actual = actual.sort_values(keys,kind='stable',na_position='last').reset_index(drop=True)
            a = expected[numeric].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
            b = actual[numeric].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
        np.testing.assert_allclose(a,b,rtol=RTOL,atol=ATOL,equal_nan=True,err_msg=name)
    finite = np.isfinite(a)&np.isfinite(b)
    return int((~np.isnan(a)|~np.isnan(b)).sum()), float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0

def compare_json(expected, actual, name):
    count, largest = 0, 0.0
    if isinstance(expected,dict):
        keys = set(expected)-META
        if keys != set(actual)-META:
            raise AssertionError('Checkpoint keys differ: '+name)
        for key in sorted(keys):
            n,d = compare_json(expected[key],actual[key],name+'/'+key)
            count += n
            largest = max(largest,d)
    elif isinstance(expected,list):
        if len(expected) != len(actual):
            raise AssertionError('Checkpoint row count differs: '+name)
        for i,(a,b) in enumerate(zip(expected,actual)):
            n,d = compare_json(a,b,name+'/'+str(i))
            count += n
            largest = max(largest,d)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        np.testing.assert_allclose(expected,actual,rtol=RTOL,atol=ATOL,equal_nan=True,err_msg=name)
        count = 1
        if np.isfinite(expected) and np.isfinite(actual):
            largest = abs(expected-actual)
    elif expected != actual:
        raise AssertionError('Checkpoint value differs: '+name)
    return count, float(largest)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--reference',type=Path,default=Path(__file__).resolve().parent/'reference_results')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--primary-report',type=Path,help='Reuse a hash-bound primary workbook comparison')
    args = parser.parse_args()
    reference, run = args.reference.resolve(), args.run.resolve()
    if args.report.resolve().is_relative_to(reference):
        raise ValueError('Do not overwrite archived reference results')
    status = json.loads((run/'workflow_status.json').read_text(encoding='utf-8'))
    if status['status'] != 'completed' or set(status['completed_stages']) != {'primary','reviewer','verification','prospective'}:
        raise AssertionError('New workflow has not completed all four stages')
    report = dict(status='passed',original_run='full_20261008_182317_470433',
        rerun_directory=run.name,tolerance=dict(rtol=RTOL,atol=ATOL),
        full_nested_search_rerun=True,all_sensitivity_models_retrained=True,
        strict_holdout_workflow_rerun=True,frozen_model_refit=True,
        tables=[],checkpoints=[],numeric_values_compared=0,max_abs_difference=0.0)
    wb = 'primary/WB_PIML_results.xlsx'
    if args.primary_report:
        previous = json.loads(args.primary_report.read_text(encoding='utf-8'))
        for field,path in [('reference_workbook_sha256',reference/wb),('rerun_workbook_sha256',run/wb)]:
            if previous[field] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise AssertionError('Primary workbook changed since comparison')
        if previous['status'] != 'passed':
            raise AssertionError('Primary comparison did not pass')
        report['tables'].extend(dict(file=wb,**item) for item in previous['tables'])
        report['primary_images'] = previous['images']
    else:
        with pd.ExcelFile(reference/wb) as e, pd.ExcelFile(run/wb) as a:
            if e.sheet_names != a.sheet_names:
                raise AssertionError('Workbook sheets differ')
            for sheet in e.sheet_names:
                count,delta = compare_table(pd.read_excel(e,sheet),pd.read_excel(a,sheet),wb+'!'+sheet)
                report['tables'].append(dict(file=wb,sheet=sheet,numeric_values=count,max_abs_difference=delta))
    expected_csv = {p.relative_to(reference).as_posix() for p in reference.rglob('*.csv')}
    expected_csv |= {p.relative_to(reference).as_posix()[:-3] for p in reference.rglob('*.csv.gz')}
    actual_csv = {p.relative_to(run).as_posix() for p in run.rglob('*.csv')}
    if expected_csv != actual_csv:
        raise AssertionError('CSV files differ: '+str(expected_csv^actual_csv))
    for name in sorted(expected_csv):
        e = reference/name
        e = e if e.is_file() else Path(str(e)+'.gz')
        count,delta = compare_table(pd.read_csv(e),pd.read_csv(run/name),name)
        report['tables'].append(dict(file=name,numeric_values=count,max_abs_difference=delta))
    for path in sorted((reference/'reviewer').rglob('checkpoints/*.json')):
        name = path.relative_to(reference).as_posix()
        expected = json.loads(path.read_text(encoding='utf-8'))
        actual = json.loads((run/name).read_text(encoding='utf-8'))
        count,delta = compare_json(expected,actual,name)
        report['checkpoints'].append(dict(file=name,numeric_values=count,max_abs_difference=delta))
    from run import load_analysis
    import joblib
    analysis = load_analysis()
    data = analysis.load_data(Path(__file__).resolve().parent/'data.xlsx')
    e = joblib.load(reference/'prospective/frozen_model.joblib')
    a = joblib.load(run/'prospective/frozen_model.joblib')
    p = analysis.revision_predict_state(e['state'],data)
    q = analysis.revision_predict_state(a['state'],data)
    np.testing.assert_allclose(p,q,rtol=RTOL,atol=ATOL)
    np.testing.assert_allclose(e['state']['coefficients'],a['state']['coefficients'],rtol=RTOL,atol=ATOL)
    report['frozen_inference'] = dict(records=len(data),max_abs_difference=float(np.max(np.abs(p-q))))
    order = pd.read_csv(run/'reviewer/water_air_counterfactual_order.csv')
    report['water_air'] = dict(comparisons=len(order),reversals=int(order.total_order_reversal.sum()),
        maximum_water_minus_air=float(order.water_minus_air_total_logN.max()))
    if report['water_air']['reversals'] != 0:
        raise AssertionError('Environment ordering failed')
    for item in report['tables']+report['checkpoints']:
        report['numeric_values_compared'] += item['numeric_values']
        report['max_abs_difference'] = max(report['max_abs_difference'],item['max_abs_difference'])
    report['max_abs_difference'] = max(report['max_abs_difference'],report['frozen_inference']['max_abs_difference'])
    source = Path(analysis.__file__)
    report['public_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    report['comparison_scope'] = 'All numeric workbook/CSV values, checkpoint numeric values and categorical identity, plus frozen inference. Runtime provenance and presentation bytes excluded.'
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('PASS full rerun:',len(report['tables']),'tables,',len(report['checkpoints']),
          'checkpoints,',report['numeric_values_compared'],'numeric values; maximum difference',report['max_abs_difference'])

if __name__ == '__main__':
    main()
