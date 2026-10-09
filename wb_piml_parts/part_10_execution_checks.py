import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def review_verify_main(argv=None):
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=root/'reviewer_results')
    parser.add_argument('--data',type=Path,default=root/'data.xlsx')
    parser.add_argument('--baseline',type=Path,default=root/'baseline_results.xlsx')
    args=parser.parse_args(argv)
    out=args.results.resolve()
    status=json.loads((out/'run_status.json').read_text(encoding='utf-8'))
    assert status['status']=='completed' and status['refits_executed'] and status['variants'] is None
    provenance=json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    assert review_sha(args.data)==provenance['data_sha256']
    assert review_sha(args.baseline)==provenance['baseline_sha256']
    for name,digest in provenance['code_sha256'].items():assert review_sha(root/name)==digest,name
    run_hash=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    df=load_data(args.data)
    expected=set(df.row_id)
    all_predictions=pd.read_csv(out/'sensitivity_predictions.csv')
    assert set(all_predictions.variant)=={v['name'] for v in review_variants()}
    checked=0
    for name,group in all_predictions.groupby('variant'):
        assert len(group)==len(df) and group.row_id.is_unique and set(group.row_id)==expected,name
        assert group.is_exact.sum()==df.is_exact.sum() and group.is_runout.sum()==df.is_runout.sum(),name
        for fold in range(1,6):
            p=json.loads((out/'checkpoints'/f'{name}_fold{fold}.json').read_text(encoding='utf-8'))
            assert p['run_hash']==run_hash
            assert p['outer_test_used_for_calibration'] is False
            test=pd.DataFrame(p['predictions']);oof=pd.DataFrame(p['calibration'])
            train=df.loc[~df.row_id.isin(test.row_id)]
            review_require_disjoint(train,test)
            assert set(oof.row_id)==set(train.row_id) and oof.row_id.is_unique
            assert set(test.row_id)|set(oof.row_id)==expected
            assert probability_row_id_hash(train.row_id)==p['train_hash']
            assert probability_row_id_hash(test.row_id)==p['test_hash']
            for audit in p['audit']:
                valid=oof.loc[oof.inner_fold.eq(audit['inner_fold'])]
                fit=train.loc[~train.row_id.isin(valid.row_id)]
                review_require_disjoint(fit,valid)
                assert probability_row_id_hash(fit.row_id)==audit['fit_hash']
                assert probability_row_id_hash(valid.row_id)==audit['validation_hash']
            family=p['assumption'].get('distribution','lognormal')
            np.testing.assert_allclose(test.row_nll,review_row_losses(test.logN,test.pred_logN,
                test.scale,test.is_exact,family),rtol=1e-12)
            np.testing.assert_allclose(p['scale_fit']['calibration_nll'],np.mean(review_row_losses(
                oof.logN,oof.mu,p['scale_fit']['scale'],oof.is_exact,family)),rtol=1e-12)
            check=test.merge(df[REVIEW_ID],on='row_id',suffixes=('_prediction','_data'),validate='one_to_one')
            np.testing.assert_array_equal(check.is_exact_prediction,check.is_exact_data)
            np.testing.assert_allclose(check.logN_prediction,check.logN_data,rtol=1e-12)
            checked+=1
    m=pd.read_csv(out/'sensitivity_metrics.csv')
    for name,g in all_predictions.groupby('variant'):
        exact=g.loc[g.is_exact].copy();exact['error']=exact.pred_logN-exact.logN
        rmse=float(np.sqrt(exact.groupby('campaign_id').error.apply(lambda x:np.mean(x*x)).mean()))
        mae=float(exact.groupby('campaign_id').error.apply(lambda x:np.mean(np.abs(x))).mean())
        row=m.set_index('variant').loc[name]
        np.testing.assert_allclose([row.SB_RMSE,row.SB_MAE],[rmse,mae],rtol=1e-12,atol=1e-12)
    if provenance.get('sensitivity_design_version')==REVIEW_DESIGN_VERSION:
        design=pd.read_csv(out/'sensitivity_design.csv')
        if set(design.variant)!=set(all_predictions.variant):raise AssertionError('Sensitivity plan coverage mismatch')
        if design.used_for_model_selection.any():raise AssertionError('Sensitivity must not choose primary model')
        expected_plan=hashlib.sha256(json.dumps(review_design_rows(),sort_keys=True).encode('utf-8')).hexdigest()
        if expected_plan!=provenance['sensitivity_plan_sha256']:raise AssertionError('Sensitivity plan changed during run')
        norm=pd.read_csv(out/'environment_scale_identifiability.csv')
        if len(norm)!=10 or norm.max_abs_difference.max()>1e-10:raise AssertionError('Scale identity diagnostics incomplete')
        response=pd.read_csv(out/'environment_response_rows.csv')
        if not np.isfinite(response[['frozen_trunk_delta_logN','refitted_pipeline_delta_logN']].to_numpy()).all():
            raise AssertionError('Nonfinite environmental response')
        expected_drivers={v['name'] for v in review_variants() if v.get('drivers')}
        if set(response.variant)!=expected_drivers:raise AssertionError('Environmental response coverage incomplete')
        for _,g in response.groupby('variant'):
            if len(g)!=len(df) or not g.row_id.is_unique:raise AssertionError('Environmental response row mismatch')
        interactions=pd.read_csv(out/'environment_interactions.csv')
        if interactions.variant.nunique()!=4:raise AssertionError('Interaction checks incomplete')
        detail=pd.read_csv(out/'data_joint_detailed.csv')
        if detail.n_records.sum()!=len(df):raise AssertionError('Detailed support counts incomplete')
    np.testing.assert_allclose(m.joint_NLL,m.exact_component_NLL+m.runout_component_NLL,atol=1e-12)
    assert pd.read_csv(out/'baseline_metric_regression.csv').passed.all()
    assert pd.read_csv(out/'baseline_prediction_regression.csv').passed.all()
    freq=pd.read_csv(out/'hyperparameter_frequency.csv')
    family=freq.loc[freq.model.eq('WB-PIML-Anchor') & freq.parameter.eq('family')].set_index('value')
    selection=pd.read_excel(args.baseline,sheet_name='Nested_Selection')
    expected_counts=selection.loc[selection.model.eq('WB-PIML-Anchor'),'family'].value_counts()
    for name,row in family.iterrows():assert row['count']==expected_counts.get(name,0)
    assert family['count'].sum()==5
    assert family.role.eq('searched').all()
    water_verify_outputs(out,df)
    strict_verify_outputs(out,df)
    summary=dict(status='passed',variants=len(review_variants()),outer_fits=checked,inner_OOF_fits=checked*4,
        prediction_rows=len(all_predictions),bootstrap_draws=status['bootstrap'],
        input_and_code_hashes='matched',campaign_leakage_in_refits=False,
        checks='coverage, train/test and inner campaign separation, likelihood/scale arithmetic, independent RMSE/MAE, baseline reproduction, selection frequencies and design diagnostics')
    summary['strict_holdout']=json.loads((out/'strict_holdout/summary.json').read_text(encoding='utf-8'))
    summary['censor_evidence']=json.loads((out/'censor_evidence_status.json').read_text(encoding='utf-8'))
    summary['status_meaning']='available-output arithmetic and isolation checks passed; see disclosed unavailable tasks and evidence gaps'
    (out/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


SINGLE_PRIMARY_SHA256 = 'dbcc6bc2ba511d391b705193ba892cbeb14048cad6314b0208d5c779cf48d962'
SINGLE_PRIMARY_GUARD = 'if __name__ == "__main__":\n    main()\n'
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import importlib.metadata
import sys
from types import SimpleNamespace

SINGLE_DATA_SHA256 = 'a29fb0c38e2dcb5bc1082be1a925d4b1cd562bca7c580ab9a932513ecb51272c'
SINGLE_DEPENDENCIES = {'numpy':'2.3.5','pandas':'2.2.3','scipy':'1.17.0',
    'scikit-learn':'1.8.0','openpyxl':'3.1.5','joblib':'1.5.3',
    'threadpoolctl':'3.6.0','matplotlib':'3.10.8','python-pptx':'1.0.2','Pillow':'12.3.0'}


def single_available_memory_gib():

    try:
        if os.name=='nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_=[('dwLength',ctypes.c_uint32),('dwMemoryLoad',ctypes.c_uint32),
                    ('ullTotalPhys',ctypes.c_uint64),('ullAvailPhys',ctypes.c_uint64),
                    ('ullTotalPageFile',ctypes.c_uint64),('ullAvailPageFile',ctypes.c_uint64),
                    ('ullTotalVirtual',ctypes.c_uint64),('ullAvailVirtual',ctypes.c_uint64),
                    ('ullAvailExtendedVirtual',ctypes.c_uint64)]
            memory=MemoryStatus();memory.dwLength=ctypes.sizeof(memory)
            query=ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
            query.argtypes=[ctypes.POINTER(MemoryStatus)];query.restype=ctypes.c_int
            if query(ctypes.byref(memory)):return memory.ullAvailPhys/(1024**3)
        elif sys.platform.startswith('linux'):
            for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
                if line.startswith('MemAvailable:'):return int(line.split()[1])/(1024**2)
    except (OSError,ValueError,AttributeError):
        pass
    return None


def single_worker_budget(logical,physical,available_gib):


    cores=max(1,min(int(logical),int(physical),16))
    if available_gib is None:return min(cores,16)
    
    memory_workers=max(1,int(max(0.0,float(available_gib)-4.0)/1.5))
    return min(cores,memory_workers)


def single_parallel_settings(requested=None):
    from joblib import cpu_count
    logical=max(1,cpu_count())
    try:physical=max(1,cpu_count(only_physical_cores=True))
    except (OSError,ValueError,NotImplementedError):physical=min(logical,16)
    available=single_available_memory_gib()
    jobs=single_worker_budget(logical,physical,available) if requested is None else requested
    return jobs,dict(mode='automatic' if requested is None else 'manual',
        available_logical_cpus=logical,available_physical_cores=physical,
        available_memory_GiB=None if available is None else round(available,2),
        workers=jobs,threads_per_worker=1,
        policy='physical cores with heuristic memory allowance; no speed benchmark or model training')


def single_environment_notes(version_info,installed):

    if tuple(version_info[:2]) < (3,12):
        raise RuntimeError('此入口要求 Python 3.12 或更高版本。')
    notes=[]
    if tuple(version_info[:2]) != (3,12):
        notes.append('当前Python不是3.12参考环境；允许继续，完整运行尚需在当前环境验证。')
    notes.extend(f'{name}: 当前 {installed[name]}，参考 {wanted}'
        for name,wanted in SINGLE_DEPENDENCIES.items() if installed[name]!=wanted)
    if notes:
        notes.append('版本差异可能影响数值结果；保留实际环境记录和原有预测/概率一致性校验，不自动放宽结果容差。')
    return notes


def single_preflight(data):

    source=Path(__file__).read_text(encoding='utf-8')
    start='# ===== REVISED PRIMARY CODE BEGIN =====\n'
    end='# ===== REVISED PRIMARY CODE END =====\n'
    preserved=source.split(start,1)[1].split(end,1)[0]+SINGLE_PRIMARY_GUARD
    if hashlib.sha256(preserved.encode('utf-8')).hexdigest()!=SINGLE_PRIMARY_SHA256:
        raise RuntimeError('修订主分析代码校验不一致，请使用完整原文件。')
    if not data.is_file():
        raise FileNotFoundError(f'未找到数据：{data}。请把修订后的 data.xlsx 放在本代码旁。')
    if review_sha(data)!=SINGLE_DATA_SHA256:
        raise RuntimeError('数据与2026-10-08修订版不一致。请使用补入Min 2013 runout后的224条记录版本 data.xlsx。')
    installed={name:importlib.metadata.version(name) for name in SINGLE_DEPENDENCIES}
    environment_notes=single_environment_notes(sys.version_info,installed)
    import inspect
    if 'return_as' not in inspect.signature(Parallel).parameters:
        raise RuntimeError('当前joblib缺少return_as接口，需安装支持此接口的版本（参考1.5.3）。')
    if any(value is None for value in (Font,Image,Presentation,RGBColor,PP_ALIGN,Inches,Pt)):
        raise ImportError('缺少 Excel/PPT 导出依赖，见文件开头。')
    frame=load_data(data)
    if (len(frame),int(frame.is_exact.sum()),int(frame.is_runout.sum()))!=(EXPECTED_VERIFIED_ROWS,EXPECTED_EXACT_ROWS,EXPECTED_RUNOUT_ROWS):
        raise RuntimeError('数据记录/失效/runout数量与返修数据不符。')
    if not frame.row_id.is_unique or not frame.record_equivalence_id.is_unique:
        raise RuntimeError('存在重复分析标识。')
    return dict(python=sys.version,versions=installed,environment_notes=environment_notes,
        reference_python='3.12',reference_versions=SINGLE_DEPENDENCIES,
        data_sha256=review_sha(data),
        revised_primary_sha256=SINGLE_PRIMARY_SHA256,single_file_sha256=review_sha(__file__),
        records=len(frame),exact=int(frame.is_exact.sum()),runout=int(frame.is_runout.sum()),framework_unchanged=True,residual_input_policy_changed=True,residual_input_policy=RESIDUAL_INPUT_POLICY,environment_assumption_changed=True,primary_model_id=REVISION_MODEL_ID)


class SingleLogTee:
    def __init__(self,console,log):
        self.console,self.log=console,log
    def write(self,text):
        self.log.write(text);self.log.flush()
        try:return self.console.write(text)
        except UnicodeEncodeError:
            encoding=getattr(self.console,'encoding',None) or 'ascii'
            return self.console.write(text.encode(encoding,errors='replace').decode(encoding))
    def flush(self):
        self.console.flush();self.log.flush()
    def __getattr__(self,name):
        return getattr(self.console,name)


def single_check_primary(workbook):
    required={'Protocol','Nested_Selection','Nested_Candidates','Prob_Predictions',
              'Prob_Calibration','Prob_Metrics','Primary_Campaign_Metrics','Paired_Bootstrap',
              'Prob_Paired_CI','Empirical_PI_Summary','External_PI_Summary','Sensitivity','External_Selection'}
    with pd.ExcelFile(workbook) as excel:
        if not required.issubset(excel.sheet_names):raise RuntimeError('主工作簿缺少必要表。')
        protocol=pd.read_excel(excel,sheet_name='Protocol').set_index('item').value
    if str(protocol.get('revision_model_id',''))!=REVISION_MODEL_ID:
        raise RuntimeError('该主工作簿不是可学习水氧倍率版；请默认完整运行，不能复用旧固定权重主结果。')
    if str(protocol['data_sha256'])!=SINGLE_DATA_SHA256:
        raise RuntimeError('主工作簿与本次数据不一致。')


def single_main(argv=None):

    parser=argparse.ArgumentParser(description='WB-PIML 单文件返修版；默认完整运行，--check 只检查环境。')
    parser.add_argument('--check',action='store_true',help='只检查环境和数据，不训练')
    parser.add_argument('--data',type=Path,default=None,help='默认自动识别同目录data.xlsx或唯一data*.xlsx')
    parser.add_argument('--out',type=Path,help='新的结果目录；默认同目录 runs/full_时间')
    parser.add_argument('--jobs',type=int,default=None,help='默认自动按CPU及可用内存设置；也可手动指定正整数')
    parser.add_argument('--censor-evidence',type=Path,help='可选：逐条停止规则证据CSV；无此文件也可完整运行')
    parser.add_argument('--reviewer-only',action='store_true',help='只对已完成的主结果运行补充分析')
    parser.add_argument('--baseline',type=Path,help='与 --reviewer-only 一起使用的主结果xlsx路径')
    parser.add_argument('--predict',type=Path,help='新实验条件CSV/XLSX；仅用冻结模型预测')
    parser.add_argument('--model',type=Path,help='--predict使用的frozen_model.joblib')
    parser.add_argument('--evaluate',type=Path,help='独立实验结果CSV/XLSX；不训练')
    parser.add_argument('--predictions',type=Path,help='--evaluate使用的预先保存predictions.csv')
    args=parser.parse_args(argv)
    if args.predict or args.evaluate:
        revision_external_cli(args,parser)
        return
    if args.model or args.predictions:parser.error('--model/--predictions须与相应独立验证模式同时给出')
    if args.jobs is not None and args.jobs<1:parser.error('--jobs 必须大于0')
    if args.reviewer_only!=bool(args.baseline):parser.error('--reviewer-only 与 --baseline 必须同时给出')
    data=revision_resolve_data(args.data)
    args.jobs,parallel_settings=single_parallel_settings(args.jobs)
    
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[variable]='1'
    environment=single_preflight(data)
    environment['parallel_settings']=parallel_settings
    print('启动检查：环境、数据和多进程函数传递（不训练模型）...',flush=True)
    environment['execution_preflight']=single_execution_preflight(args.jobs)
    if args.check:
        print(json.dumps(environment,ensure_ascii=False,indent=2))
        print('PREFLIGHT PASSED. 环境检查通过；未启动训练。')
        return
    output=(args.out or Path(__file__).resolve().parent/'runs'/('full_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists() and any(output.iterdir()):parser.error(f'结果目录非空，请使用新目录：{output}')
    output.mkdir(parents=True,exist_ok=True)
    review_write_design(output)
    (output/'analysis_plan.json').write_text(json.dumps(review_design_rows(),ensure_ascii=False,indent=2),encoding='utf-8')
    status=dict(status='running',mode='reviewer_only' if args.reviewer_only else 'full',
        environment=environment,completed_stages=[],source_file=Path(__file__).name,
        settings=dict(seed=SEED,trees=400,audit_trees=160,repeats=10,bootstrap=5000,jobs=args.jobs),
        primary_model_id=REVISION_MODEL_ID,supplement_role='固定修订主模型已选配置的敏感性及诊断；不据此再改选')
    def write_status():
        (output/'workflow_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    write_status()
    global WORKERS,TREE_JOBS,BLAS_THREADS
    WORKERS,TREE_JOBS,BLAS_THREADS=args.jobs,1,1
    try:
        with (output/'run.log').open('w',encoding='utf-8') as log:
            with redirect_stdout(SingleLogTee(sys.stdout,log)),redirect_stderr(SingleLogTee(sys.stderr,log)):
                print(json.dumps(environment,ensure_ascii=False,indent=2),flush=True)
                print(f'自动/手动并行设置：{args.jobs} 个进程，每进程 1 个计算线程。',flush=True)
                print('每个阶段按实际任务数使用进程；5折主验证最多同时处理5折。',flush=True)
                if args.reviewer_only:
                    baseline=args.baseline.resolve();single_check_primary(baseline)
                else:
                    primary=output/'primary'
                    print('阶段 1/4：可学习水氧倍率主分析（原候选池及验证规模）',flush=True)
                    primary_args=SimpleNamespace(data=str(data),out=str(primary),trees=400,
                        audit_trees=160,repeats=10,bootstrap=5000,export_csv=False)
                    with threadpool_limits(limits=1):_run_analysis(primary_args)
                    baseline=primary/'WB_PIML_results.xlsx';single_check_primary(baseline)
                    if not (primary/'WB_PIML_figures.pptx').is_file() or len(list(primary.glob('*.png')))<10:
                        raise RuntimeError('修订主分析图件/PPT输出不完整。')
                    status['completed_stages'].append('primary');write_status()
                status['primary_workbook']=str(baseline)
                status['primary_workbook_sha256']=review_sha(baseline);write_status()
                reviewer=output/'reviewer'
                print(f'阶段 2/4：审稿补充统计及{len(review_variants())}组具名敏感性分析（不改选主模型）',flush=True)
                review_main(['--refit','--data',str(data),'--baseline',str(baseline),
                    '--out',str(reviewer),'--jobs',str(args.jobs),'--trees','400','--bootstrap','5000']
                    +(['--censor-evidence',str(args.censor_evidence)] if args.censor_evidence else []))
                status['completed_stages'].append('reviewer')
                status['strict_holdout']=json.loads((reviewer/'strict_holdout/summary.json').read_text(encoding='utf-8'))
                status['censor_evidence']=json.loads((reviewer/'censor_evidence_status.json').read_text(encoding='utf-8'))
                write_status()
                print('阶段 3/4：预测、概率计算和训练/测试隔离检查',flush=True)
                review_verify_main(['--results',str(reviewer),'--data',str(data),'--baseline',str(baseline)])
                status['completed_stages'].append('verification')
                write_status()
                print('阶段 4/4：按原内层规则选择并冻结全数据模型，生成独立实验模板',flush=True)
                with threadpool_limits(limits=1):
                    revision_export_frozen(data,output/'prospective',trees=400)
                status['completed_stages'].append('prospective')
                status['status']='completed_with_strict_fit_failures' if status['strict_holdout']['n_fit_failed'] else 'completed'
                status['scientific_readiness']='requires_manuscript_result_consistency_review'
                write_status()
                print(f'计算流程结束（状态：{status["status"]}）；请阅读返修关键结论与缺口.md。\n结果目录：{output}',flush=True)
    except BaseException as exc:
        status['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        status['error']=str(exc);write_status()
        raise


REVIEW_DESIGN_VERSION = 'reviewer-audit-20261008-v5-shared-environment-residual'
REVIEW_ENVIRONMENT_BASIS = '''# 可学习水氧倍率的假设与验证

主模型：E(vacuum)=0，E(air/oxygen)=1，E(water/oxygen)=10**beta，beta>=0。
只增加一个物理位置参数；原残差候选池、种子、删失似然及组隔离验证规模保留。
每个内层训练子集重新估计beta，外层测试寿命不用于参数、范围或候选选择。
真空0为理想化边界；空气1固定尺度；水氧方向约束是适用条件下的物理先验，不是全条件实验证明。
主拟合无有限beta上界，无新增beta正则项。原有物理参数边界、ridge和环境惩罚仍适用。
计算域对照10/100仅检验一个/两个数量级限制是否改变估计，不是物理上限或待选超参数。
初值中的1/1.5/5只用于同一目标的数值多起点，不是允许权重的离散集合。

数据限制：水氧12条记录仅1300℃、3Hz、R=0.1、2D编织，来自两个试验组，无匹配空气数据。
参数可能与温度、K和试验来源混杂。可拟合不等于可辨识，更不等于测得独立环境反应速率。
若训练无水氧或无空气参考，beta固定0且标记未辨识；不能利用留出环境寿命拟合。
水氧LOEO中的参考倍率1只是未辨识情况下的公开假设，不能称为学习了未知气氛效果。

对照：selected_pipeline为训练内估计；water_1和water_1.5为固定权重条件对照。
所有对照保留修订模型选定残差配置，但重新拟合其训练物理参数和OOF尺度。
原0.5/1.25/1.75/2/3情景仍保留，0.5是允许方向反转的压力测试，不是主模型先验。
no_environment_damage关闭显式环境物理项；所有残差均不使用环境输入，因此该消融移除显式环境作用。environment_penalty_0仅取消惩罚。
温度/应力/频率驱动消融、真空小非零情景及有限权重×频率交互全部保留。
水氧系数不由环境样本比例或不同环境原始平均寿命直接计算。

water_identifiability_profiles.csv同时报告训练集原惩罚目标剖面和保留干扰参数边界的无惩罚似然剖面。
不从剖面挑更高外测精度，不自动给95%区间；若范围内曲线平坦或自由拟合存在局部极值，明确输出。
water_air_counterfactual_order.csv检查物理主干和完整模型；环境共享残差使最终预测保持主干的环境顺序，发现逆转即报错。
这些环境切换是模型情景检查，不是实测空气/水氧配对验证。

本次是审稿和已查看结果之后的模型修订，不称为预注册。敏感性区间是固定OOF预测的试验组bootstrap，
不包含模型重选不确定性，也未作多重比较校正。本次结构修订用于满足既定环境方向先验，不根据外层误差改选水氧权重。

机制资料只支持环境作用及条件依赖，不支持通用1.5倍率：
Liu C.-D. et al. Fatigue Behavior of 2D C/SiC Composites in Water-vapor Containing Environment at 1300℃.
Journal of Inorganic Materials, 2008, 23(4), 729. https://doi.org/10.3724/SP.J.1077.2008.00729
C/SiC Life Prediction for Propulsion Applications. NASA NTRS 20040033966.
https://ntrs.nasa.gov/citations/20040033966
参数诊断方法参考：Raue A. et al. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood.
Bioinformatics 2009;25(15):1923–1929. https://doi.org/10.1093/bioinformatics/btp358
本程序没有直接使用该文的渐近置信阈值，因本模型含约束、惩罚和少量异质试验组。

后续先冻结预测再记录实验结果；空气试验不能标定水氧倍率。倍率需同批材料同工况气氛对照及独立验证组。
替代生存分布仍假设条件独立删失；runout权重分析不证明该假设成立。停止规则仍需文献证据。
预测区间尚非工程设计寿命界限。
'''


def review_design_rows():
    rows=[]
    for v in review_variants():
        n=v['name'];d=v.get('drivers',{})
        if n=='selected_pipeline':
            role,reason='primary_reference','训练内估计水氧倍率参照；不按外层误差改选'
        elif n.startswith('water_cap_'):
            role,reason='computational_boundary_test','log10倍率上界1/2即10/100，仅为一/两个数量级计算域依赖检查；不是物理上界或待选超参数'
        elif n.startswith('water_'):
            x=d['water_factor']
            role=('stress_test' if x in (.5,3.) else 'fixed_equal_control' if x==1 else 'legacy_weight_control' if x==1.5 else 'local_perturbation')
            reason=f'相对于原水氧1.5的比值={x/1.5:g}；相对于空气1的比值={x:g}；范围是分析假设，不是实测置信区间'
        elif n.startswith('inert_'):
            role,reason='zero_boundary_test',f'真空显式通道从0改为参考空气的{100*d["inert_factor"]:g}%；不是残氧测量值'
        elif n.startswith('joint_water_'):
            role,reason='limited_interaction','水氧0.5/2与频率指数0.5/1.5的2×2交叉；保留对应单因素对照'
        elif n.startswith('frequency_'):
            role,reason='exposure_time_test','指数1对应每循环暴露时间1/f；0关闭该显式频率驱动，0.5/1.5为对称扰动'
        elif n.startswith('runout_'):
            role,reason='censor_loss_influence','删失损失乘子0/0.5/2；评价用标准NLL；不代表改变真实停止机制'
        elif n.startswith('refit_') or n.startswith('WB_CD_'):
            role,reason='distribution_assumption','替代生存分布；参数仅由开发集估计；全部在log10寿命坐标评分'
        elif n.startswith('environment_penalty_'):
            role,reason='regularization_audit','原候选池已有惩罚水平；与关闭环境损伤通道严格区分'
        else:
            role,reason='component_ablation','具名组件消融；沿用环境共享残差输入规则，不改变主模型'
        rows.append(dict(variant=n,role=role,rationale=reason,settings_json=json.dumps(v,sort_keys=True),
            water_to_air=d.get('water_factor','learned_training_only'),inert_to_air=d.get('inert_factor',0.),
            frequency_exponent=d.get('frequency_exponent',1.),
            numeric_weights_measured=False,used_for_model_selection=False,
            design_timing='revision-stage post-hoc; previously inspected data',
            scope='conditional sensitivity, not global sensitivity or new external validation'))
    return rows


def review_write_design(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    review_save(pd.DataFrame(review_design_rows()),out,'sensitivity_design')
    (out/'运行与修改说明.txt').write_text(REVISION_RUN_GUIDE,encoding='utf-8')
    (out/'environment_weight_rationale.md').write_text(REVIEW_ENVIRONMENT_BASIS,encoding='utf-8')
    mapping=[
        ('R2.1','Novelty','manuscript_required','Established components must be distinguished from integration contribution'),
        ('R2.2/R2.8','Suggested literature','manuscript_required','10.3390/machines13100906; 10.1002/srin.202400850; no automatic algorithm adoption'),
        ('R2.3','Database support','data_joint_grid.csv; data_joint_detailed.csv; sparse_cell_performance.csv','Counts are not formal independent effective sample size'),
        ('R2.4','Low R2','retained_Primary_Campaign_Metrics.csv; retained_Paired_Bootstrap.csv','Absolute errors, intervals, stronger baselines; no design-life claim'),
        ('R2.5/R3.1','Environment weights','sensitivity_design.csv; sensitivity_metrics.csv; environment_improvement_attribution.csv; water_fallback_attribution.csv; water_air_counterfactual_order.csv','No calibrated universal weights; limited interactions; no outcome-driven winner'),
        ('R2.6','Selection overfitting','candidate_selection_frequency.csv; hyperparameter_frequency.csv','Nested validation does not eliminate small-campaign uncertainty'),
        ('R2.7','Runouts','distribution_refit_PIML_vs_CD.csv; censor_protocol_audit.csv; censor_evidence_gaps.csv','Stopping mechanism requires source evidence; no fabricated completed audit'),
        ('R2.9','Generalization','holdout_overlap.csv; strict_holdout/task_status.csv; strict_holdout/metrics_matched_population.csv; support_stratum_performance.csv','Factor holdouts with campaign overlap are not independent material validation'),
        ('R2.10/R3.5','Conclusions and intervals','retained_Empirical_PI_Summary.csv; retained_External_PI_Summary.csv; ../prospective/model_manifest.json; ../prospective/独立实验使用说明.txt','Future independent matched-condition validation required'),
        ('R3.2','NLL tradeoff','probability_score_decomposition.csv; primary_NLL_difference_CI.csv','Exact and runout contributions use common population weights'),
        ('R3.3','Weak holdouts','holdout_performance_campaign_weighted.csv','Report vacuum/water and temperature failures, do not tune them away'),
        ('R3.4','Penalty ablation','sensitivity_paired_CI.csv; sensitivity_subgroups.csv','Zero regularization is not zero environmental damage'),
    ]
    review_save(pd.DataFrame(mapping,columns=['reviewer_comment','topic','evidence_or_action','limitation']),out,'reviewer_response_map')


def review_environment_diagnostics(df,predictions,results,out):

    base=predictions.loc[predictions.variant.eq('selected_pipeline')].set_index('row_id')
    if base.empty:return
    reference={int(r['fold']):r for r in results if r['variant']=='selected_pipeline'}
    details=[];normalization=[]
    for fold,r in reference.items():
        ids=[p['row_id'] for p in r['predictions']]
        raw=df.set_index('row_id').loc[ids].reset_index()
        reference_frame=review_apply_physics_drivers(raw)
        coef=np.asarray(r['coefficients'],float);gamma=float(r['settings']['gamma'])
        original=review_predict_competing_damage(reference_frame,coef,gamma)
        for scale in (.5,2.):
            frame=reference_frame.copy();frame['environment_exposure']*=scale
            adjusted=coef.copy();adjusted[2]-=math.log10(scale)
            difference=np.max(np.abs(review_predict_competing_damage(frame,adjusted,gamma)-original))
            if difference>1e-10:raise AssertionError('Environment scale/K identity failed')
            normalization.append(dict(fold=fold,common_E_multiplier=scale,max_abs_difference=difference,
                refitted=False,note='algebraic invariance; transformed coefficients need not satisfy fit bounds'))
        for v in review_variants():
            if not v.get('drivers'):continue
            frame=review_apply_physics_drivers(raw,**v['drivers'])
            shift=review_predict_competing_damage(frame,coef,gamma)-original
            alt=predictions.loc[predictions.variant.eq(v['name'])].set_index('row_id')
            if alt.empty:continue
            for i,row in raw.iterrows():
                rid=row.row_id
                details.append(dict(variant=v['name'],fold=fold,row_id=rid,campaign_id=row.campaign_id,
                    source_id=row.source_id,record_equivalence_id=row.record_equivalence_id,
                    env_class=row.env_class,is_exact=bool(row.is_exact),
                    frozen_trunk_delta_logN=float(shift[i]),
                    refitted_pipeline_delta_logN=float(alt.loc[rid,'pred_logN']-base.loc[rid,'pred_logN'])))
    detail=pd.DataFrame(details);review_save(detail,out,'environment_response_rows')
    summary=[]
    for (variant,env),g in detail.groupby(['variant','env_class']):
        w=analysis_weights(g);w=w/w.sum()
        row=dict(variant=variant,env_class=env,n_records=len(g),n_campaigns=g.campaign_id.nunique())
        for c in ['frozen_trunk_delta_logN','refitted_pipeline_delta_logN']:
            row[c+'_weighted_mean']=float(np.dot(w,g[c]))
            row[c+'_weighted_mean_abs']=float(np.dot(w,np.abs(g[c])))
            row[c+'_max_abs']=float(np.abs(g[c]).max())
        summary.append(row)
    review_save(pd.DataFrame(summary),out,'environment_response_summary')
    review_save(pd.DataFrame(normalization),out,'environment_scale_identifiability')
    interaction=[]
    pivot=predictions.pivot(index='row_id',columns='variant',values='pred_logN')
    for water in (.5,2.):
        for freq in (.5,1.5):
            name=f'joint_water_{water:g}_frequency_{freq:g}'
            names=[name,f'water_{water:g}',f'frequency_power_{freq:g}','selected_pipeline']
            if not set(names).issubset(pivot.columns):continue
            delta=pivot[name]-pivot[names[1]]-pivot[names[2]]+pivot[names[3]]
            part=df.set_index('row_id').loc[delta.index].copy();part['interaction_logN']=delta
            for env,g in part.groupby('env_class'):
                w=analysis_weights(g);w/=w.sum()
                interaction.append(dict(variant=name,env_class=env,n_records=len(g),n_campaigns=g.campaign_id.nunique(),
                    weighted_mean_interaction=float(np.dot(w,g.interaction_logN)),
                    weighted_mean_absolute_interaction=float(np.dot(w,np.abs(g.interaction_logN))),
                    meaning='combined-minus-water-minus-frequency-plus-baseline; refitted pipeline, not causal interaction'))
    review_save(pd.DataFrame(interaction,columns=['variant','env_class','n_records','n_campaigns',
        'weighted_mean_interaction','weighted_mean_absolute_interaction','meaning']),out,'environment_interactions')


def single_worker_probe(functions):

    names=[fn.__name__ for fn in functions]
    y=np.array([4.,5.]);mu=np.array([4.2,4.8]);exact=np.array([True,False])
    losses=review_row_losses(y,mu,.7,exact)
    quantiles=review_quantile(mu,.7,np.array([.05,.95]))
    return dict(functions=names,losses=losses.tolist(),quantiles=quantiles.tolist())


def single_execution_preflight(jobs):

    if sys.flags.optimize:raise RuntimeError('请不要使用python -O/-OO；该选项会关闭结果校验断言。')
    funcs=[review_refit_one,_run_analysis,review_verify_main,water_profile_one,revision_export_frozen,strict_holdout_task]
    expected=single_worker_probe(funcs)
    if jobs>1:
        with parallel_config(backend='loky',inner_max_num_threads=1):
            rows=Parallel(n_jobs=2)(delayed(single_worker_probe)(funcs) for _ in range(2))
        if any(r!=expected for r in rows):raise RuntimeError('Worker startup self-check failed')
    return dict(status='passed',worker_processes_checked=2 if jobs>1 else 0,
                model_training_executed=False,checks='function transport, likelihood, quantiles')


REVISION_MODEL_ID = 'WB-PIML-environment-shared-residual-20261008'
REVISION_RUN_GUIDE = """本次版本：WB-PIML-environment-shared-residual-20261008
把本Python文件与data.xlsx（或唯一data*.xlsx）放在同一目录，VS Code运行Python文件即可。
无需旧代码和历史结果。完整运行四阶段：主分析、34组条件敏感性、结果校验、冻结模型导出。
自动按物理核心和可用内存安排多进程，每进程限制1个数值线程。

本次修改残差输入，环境效应仅由物理主干描述；主分析、严格因素留出、敏感性和区间校准全部重新运行。
优先阅读 reviewer/返修关键结论与缺口.md；严格验证见 reviewer/strict_holdout/。
同一轮主分析中原本隔离的因素任务复用，有重叠任务重新选型/校准；数据不足或失败明确报告。
主干环境因子为0/1/10**beta，beta>=0且在训练内拟合。残差仅使用非环境输入和机械派生特征；候选池、种子、验证划分及规模保留。
所有主分析与因素保留验证重新计算。原0/1/1.5在补充分析中作为固定修订配置的条件对照。
不按敏感性分数寻找最优环境权重，不把模型估计当作已独立标定的环境物理常数。

输出：
primary/WB_PIML_results.xlsx：修订主结果、图件和PPT在primary目录。
reviewer/environment_model_comparison.csv：可学习倍率/固定1/固定1.5/关闭环境项的条件比较。
reviewer/environment_matched_support.csv：跨气氛可比条件数量，非因果辨识证据。
reviewer/verification_summary.json：完成全部34组以后生成的计算和隔离校验。
prospective/frozen_model.joblib及model_manifest.json：未来实验用冻结模型，不等于已经完成外部验证。
prospective/独立实验使用说明.txt及空白CSV模板：实验前预测与实验后评价方法。
workflow_status.json和run.log：当前状态和运行日志。

所有结果写入本Python文件旁runs/full_时间目录，不覆盖旧结果。
旧固定权重及旧含环境残差结果不能通过--reviewer-only当作新版主结果复用。旧冻结模型须重新生成。
完整运行由用户执行，本版本未声称已在用户Windows/Python环境全量训练通过。
启动--check可验证环境和两进程传递，但不会计算论文预测结果。
"""

PROSPECTIVE_INPUT_COLUMNS = ['specimen_id','campaign_id','architecture','arch_group',
    'env_class','T_C','R','frequency_Hz','sigma_max_MPa_before_normalization','UTS_MPa']

