def revision_resolve_data(argument=None):
    if argument is not None:
        p=Path(argument).expanduser().resolve()
        if not p.is_file():raise FileNotFoundError(f'指定的数据文件不存在：{p}')
        return p
    here=Path(__file__).resolve().parent
    default=here/'data.xlsx'
    if default.is_file():return default
    candidates=sorted(p for p in here.glob('data*.xlsx') if p.is_file() and not p.name.startswith('~$'))
    if len(candidates)==1:return candidates[0]
    if not candidates:raise FileNotFoundError('请把本Python文件与data.xlsx放在同一文件夹；data(9).xlsx也可自动识别。')
    raise ValueError('同目录存在多个data*.xlsx，请保留需要运行的一份，或用 --data 指定文件。')


def review_environment_overlap(df,out):

    keys=['architecture_detail','T_C','R','frequency_Hz']
    cells=review_counts(df,keys+['env_class'])
    review_save(cells,out,'environment_condition_counts')
    rows=[]
    for scope,matched in [('same_recorded_conditions',keys),
            ('same_conditions_and_stress',keys+['sigma_max_MPa_before_normalization']),
            ('same_campaign_conditions_and_stress',['campaign_id']+keys+['sigma_max_MPa_before_normalization'])]:
        for a,b in itertools.combinations(sorted(df.env_class.unique()),2):
            left=df.loc[df.env_class.eq(a)];right=df.loc[df.env_class.eq(b)]
            common=left[matched].drop_duplicates().merge(right[matched].drop_duplicates(),on=matched,validate='one_to_one')
            na=len(left.merge(common,on=matched,validate='many_to_one'))
            nb=len(right.merge(common,on=matched,validate='many_to_one'))
            rows.append(dict(scope=scope,environment_a=a,environment_b=b,
                matched_condition_cells=len(common),records_a=na,records_b=nb,
                interpretation='recorded covariate match only; material-batch/atmosphere equivalence requires source evidence'))
    review_save(pd.DataFrame(rows),out,'environment_matched_support')


def revision_json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def revision_json(path,payload):
    Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2,
        default=revision_json_default),encoding='utf-8')


def revision_read_table(path):
    path=Path(path)
    if path.suffix.lower()=='.csv':return pd.read_csv(path,encoding='utf-8-sig')
    if path.suffix.lower() in ('.xlsx','.xlsm'):return pd.read_excel(path,sheet_name=0)
    raise ValueError('独立实验输入仅支持UTF-8 CSV或XLSX。')


def revision_prediction_frame(raw):
    missing=sorted(set(PROSPECTIVE_INPUT_COLUMNS)-set(raw.columns))
    if missing:raise ValueError(f'独立实验条件缺少列：{missing}')
    if raw.empty:raise ValueError('实验条件模板尚未填写。')
    d=raw[PROSPECTIVE_INPUT_COLUMNS].copy()
    for key in ['specimen_id','campaign_id','architecture','arch_group','env_class']:
        if d[key].isna().any():raise ValueError(f'{key}不能留空')
        d[key]=d[key].astype(str).str.strip()
        if d[key].eq('').any():raise ValueError(f'{key}不能留空')
    if d.specimen_id.duplicated().any():raise ValueError('specimen_id必须唯一')
    for key in ['T_C','R','frequency_Hz','sigma_max_MPa_before_normalization','UTS_MPa']:
        d[key]=pd.to_numeric(d[key],errors='raise')
        if not np.isfinite(d[key]).all():raise ValueError(f'{key}包含非有限值')
    if (d[['frequency_Hz','sigma_max_MPa_before_normalization','UTS_MPa']]<=0).any().any():
        raise ValueError('频率、最大应力和UTS必须为正数')
    if (d.R>=1).any() or (d.T_C<=-273.15).any():raise ValueError('R必须小于1，温度必须高于绝对零度')
    d['architecture_detail']=d.architecture.str.lower()
    d['architecture']=d.arch_group.str.lower();d['env_class']=d.env_class.str.lower()
    allowed={'air_or_oxygen','water_oxygen','vacuum'}
    if not set(d.env_class).issubset(allowed):raise ValueError(f'env_class请使用{sorted(allowed)}')
    d['S']=d.sigma_max_MPa_before_normalization/d.UTS_MPa
    d['stress_level_sigma_max_over_UTS']=d.S
    d['logS']=np.log10(d.S.clip(lower=1e-8));d['logf']=np.log10(d.frequency_Hz.clip(lower=1e-8))
    d['logUTS']=np.log10(d.UTS_MPa.clip(lower=1e-8));d['Tn']=(d.T_C-25.)/1000.
    d['Sa']=.5*d.S*np.clip(1.-d.R,0.,2.);d['Smean']=.5*d.S*(1.+d.R)
    return apply_physics_drivers(d)


def revision_predict_state(state,frame):
    if state.get('residual_input_policy') != RESIDUAL_INPUT_POLICY:
        raise ValueError('冻结模型采用旧残差输入规则，请用本版本重新训练并导出模型。')
    assert_residual_preprocessor(state['preprocessor'])
    coef=np.asarray(state['coefficients']);gamma=state['gamma']
    base=predict_physical(frame,COMPETING_WB_SPEC,coef,gamma)
    augmented,_=residual_design_frame(frame,coef,gamma,state['family'])
    xv=state['preprocessor'].transform(augmented)
    if state['eta']==0.0:correction=np.zeros(len(frame))
    elif state['family']=='Matern_residual':
        correction=Matern(length_scale=state['kernel_length_scale'],nu=1.5)(xv,state['kernel_x'])@state['kernel_weights']
    else:correction=state['estimator'].predict(xv)
    distance=np.abs(frame.T_C.to_numpy(float)[:,None]-np.asarray(state['training_temperatures'])[None,:]).min(axis=1)
    if not state['temperature_support_enabled']:support=np.ones(len(frame))
    elif state['temperature_policy'] in ('soft','soft250'):support=np.exp(-(distance/250.)**2)
    else:raise ValueError('独立预测只支持本研究既定soft250温度支持，不默默替代其他规则')
    mu=base+state['eta']*support*correction
    if not np.isfinite(mu).all():raise FloatingPointError('独立预测出现非有限值')
    return mu


def revision_export_frozen(data,out,trees=400):


    import joblib
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=load_data(data)
    selection_seed=SEED+900000
    choice,oof,candidates=nested_select_hybrid(d,selection_seed,trees)
    keys=['family','gamma','strength','eta','min_samples_leaf','max_features',
          'ridge_alpha','kernel_length_scale','temperature_policy']
    settings={key:choice[key] for key in keys};settings['min_samples_leaf']=int(settings['min_samples_leaf'])
    mu,coef,meta=fit_hybrid_policy(d,d,COMPETING_WB_SPEC,selection_seed+1,trees,
        return_frozen_state=True,**settings)
    state=meta.pop('_frozen_state')
    
    np.testing.assert_allclose(revision_predict_state(state,d),mu,rtol=1e-11,atol=1e-11)
    scale=calibrate_aft_scale(oof);h80,h90=empirical_interval_halfwidths(oof)
    from datetime import timezone
    support_cols=['architecture','architecture_detail','env_class','T_C','R','frequency_Hz','S','UTS_MPa']
    manifest=dict(model_id=REVISION_MODEL_ID,created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=review_sha(__file__),data_sha256=review_sha(data),seed=selection_seed,trees=trees,
        environment_factors={'vacuum':0.,'air_or_oxygen':1.,'water_oxygen':water_multiplier_value(coef[5])},
        water_fit=water_fit_metadata(d,coef),
        settings=settings,residual_input_policy=RESIDUAL_INPUT_POLICY,
        residual_inputs=list(RESIDUAL_NUMERIC+RESIDUAL_CATEGORICAL),
        residual_physics_inputs=list(RESIDUAL_PHYSICS_NUMERIC),
        training_campaigns=sorted(d.campaign_id.unique().tolist()),
        training_records=len(d),interval_calibration='selected-candidate development OOF; selection and calibration share development records; not independent validation',
        claim='Frozen for prospective evaluation; not engineering design-life bounds',
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','pandas','joblib']})
    bundle=dict(manifest=manifest,state=state,scale=float(scale['sigma']),
        empirical_halfwidth80=float(h80),empirical_halfwidth90=float(h90),support=d[support_cols].copy())
    model_path=out/'frozen_model.joblib';joblib.dump(bundle,model_path,compress=3)
    loaded=joblib.load(model_path)
    np.testing.assert_allclose(revision_predict_state(loaded['state'],d),mu,rtol=1e-11,atol=1e-11)
    manifest['model_sha256']=review_sha(model_path)
    revision_json(out/'model_manifest.json',manifest)
    review_save(candidates,out,'final_training_selection');review_save(oof,out,'final_training_OOF')
    pd.DataFrame(columns=PROSPECTIVE_INPUT_COLUMNS+['material_batch','coating','waveform',
        'water_partial_pressure_kPa','oxygen_partial_pressure_kPa','gas_flow_rate','planned_stop_cycles']).to_csv(out/'new_conditions_TEMPLATE.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(columns=['specimen_id','Nf_cycles','is_runout','stop_reason','result_date']).to_csv(out/'new_outcomes_TEMPLATE.csv',index=False,encoding='utf-8-sig')
    (out/'独立实验使用说明.txt').write_text(PROSPECTIVE_GUIDE,encoding='utf-8')
    order=pd.DataFrame(water_order_audit(loaded['state'],d))
    review_save(order,out,'environment_order_check')
    revision_json(out/'export_verification.json',dict(status='passed',
        residual_input_policy=RESIDUAL_INPUT_POLICY,
        environment_order_checked_conditions=len(order),
        environment_order_reversals=int(order.total_order_reversal.sum()),
        checks='fit-vs-frozen and serialized-reload prediction equality',training_records=len(d),
        independent_experiments_evaluated=0))
    print(f'已保存冻结模型和空白实验模板：{out}',flush=True)


def revision_domain_flags(bundle,frame):
    support=bundle['support'];flags=[]
    for _,row in frame.iterrows():
        issues=[]
        for c in ['architecture','architecture_detail','env_class']:
            if row[c] not in set(support[c]):issues.append('unseen_'+c)
        for c in ['T_C','R','frequency_Hz','S','UTS_MPa']:
            if row[c]<support[c].min() or row[c]>support[c].max():issues.append('outside_'+c)
        matched=support.loc[(support.architecture_detail==row.architecture_detail)&
            (support.env_class==row.env_class)&np.isclose(support.T_C,row.T_C)&
            np.isclose(support.R,row.R)&np.isclose(support.frequency_Hz,row.frequency_Hz)]
        if matched.empty:issues.append('unseen_joint_conditions')
        elif row.S<matched.S.min() or row.S>matched.S.max():issues.append('outside_matched_stress_range')
        flags.append('|'.join(issues) if issues else 'within_recorded_covariate_support_not_material_validation')
    return flags


def revision_write_predictions(model,input_path,out):
    import joblib
    from datetime import timezone
    model=Path(model);side=model.with_name('model_manifest.json')
    if not side.is_file():raise FileNotFoundError('冻结模型缺少同目录model_manifest.json')
    manifest=json.loads(side.read_text(encoding='utf-8'))
    if manifest.get('model_id')!=REVISION_MODEL_ID or manifest.get('model_sha256')!=review_sha(model):
        raise ValueError('模型版本或文件哈希与说明不一致')
    for name,version in manifest['versions'].items():
        if importlib.metadata.version(name)!=version:
            raise RuntimeError(f'冻结模型要求{name}=={version}；请在原训练环境预测，避免反序列化差异。')
    
    bundle=joblib.load(model)
    raw=revision_read_table(input_path)
    for c in ('Nf_cycles','is_runout','logN','event_observed'):
        if c in raw and raw[c].notna().any():raise ValueError('预测输入包含实验结果，请仅提供实验前条件表')
    frame=revision_prediction_frame(raw)
    old=set(frame.campaign_id)&set(manifest['training_campaigns'])
    if old:raise ValueError(f'新试验组ID与训练数据重叠：{sorted(old)}')
    mu=revision_predict_state(bundle['state'],frame)
    result=raw.copy();result['specimen_id']=frame.specimen_id;result['campaign_id']=frame.campaign_id
    result['model_id']=REVISION_MODEL_ID;result['pred_logN']=mu;result['scale']=bundle['scale']
    result['support_status']=revision_domain_flags(bundle,frame)
    for key,value in bundle['state'].get('water_fit',{}).items():result[key]=value
    result['predicted_median_cycles']=np.power(10.,np.clip(mu,-300,300))
    for level in (80,90):
        half=bundle[f'empirical_halfwidth{level}']
        result[f'empirical_lower{level}_logN']=mu-half;result[f'empirical_upper{level}_logN']=mu+half
        p=(1-level/100)/2
        result[f'prob_lower{level}_logN']=review_quantile(mu,bundle['scale'],p)
        result[f'prob_upper{level}_logN']=review_quantile(mu,bundle['scale'],1-p)
    for c in list(result.columns):
        if c.endswith('_logN') and ('lower' in c or 'upper' in c):
            result[c.replace('_logN','_cycles')]=np.power(10.,np.clip(result[c],-300,300))
    result['cycle_conversion_clipped']=np.abs(mu)>300
    for c in result:
        if c.endswith('_logN') and ('lower' in c or 'upper' in c):result['cycle_conversion_clipped']|=result[c].abs()>300
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prediction_path=out/'predictions.csv';result.to_csv(prediction_path,index=False,encoding='utf-8-sig')
    revision_json(out/'prediction_manifest.json',dict(model_id=REVISION_MODEL_ID,
        created_utc=datetime.now(timezone.utc).isoformat(),model_sha256=review_sha(model),
        conditions_sha256=review_sha(input_path),predictions_sha256=review_sha(prediction_path),
        specimen_count=len(result),outcomes_used=False,
        note='Local timestamp/hash is a record, not external certification; archive before experiments.'))
    print(f'独立预测完成，未训练：{prediction_path}',flush=True)


def revision_evaluate(predictions,outcomes,out):
    predictions=Path(predictions);side=predictions.with_name('prediction_manifest.json')
    if not side.is_file():raise FileNotFoundError('需使用实验前生成的预测及同目录prediction_manifest.json')
    manifest=json.loads(side.read_text(encoding='utf-8'))
    if manifest['predictions_sha256']!=review_sha(predictions) or manifest['model_id']!=REVISION_MODEL_ID:
        raise ValueError('预测文件与保存时的哈希或模型版本不一致')
    pred=revision_read_table(predictions);obs=revision_read_table(outcomes)
    required={'specimen_id','Nf_cycles','is_runout'}
    if not required.issubset(obs) or obs.empty:raise ValueError('实验结果需填写specimen_id、Nf_cycles、is_runout')
    for d in (pred,obs):
        if d.specimen_id.isna().any():raise ValueError('specimen_id不能空缺')
        d['specimen_id']=d.specimen_id.astype(str).str.strip()
        if d.specimen_id.duplicated().any() or d.specimen_id.eq('').any():raise ValueError('specimen_id需非空且唯一')
    if not set(obs.specimen_id).issubset(pred.specimen_id):raise ValueError('存在没有预先预测的试样')
    obs['Nf_cycles']=pd.to_numeric(obs.Nf_cycles,errors='raise')
    if not np.isfinite(obs.Nf_cycles).all() or (obs.Nf_cycles<=0).any():raise ValueError('寿命或停止循环数必须为有限正数')
    obs['is_runout']=parse_bool(obs.is_runout,'is_runout')
    
    keep=['specimen_id','Nf_cycles','is_runout']+[c for c in ['stop_reason','result_date'] if c in obs]
    joined=pred.merge(obs[keep],on='specimen_id',validate='one_to_one')
    joined['is_exact']=~joined.is_runout;joined['logN']=np.log10(joined.Nf_cycles)
    joined['row_nll']=review_row_losses(joined.logN,joined.pred_logN,joined.scale,joined.is_exact)
    joined['exact_error_logN']=np.where(joined.is_exact,joined.pred_logN-joined.logN,np.nan)
    for kind in ('empirical','prob'):
        for level in (80,90):
            joined[f'{kind}_covered{level}_exact_only']=np.where(joined.is_exact,
                (joined.logN>=joined[f'{kind}_lower{level}_logN'])&(joined.logN<=joined[f'{kind}_upper{level}_logN']),np.nan)
    summary=[]
    for aggregation in ('record_equal','campaign_equal'):
        w=np.ones(len(joined)) if aggregation=='record_equal' else 1/joined.groupby('campaign_id').specimen_id.transform('size').to_numpy(float)
        w=w/w.sum();exact=joined.loc[joined.is_exact].copy()
        row=dict(aggregation=aggregation,n_evaluated=len(joined),n_pending=len(pred)-len(joined),
            n_campaigns=joined.campaign_id.nunique(),n_exact=len(exact),n_runout=int(joined.is_runout.sum()),
            joint_NLL=float(np.dot(w,joined.row_nll)),
            coverage_scope='exact failures only; runout latent-life coverage unknown')
        if len(exact):
            ew=np.ones(len(exact)) if aggregation=='record_equal' else 1/exact.groupby('campaign_id').specimen_id.transform('size').to_numpy(float)
            ew=ew/ew.sum();err=exact.exact_error_logN.to_numpy(float);y=exact.logN.to_numpy(float)
            sse=float(np.dot(ew,err**2));sst=float(np.dot(ew,(y-np.dot(ew,y))**2))
            row.update(RMSE=sse**.5,MAE=float(np.dot(ew,np.abs(err))),R2=1-sse/sst if sst>0 else np.nan,
                F5=float(np.dot(ew,np.abs(err)<=LOG5)))
            for kind in ('empirical','prob'):
                for level in (80,90):row[f'{kind}_coverage{level}_exact_only']=float(np.dot(ew,exact[f'{kind}_covered{level}_exact_only']))
        summary.append(row)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    review_save(joined,out,'independent_evaluation_rows');review_save(pd.DataFrame(summary),out,'independent_evaluation_summary')
    revision_json(out/'evaluation_manifest.json',dict(model_id=REVISION_MODEL_ID,
        predictions_sha256=review_sha(predictions),outcomes_sha256=review_sha(outcomes),
        model_refitted=False,n_pending=len(pred)-len(joined),
        limitation='No automatic certification of blinded timing, censoring independence or engineering design reliability.'))
    print(f'独立实验评价完成，未训练：{out}',flush=True)


def revision_external_cli(args,parser):
    if args.predict and args.evaluate:parser.error('--predict与--evaluate不可同时使用')
    if args.check or args.reviewer_only or args.baseline or args.data or args.censor_evidence:parser.error('独立预测/评价不能与主分析参数混用')
    if args.predict and (not args.model or args.predictions):parser.error('--predict须配合--model')
    if args.evaluate and (not args.predictions or args.model):parser.error('--evaluate须配合--predictions')
    out=(args.out or Path(__file__).resolve().parent/'runs'/('predict_' if args.predict else 'evaluate_')/
         datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if out.exists() and any(out.iterdir()):parser.error('为保护预先保存的预测，请指定新的空输出目录')
    if args.predict:revision_write_predictions(args.model,args.predict,out)
    else:revision_evaluate(args.predictions,args.evaluate,out)


PROSPECTIVE_GUIDE = f'''独立实验使用说明

本目录的模型由原候选池和原内层选择规则，在{EXPECTED_VERIFIED_ROWS}条文献记录上最终拟合。
它与主分析的五折泛化评价不同。新增的最终拟合不用于改写五折结果。
区间由选中配置的开发集OOF校准，选型与校准共用开发数据，不能称为独立验证或工程设计下限。
环境倍率在训练内估计，各环境共享残差修正，因此完整预测继承物理主干的环境顺序。beta=0时允许空气与水氧预测寿命相同。

1. 实验前：复制new_conditions_TEMPLATE.csv并填写，不填寿命和runout结果。
architecture填写详细编织方式（与训练表原architecture字段一致），arch_group填粗类别（如2D、2.5D、3D）。
env_class只能填写air_or_oxygen、water_oxygen、vacuum。T_C单位℃，应力/UTS为MPa，频率为Hz。
UTS应来自独立拉伸表征并记录采用的温度/批次依据，不能从本次疲劳结果反推。
新specimen_id、campaign_id必须非空。记录涂层、材料批次、波形、分压、气流和停止规则。
下面命令路径请换成你的实际路径；保持同一Python依赖环境，仅加载自己生成的模型。
python run.py --predict "新实验条件.csv" --model "本目录/frozen_model.joblib" --out "新预测目录"
保留生成的predictions.csv及prediction_manifest.json。不要覆盖。先保存预测，再获得实验结果。
本地时间戳和哈希便于核对，但不是第三方的盲测时间证明。

2. 实验后：复制new_outcomes_TEMPLATE.csv，填specimen_id、Nf_cycles、is_runout、停止原因和日期。
断裂填False；到停止循环数仍未断填True，Nf_cycles填实际停止循环数。
python run.py --evaluate "新实验结果.csv" --predictions "新预测目录/predictions.csv" --out "新评价目录"
程序不训练或调参，支持部分完成试样，并报告尚未完成数量。
runout只进入生存评分，不进入断裂误差，也不冒充双侧预测区间覆盖率。

3. 试验分工：空气试验检验新材料批次上的温度/频率/应力预测，不能标定水氧倍率。
空气—水氧差异需要同批材料、相同温度/频率/R/应力的对照，并记录气氛条件。
1300℃、3Hz、R=0.1可作为衔接已有水氧数据的候选条件，不是强制试验方案。
多个应力级、重复件及跨温度/批次数据用于区分离散性和环境效应；样本量另按资源和目标精度确定。
如果用新实验重新标定参数，必须更新模型版本，并另留独立试验组评价，不能继续称其为原模型盲测。
当前覆盖率不构成工程设计资格；需跨批次独立验证、覆盖率不确定性及目标可靠度下的验证。
'''

