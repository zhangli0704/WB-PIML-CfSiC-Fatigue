def water_mask(frame):
    return frame.env_class.astype(str).str.lower().str.contains('water|steam',regex=True).to_numpy(bool)


def water_multiplier_value(beta):
    
    beta=float(beta)
    return float(10.**beta) if np.isfinite(beta) and beta<=300 else None


def water_training_policy(train):
    requested=bool(train['fit_water_multiplier'].iloc[0]) if 'fit_water_multiplier' in train else False
    if 'fit_water_multiplier' in train and train.fit_water_multiplier.nunique()!=1:
        raise ValueError('A training fit must use one water-parameter policy')
    wet=water_mask(train)
    air=train.env_class.astype(str).str.lower().isin(['air_or_oxygen','air','oxygen']).to_numpy(bool)
    active_channel=train.environment_exposure.to_numpy(float)>0
    wet=wet&active_channel;air=air&active_channel
    upper=None
    if 'water_log10_upper' in train:
        bounds=train.water_log10_upper.dropna().unique()
        if len(bounds)>1:raise ValueError('Mixed computational bounds within a training fit')
        if len(bounds):upper=float(bounds[0])
    active=requested and bool(wet.any()) and bool(air.any())
    status=('estimated_training_only_identifiability_not_guaranteed' if active else
            'unidentified_no_water_reference_beta0' if requested and not wet.any() else
            'unidentified_no_air_reference_beta0' if requested and not air.any() else 'fixed_weight_control')
    return dict(requested=requested,active=active,status=status,upper=upper,
        n_water=int(wet.sum()),n_air=int(air.sum()),
        n_water_campaigns=int(train.loc[wet,'campaign_id'].nunique()) if 'campaign_id' in train else 0,
        n_air_campaigns=int(train.loc[air,'campaign_id'].nunique()) if 'campaign_id' in train else 0)


def water_log_adjustment(frame,coef):
    if len(coef)<6:return np.zeros(len(frame))
    enabled=(frame.fit_water_multiplier.to_numpy(bool) if 'fit_water_multiplier' in frame else np.ones(len(frame),bool))
    return np.where(water_mask(frame)&enabled,float(coef[5]),0.)


def water_fit_metadata(train,coef):
    policy=water_training_policy(train)
    beta=float(coef[5]) if len(coef)==6 else 0.
    if not policy['requested']:
        wet=water_mask(train)
        fixed=float(train.loc[wet,'environment_exposure'].iloc[0]) if wet.any() else None
        multiplier=fixed
    else:multiplier=water_multiplier_value(beta)
    return dict(water_parameter_status=policy['status'],water_parameter_estimated=policy['active'],
        water_log10_multiplier=beta if policy['requested'] else None,water_multiplier=multiplier,
        water_training_records=policy['n_water'],water_training_campaigns=policy['n_water_campaigns'],
        air_training_records=policy['n_air'],air_training_campaigns=policy['n_air_campaigns'],
        water_lower_boundary=bool(policy['active'] and beta<=1e-6),
        water_log10_computational_upper=policy['upper'],
        water_upper_boundary=bool(policy['active'] and policy['upper'] is not None and beta>=policy['upper']-1e-6),
        water_outer_responses_used=False)


def water_fit_competing(train,gamma,config,classic,predictor,nll_function,decrement_function,
                        likelihood_options=None,fixed_beta=None,return_diagnostics=False,
                        penalty_free=False):


    policy=water_training_policy(train)
    if fixed_beta is not None and (not np.isfinite(fixed_beta) or fixed_beta<0):
        raise ValueError('Fixed profile beta must be finite and nonnegative')
    free_beta=policy['active'] and fixed_beta is None
    emit_beta=policy['requested'] or fixed_beta is not None
    ca,cb=float(classic[0]),float(classic[1])
    bounds=[(ca-config.intercept_half_width,ca+config.intercept_half_width),
        (max(config.walker_min,cb-config.walker_half_width),min(config.walker_max,cb+config.walker_half_width)),
        (config.environment_ratio_min,config.environment_ratio_max),
        (config.environment_stress_min,config.environment_stress_max),(config.arrhenius_min,config.arrhenius_max)]
    if free_beta:bounds.append((0.,policy['upper']))
    bounds.append((math.log(.08),math.log(4.)))
    def decode(parameter):
        coef=np.asarray(parameter[:5],float)
        if emit_beta:
            beta=parameter[5] if free_beta else 0. if fixed_beta is None else fixed_beta
            coef=np.r_[coef,beta]
        return coef,float(np.exp(parameter[-1]))
    def components(parameter):
        coef,sigma=decode(parameter)
        mu=predictor(train,coef,gamma)
        if not np.isfinite(mu).all():return 1e100,0.,0.
        nll=float(nll_function(train,mu,sigma,**(likelihood_options or {}))['joint_censored_NLL'])
        ridge=0. if penalty_free else float(config.ridge_strength)*((coef[0]-ca)**2+.1*(coef[1]-cb)**2+.01*coef[3]**2+.01*coef[4]**2)
        env=0. if penalty_free or config.environment_strength==0 else float(config.environment_strength)*float(np.mean(decrement_function(train,coef,gamma)**2))
        return nll,ridge,env
    def loss(parameter):
        values=components(parameter);total=sum(values)
        return total if np.isfinite(total) else 1e100
    starts=[]
    for k,m,q,sigma,beta in [(-2.,1.,4.,.8,0.),(-4.,2.,7.,1.2,math.log10(1.5)),(0.,4.,2.,1.,math.log10(5.))]:
        start=[ca,cb,k,m,q]+([beta] if free_beta else [])+[math.log(sigma)]
        starts.append(np.clip(start,[b[0] if b[0] is not None else -np.inf for b in bounds],
                                   [b[1] if b[1] is not None else np.inf for b in bounds]))
    results=[minimize(loss,x,method='L-BFGS-B',bounds=bounds,
        options={'maxiter':1200,'ftol':1e-12,'gtol':1e-8}) for x in starts]
    valid=[r for r in results if r.success and np.isfinite(r.fun) and r.fun<1e99]
    if not valid:
        
        finite=[r for r in results if np.isfinite(r.fun) and np.isfinite(r.x).all()]
        start=min(finite,key=lambda r:r.fun).x if finite else starts[0]
        retry=minimize(loss,start,method='Powell',bounds=bounds,
            options={'maxiter':6000,'xtol':1e-9,'ftol':1e-10})
        if retry.success and np.isfinite(retry.fun) and retry.fun<1e99:valid=[retry]
        else:raise RuntimeError('Water-enhancement fit did not converge: '+'; '.join(str(r.message) for r in results)+f'; retry={retry.message}')
    best=min(valid,key=lambda r:float(r.fun));coef,sigma=decode(best.x)
    if return_diagnostics:
        nll,ridge,env=components(best.x)
        return dict(coefficients=coef,sigma=sigma,objective=float(best.fun),mean_NLL=nll,
            ridge=ridge,environment_penalty=env,free_water_parameter=free_beta,
            policy=policy,successful_starts=len(valid),start_objective_spread=float(max(r.fun for r in valid)-min(r.fun for r in valid)))
    return coef


def water_order_audit(state,test):


    air=test.copy();air['env_class']='air_or_oxygen';air=apply_physics_drivers(air)
    wet=test.copy();wet['env_class']='water_oxygen';wet=apply_physics_drivers(wet)
    a=revision_predict_state(state,air);w=revision_predict_state(state,wet)
    pa=predict_physical(air,COMPETING_WB_SPEC,state['coefficients'],state['gamma'])
    pw=predict_physical(wet,COMPETING_WB_SPEC,state['coefficients'],state['gamma'])
    if np.any(pw>pa+1e-10):raise AssertionError('Constrained trunk reverses water/air direction')
    if np.any(w>a+1e-10):raise AssertionError('Complete model reverses water/air direction')
    np.testing.assert_allclose(w-pw,a-pa,rtol=0.,atol=1e-10,
        err_msg='Residual corrections must be shared across environment labels')
    return [dict(row_id=int(row.row_id),campaign_id=str(row.campaign_id),
        water_minus_air_trunk_logN=float(pw[i]-pa[i]),water_minus_air_total_logN=float(w[i]-a[i]),
        water_minus_air_residual_logN=float((w[i]-pw[i])-(a[i]-pa[i])),
        residual_input_policy=RESIDUAL_INPUT_POLICY,
        total_order_reversal=bool(w[i]>a[i]+1e-10),
        interpretation='counterfactual covariates; no observed paired lifetimes; shared residual preserves trunk ordering')
        for i,row in test.reset_index(drop=True).iterrows()]


def water_profile_one(df,result):


    held={p['row_id'] for p in result['predictions']}
    train=df.loc[~df.row_id.isin(held)].reset_index(drop=True)
    policy=water_training_policy(train);gamma=float(result['settings']['gamma'])
    config=replace(DEFAULT_COMPETING_CONFIG,environment_strength=float(result['settings']['strength']))
    rows=[]
    if not policy['active']:
        return [dict(fold=result['fold'],status=policy['status'],outer_test_responses_used=False,confidence_interval_claim=False)]
    classic=_fit_linear_physical(train,CLASSIC_WB_SPEC,gamma)
    estimated_beta=float(result['coefficients'][5])
    
    grid=sorted(set([0.,math.log10(1.25),math.log10(1.5),math.log10(2.),math.log10(3.),
                     math.log10(5.),1.,math.log10(30.),2.,estimated_beta]))
    for penalty_free in (False,True):
        reference=water_fit_competing(train,gamma,config,classic,_predict_competing_damage,
            joint_aft_nll,environmental_decrement,return_diagnostics=True,penalty_free=penalty_free)
        reference_beta=float(reference['coefficients'][5])
        for beta in sorted(set(grid+[reference_beta])):
            fitted=water_fit_competing(train,gamma,config,classic,_predict_competing_damage,
                joint_aft_nll,environmental_decrement,fixed_beta=beta,
                return_diagnostics=True,penalty_free=penalty_free)
            rows.append(dict(fold=result['fold'],profile_type='unpenalized_likelihood_with_existing_bounds' if penalty_free else 'original_penalized_objective',
                status='computed',beta=beta,water_multiplier=water_multiplier_value(beta),
                objective=fitted['objective'],mean_NLL=fitted['mean_NLL'],sigma=fitted['sigma'],
                reference_objective=reference['objective'],reference_beta=reference_beta,
                delta_mean_objective=fitted['objective']-reference['objective'],
                fixed_profile_better_than_free_fit=fitted['objective']<reference['objective']-1e-6,
                n_training=len(train),n_water=policy['n_water'],n_water_campaigns=policy['n_water_campaigns'],
                nuisance_log10_K=float(fitted['coefficients'][2]),nuisance_temperature=float(fitted['coefficients'][4]),
                nuisance_stress=float(fitted['coefficients'][3]),outer_test_responses_used=False,
                confidence_interval_claim=False,scope='conditional diagnostic; finite scan not a physical upper bound'))
    return rows


def water_analysis_diagnostics(df,results,out):
    if not any(r['variant']=='selected_pipeline' for r in results):return
    audit=[];orders=[]
    for r in results:
        audit.append(dict(variant=r['variant'],fold=r['fold'],**r['water_fit']))
        for row in r.get('environment_order_audit',[]):orders.append(dict(fold=r['fold'],**row))
    review_save(pd.DataFrame(audit),out,'water_parameter_by_fold')
    inner=[dict(variant=r['variant'],outer_fold=r['fold'],inner_fold=a['inner_fold'],**a['water_fit'])
           for r in results for a in r['audit']]
    review_save(pd.DataFrame(inner),out,'water_parameter_inner_folds')
    order=pd.DataFrame(orders);review_save(order,out,'water_air_counterfactual_order')
    summary=order.groupby('fold').agg(n_conditions=('row_id','size'),
        total_order_reversals=('total_order_reversal','sum'),
        maximum_trunk_water_minus_air=('water_minus_air_trunk_logN','max'),
        maximum_total_water_minus_air=('water_minus_air_total_logN','max')).reset_index()
    review_save(summary,out,'water_air_order_summary')
    reference=[r for r in results if r['variant']=='selected_pipeline']
    print('水氧参数诊断：只使用5个外层训练集，剖面拟合不参与主模型选择...',flush=True)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        parts=Parallel(n_jobs=min(WORKERS,len(reference)),return_as='generator')(
            delayed(water_profile_one)(df,r) for r in reference)
        profiles=[]
        for part in parts:
            profiles.extend(part);print(f'水氧剖面完成：外层折 {part[0]["fold"]}',flush=True)
    review_save(pd.DataFrame(profiles),out,'water_identifiability_profiles')
    a=pd.DataFrame(audit);base=a.loc[a.variant.eq('selected_pipeline')].set_index('fold')
    bounds=[]
    for n in ('water_cap_10','water_cap_100'):
        for _,row in a.loc[a.variant.eq(n)].iterrows():
            b=base.loc[row['fold']]
            bounds.append(dict(variant=n,fold=row['fold'],unbounded_beta=b.water_log10_multiplier,
                capped_beta=row.water_log10_multiplier,
                beta_change=row.water_log10_multiplier-b.water_log10_multiplier,
                cap_reached=row.water_upper_boundary,scope='computational cap dependency, not parameter selection'))
    review_save(pd.DataFrame(bounds),out,'water_computational_bound_sensitivity')
    (Path(out)/'水氧参数诊断说明.txt').write_text(WATER_DIAGNOSTIC_GUIDE,encoding='utf-8')


def water_verify_outputs(out,df):
    a=pd.read_csv(Path(out)/'water_parameter_by_fold.csv')
    selected=a.loc[a.variant.eq('selected_pipeline')]
    if len(selected)!=5 or selected.water_log10_multiplier.isna().any():raise AssertionError('Missing learned water estimates')
    if selected.water_log10_multiplier.lt(-1e-10).any():raise AssertionError('Water direction constraint failed')
    if a.water_outer_responses_used.any():raise AssertionError('Water estimation used outer responses')
    profile=pd.read_csv(Path(out)/'water_identifiability_profiles.csv')
    if profile.fold.nunique()!=5 or profile.outer_test_responses_used.any():raise AssertionError('Profile partition mismatch')
    if profile.confidence_interval_claim.any():raise AssertionError('Profiles must not claim unsupported confidence intervals')
    order=pd.read_csv(Path(out)/'water_air_counterfactual_order.csv')
    if set(order.row_id)!=set(df.row_id) or not order.row_id.is_unique:raise AssertionError('Counterfactual audit incomplete')
    if order.water_minus_air_trunk_logN.gt(1e-10).any():raise AssertionError('Trunk order violation')
    if order.water_minus_air_total_logN.gt(1e-10).any():raise AssertionError('Complete-model order violation')
    if order.total_order_reversal.any():raise AssertionError('Environment audit reports an order reversal')
    np.testing.assert_allclose(order.water_minus_air_residual_logN,0.,rtol=0.,atol=1e-10)


WATER_DIAGNOSTIC_GUIDE = '''水氧系数说明
主模型：E(vacuum)=0，E(air)=1，E(water)=10**beta，beta>=0。
它与exp(delta)等价（delta=ln(10)*beta），只增加一个物理参数，不扩大残差候选池。
该倍率是相同其他工况下水氧/空气环境损伤通道的比值，不是总损伤比或寿命缩短倍数。
air_or_oxygen是原数据库合并类别，当前编码不能单独识别空气与纯氧的差异。
beta在每个对应训练子集内与原物理参数及删失尺度一起拟合；外层寿命不用于选择或拟合。
没有新增beta正则化，也不假定1.5为中心先验。原有物理参数约束和惩罚仍保留。
主拟合beta无有限上界。10和100分别代表一个、两个数量级的额外计算域对照，不是物理边界。
水氧训练样本/空气参考缺失时，beta不可估计，固定为0作为明确的参考假设，绝不称作学得倍率。
包括无水氧训练数据的LOEO水氧留出，必须结合情景分析解释，不能称作学会了水氧外推。

water_parameter_by_fold.csv：每折估计及训练支持、边界状态。
water_identifiability_profiles.csv：只使用外层训练集，固定beta后重拟合其他物理系数和尺度。
同时报告原惩罚目标剖面，以及保留原干扰参数边界但去掉惩罚的似然剖面；不重新拟合残差。
扫描点只是诊断坐标，不是主模型可选权重。平坦或广阔的低目标区域提示弱辨识；不自动换算95%区间。
若fixed_profile_better_than_free_fit为True，说明某固定点找到了更低目标，需要关注局部极值；程序不偷偷替换主模型。
水氧只有两个试验组且温度/来源混杂，折间离散性和剖面不能证明独立因果环境效应。
water_computational_bound_sensitivity.csv：限制计算域后beta变化及触顶情况。
water_air_counterfactual_order.csv：同一协变量下改变环境类别的情景检查，不是真实配对试验。
同条件下各环境共享残差修正，完整模型继承物理主干的环境顺序；主模型beta>=0保证水氧寿命不长于空气，beta=0允许相等。
该约束是本文的物理先验，不是实测气氛效应；程序检查完整预测顺序及共享残差，发现逆转即停止，不裁剪预测。

固定1、1.5及其他旧权重对照全部保留，不挑外层分数最好的倍率。
估计等于1不等于证明真实损伤相同；估计很大或依赖上界，也不等于发现巨大真实损伤。
独立的同批材料、同温度/应力/频率/R气氛对照才是校准环境倍率的后续证据。
'''


STRICT_REVIEW_VERSION='strict-campaign-holdout-20261008-v2-shared-environment-residual'
STRICT_MODELS={'WB-PIML','WB','ExtraTrees','ET-Walker','ML-Ens','ML-Strong',*SIMPLE_CONTROL_MODELS}


def strict_holdout_plan(df,overlap):


    columns={'LOSO':'source_id','LOAO':'architecture','LOTO':'temp_bin','LOEO':'env_class'}
    rows=[]
    for task_index,rec in enumerate(overlap.loc[overlap.scenario.isin(columns)].to_dict('records'),1):
        scenario=rec['scenario'];column=columns[scenario];level=str(rec['holdout_group'])
        te=np.flatnonzero(df[column].astype(str).eq(level).to_numpy())
        original=np.flatnonzero(~df[column].astype(str).eq(level).to_numpy())
        excluded=set(df.iloc[te].campaign_id)
        tr=np.array([i for i in original if df.iloc[i].campaign_id not in excluded],dtype=int)
        train=df.iloc[tr].reset_index(drop=True);test=df.iloc[te].reset_index(drop=True)
        review_require_disjoint(train,test)
        seed=SEED+700000+1000000*list(columns).index(scenario)
        selection_seed=seed+10000*int(rec['fold'])+500
        reason='';inner=[]
        if train.campaign_id.nunique()<4:reason='fewer_than_four_training_campaigns'
        elif not train.is_exact.any() or not test.is_exact.any():reason='no_exact_failures_for_existing_protocol'
        else:
            try:
                splits,_=grouped_folds(train,4,selection_seed)
                for k,(a,b) in enumerate(splits,1):
                    review_require_disjoint(train.iloc[a],train.iloc[b])
                    inner.append(dict(inner_fold=k,fit_ids=train.iloc[a].row_id.astype(int).tolist(),
                        validation_ids=train.iloc[b].row_id.astype(int).tolist()))
            except (RuntimeError,ValueError) as exc:reason='inner_partition_not_feasible: '+str(exc)
        
        action='not_estimable' if reason else 'refit' if len(tr)!=len(original) else 'reuse_identical_partition'
        rows.append(dict(task_id=f'{scenario}_{int(rec["fold"]):02d}',scenario=scenario,
            fold=int(rec['fold']),holdout_group=level,holdout_column=column,seed=seed,
            action=action,reason=reason,n_train_original=len(original),n_train=len(tr),n_test=len(te),
            n_train_campaigns=int(train.campaign_id.nunique()),n_removed=len(original)-len(tr),
            train_indices=tr.tolist(),test_indices=te.tolist(),train_ids=train.row_id.astype(int).tolist(),
            test_ids=test.row_id.astype(int).tolist(),inner_partitions=inner,
            train_campaigns=sorted(train.campaign_id.unique().tolist()),test_campaigns=sorted(test.campaign_id.unique().tolist())))
    return rows


def strict_holdout_task(df,task,trees):
    import traceback
    try:
        tr=np.asarray(task['train_indices'],int);te=np.asarray(task['test_indices'],int)
        review_require_disjoint(df.iloc[tr],df.iloc[te])
        with threadpool_limits(limits=1):
            point,choices,prob,cal=_external_fold_task(df,task['fold'],tr,te,task['holdout_group'],
                task['holdout_column'],task['seed'],trees,task['scenario'],1)
        return dict(task_id=task['task_id'],status='refitted',point=pd.concat(point,ignore_index=True),
            selection=pd.DataFrame(choices),probability=pd.concat(prob,ignore_index=True),
            calibration=pd.concat(cal,ignore_index=True),error='')
    except Exception:
        
        return dict(task_id=task['task_id'],status='fit_failed',error=traceback.format_exc())


def strict_scoring_frame(frame):
    d=frame.copy()
    d['row_nll']=distribution_nll(d.prob_family,d.logN,d.prob_location,d.prob_scale,d.is_exact)
    for level in (80,90):
        tail=(1-level/100)/2
        d[f'prob_lower{level}']=distribution_quantile(d.prob_family,d.prob_location,d.prob_scale,tail)
        d[f'prob_upper{level}']=distribution_quantile(d.prob_family,d.prob_location,d.prob_scale,1-tail)
    return d


def strict_accept_result(df,task,result):
    p=result['probability'];c=result['calibration'];s=result['selection']
    if set(p.model)!=STRICT_MODELS:raise AssertionError('Strict comparison model coverage differs')
    train=set(task['train_ids']);test=set(task['test_ids'])
    if not set(c.row_id).issubset(train) or set(c.row_id)&test:
        raise AssertionError('Strict interval calibration includes held-out rows')
    if not STRICT_MODELS.issubset(set(c.model)):raise AssertionError('Strict calibration models missing')
    if c.outer_test_used_for_calibration.astype(bool).any():raise AssertionError('Strict calibration admits outer responses')
    for _,part in c.groupby('model'):
        if part.inner_fold.nunique()!=4:raise AssertionError('Strict calibration must retain four campaign folds')
        if set(part.row_id)!=train or part.row_id.duplicated().any():
            raise AssertionError('Strict calibration does not cover development set exactly once')
    for _,part in p.groupby('model'):
        if set(part.row_id)!=test or part.row_id.duplicated().any():raise AssertionError('Strict prediction coverage mismatch')
    for f in (p,c,s):
        f['strict_task_id']=task['task_id'];f['validation_scope']='factor_and_campaign_disjoint'
        f['strict_result_origin']=result['status'];f['campaign_disjoint_by_design']=True
    s['campaign_overlap']=False;s['holdout_campaign_disjoint']=True;s['n_overlapping_campaigns']=0
    return strict_scoring_frame(p),c,s


def strict_holdout_review(df,tables,out,args):
    out=Path(out);dest=out/'strict_holdout';dest.mkdir(exist_ok=True)
    overlap=pd.read_csv(out/'holdout_overlap.csv')
    plan=strict_holdout_plan(df,overlap);revision_json(dest/'partition_plan.json',plan)
    public=[{k:v for k,v in t.items() if k not in {'train_indices','test_indices','inner_partitions','train_ids','test_ids','train_campaigns','test_campaigns'}} for t in plan]
    review_save(pd.DataFrame(public),dest,'task_plan')
    indexed={t['task_id']:t for t in plan};results=[];status=[]
    for t in plan:
        if t['action']=='not_estimable':
            status.append(dict(task_id=t['task_id'],status='not_estimable',reason=t['reason']));continue
        if t['action']=='reuse_identical_partition':
            def subset(key):
                f=tables[key];return f.loc[f.scenario.eq(t['scenario']) & f.fold.eq(t['fold'])].copy()
            results.append(dict(task_id=t['task_id'],status='reused_identical_partition',
                probability=subset('Prob_Predictions'),calibration=subset('Prob_Calibration'),
                selection=subset('External_Selection'),error=''))
    todo=[t for t in plan if t['action']=='refit']
    print(f'严格因素留出：{len(todo)}项重新训练；{len(results)}项原划分已隔离；{len(status)}项条件不足。',flush=True)
    if todo:
        with parallel_config(backend='loky',inner_max_num_threads=1):
            stream=Parallel(n_jobs=min(args.jobs,len(todo)),return_as='generator')(
                delayed(strict_holdout_task)(df,t,args.trees) for t in todo)
            for res in stream:
                results.append(res);print(f'严格留出 {res["task_id"]}: {res["status"]}',flush=True)
    predictions=[];calibrations=[];selections=[]
    for res in results:
        t=indexed[res['task_id']]
        if res['status']=='fit_failed':
            status.append(dict(task_id=t['task_id'],status='fit_failed',reason=res['error']))
            (dest/(t['task_id']+'_error.txt')).write_text(res['error'],encoding='utf-8');continue
        p,c,s=strict_accept_result(df,t,res)
        predictions.append(p);calibrations.append(c);selections.append(s)
        status.append(dict(task_id=t['task_id'],status=res['status'],reason=''))
    review_save(pd.DataFrame(status),dest,'task_status')
    if not predictions:raise RuntimeError('No strict holdout task could be evaluated; see task_status.csv')
    pred=pd.concat(predictions,ignore_index=True);cal=pd.concat(calibrations,ignore_index=True)
    review_save(pred,dest,'predictions');review_save(cal,dest,'calibration');review_save(pd.concat(selections,ignore_index=True),dest,'selection')
    metrics=[];comparisons=[];paired=[]
    base=strict_scoring_frame(tables['Prob_Predictions'].loc[lambda x:x.scenario.ne('Primary')])
    for (scenario,model),g in pred.groupby(['scenario','model']):
        b=base.loc[base.scenario.eq(scenario)&base.model.eq(model)&base.row_id.isin(g.row_id)].copy()
        for label,d in [('strict',g),('original_same_test_rows',b)]:
            metrics.append(dict(scenario=scenario,model=model,evaluation=label,n_tasks=d.fold.nunique(),
                population='estimable strict tasks only; excluded tasks remain in task_status',**review_primary_scores(d)))
        for metric in ('SB_RMSE','SB_MAE','SB_R2','joint_NLL'):
            comparisons.append(dict(scenario=scenario,model=model,metric=metric,
                strict_minus_original=review_primary_scores(g)[metric]-review_primary_scores(b)[metric],
                n_test_records=len(g),same_test_population=True))
    for scenario,g in pred.groupby('scenario'):
        a=g.loc[g.model.eq('WB-PIML')]
        for model,b in g.loc[g.model.ne('WB-PIML')].groupby('model'):
            if a.campaign_id.nunique()<2:continue
            ci=review_paired_ci(a,b,args.bootstrap,SEED+7100000+len(paired))
            ci['scenario']=scenario;ci['proposed']='WB-PIML';ci['comparator']=model
            ci['direction']='WB-PIML minus comparator; negative favors WB-PIML for errors/NLL'
            paired.append(ci)
    review_save(pd.DataFrame(metrics),dest,'metrics_matched_population')
    review_save(pd.DataFrame(comparisons),dest,'strict_vs_original')
    review_save(pd.concat(paired,ignore_index=True) if paired else pd.DataFrame(columns=['scenario','metric']),dest,'paired_model_CI')
    groupmetrics=[]
    for (scenario,level,model),g in pred.groupby(['scenario','holdout_group','model']):
        groupmetrics.append(dict(scenario=scenario,holdout_group=level,model=model,**review_primary_scores(g)))
    review_save(pd.DataFrame(groupmetrics),dest,'holdout_metrics')
    
    coverage=[]
    for (scenario,model),g in pred.groupby(['scenario','model']):
        exact=g.loc[g.is_exact]
        for level in (80,90):
            lo,hi=f'lower{level}',f'upper{level}'
            if lo not in exact or hi not in exact:continue
            valid=exact[lo].notna()&exact[hi].notna();value=np.nan
            if len(exact) and valid.all():
                w=analysis_weights(exact);w/=w.sum();value=float(np.dot(w,(exact.logN>=exact[lo])&(exact.logN<=exact[hi])))
            coverage.append(dict(scenario=scenario,model=model,nominal=level/100,coverage=value,
                n_exact=len(exact),n_with_interval=int(valid.sum()),scope='exact failures in estimable strict tasks only'))
    review_save(pd.DataFrame(coverage),dest,'empirical_coverage')
    summary=dict(version=STRICT_REVIEW_VERSION,n_tasks=len(plan),n_refitted=sum(s['status']=='refitted' for s in status),
        n_reused=sum(s['status']=='reused_identical_partition' for s in status),
        n_not_estimable=sum(s['status']=='not_estimable' for s in status),n_fit_failed=sum(s['status']=='fit_failed' for s in status),
        main_model_changed=False,selection_and_calibration='remaining training campaigns only',
        claim='Strictness of this split does not establish independent material-system or causal environmental validation.')
    summary['status']='incomplete_numerical_failures' if summary['n_fit_failed'] else 'completed_with_disclosed_ineligible_tasks' if summary['n_not_estimable'] else 'completed'
    revision_json(dest/'summary.json',summary)
    (dest/'使用说明.txt').write_text(STRICT_GUIDE,encoding='utf-8')


def strict_verify_outputs(out,df):
    dest=Path(out)/'strict_holdout';plan=json.loads((dest/'partition_plan.json').read_text(encoding='utf-8'))
    status=pd.read_csv(dest/'task_status.csv').set_index('task_id');p=pd.read_csv(dest/'predictions.csv');c=pd.read_csv(dest/'calibration.csv')
    if set(status.index)!={t['task_id'] for t in plan}:raise AssertionError('Strict tasks silently missing')
    for t in plan:
        train=df.loc[df.row_id.isin(t['train_ids'])];test=df.loc[df.row_id.isin(t['test_ids'])]
        review_require_disjoint(train,test)
        if train[t['holdout_column']].astype(str).eq(t['holdout_group']).any():raise AssertionError('Factor leaked into strict training')
        q=p.loc[p.strict_task_id.eq(t['task_id'])];s=str(status.loc[t['task_id'],'status'])
        if s in ('fit_failed','not_estimable'):
            if len(q):raise AssertionError('Unavailable task has fabricated predictions')
            continue
        for _,g in q.groupby('model'):
            if set(g.row_id)!=set(t['test_ids']) or g.row_id.duplicated().any():raise AssertionError('Strict test coverage incomplete')
        if set(q.model)!=STRICT_MODELS:raise AssertionError('Strict comparators missing')
        cc=c.loc[c.strict_task_id.eq(t['task_id'])]
        if not STRICT_MODELS.issubset(set(cc.model)):raise AssertionError('Strict calibration models missing')
        for _,g in cc.groupby('model'):
            if set(g.row_id)!=set(t['train_ids']) or g.row_id.duplicated().any():raise AssertionError('Strict calibration coverage incorrect')
            if g.inner_fold.nunique()!=4:raise AssertionError('Strict actual calibration is not four-fold')
            for _,v in g.groupby('inner_fold'):
                fitting=train.loc[~train.row_id.isin(v.row_id)]
                valid=train.loc[train.row_id.isin(v.row_id)]
                review_require_disjoint(fitting,valid)
                if not v.fit_row_ids_sha256.eq(probability_row_id_hash(fitting.row_id)).all():raise AssertionError('Strict actual fit trace mismatch')
                if not v.validation_row_ids_sha256.eq(probability_row_id_hash(valid.row_id)).all():raise AssertionError('Strict actual validation trace mismatch')
        for inner in t['inner_partitions']:
            if set(inner['fit_ids'])|set(inner['validation_ids'])!=set(t['train_ids']):raise AssertionError('Strict inner partition incomplete')
            review_require_disjoint(df.loc[df.row_id.isin(inner['fit_ids'])],df.loc[df.row_id.isin(inner['validation_ids'])])
    np.testing.assert_allclose(p.row_nll,distribution_nll(p.prob_family,p.logN,p.prob_location,p.prob_scale,p.is_exact),rtol=1e-10)
    revision_json(dest/'verification.json',dict(status='passed_for_available_tasks',
        n_fit_failed=int(status.status.eq('fit_failed').sum()),n_not_estimable=int(status.status.eq('not_estimable').sum()),
        meaning='Arithmetic and partition checks; not a certificate of predictive or scientific adequacy'))


def revision_censor_evidence(df,out,evidence_path=None):
    out=Path(out);run=df.loc[df.is_runout,['row_id','source_id','campaign_id','Nf_cycles','env_class','T_C']].copy()
    fields=['documented_stop_type','source_page_or_figure','verbatim_stop_rule','evidence_document','evidence_scope','reviewer_note']
    for f in fields:run[f]=''
    run['documented_stop_type']='unreviewed'
    evidence_hash=None
    if evidence_path is not None:
        evidence_path=Path(evidence_path)
        supplied=pd.read_csv(evidence_path).fillna('')
        if 'row_id' not in supplied or supplied.row_id.duplicated().any():raise ValueError('停止证据表需唯一row_id')
        if not set(supplied.row_id).issubset(set(run.row_id)):raise ValueError('停止证据表含非runout记录')
        allowed={'unreviewed','unknown','administrative_cycle_limit','damage_threshold','equipment_or_interruption','other_documented'}
        for _,row in supplied.iterrows():
            i=run.index[run.row_id.eq(row.row_id)][0]
            for identity in ('source_id','campaign_id'):
                if identity in supplied and str(row[identity]) and str(row[identity])!=str(run.loc[i,identity]):raise ValueError('停止证据表来源与记录不一致')
            for f in fields:
                if f in supplied:run.loc[i,f]=str(row[f]).strip()
            if run.loc[i,'documented_stop_type'] not in allowed:raise ValueError('未知停止规则类别；请保留unreviewed或unknown')
        evidence_hash=review_sha(evidence_path)
    complete=run[['source_page_or_figure','verbatim_stop_rule','evidence_document','evidence_scope']].ne('').all(axis=1)
    documented=~run.documented_stop_type.isin(['unreviewed','unknown'])
    run['evidence_fields_complete']=complete
    run['audit_status']=np.where(documented&complete,'documented_pending_source_authenticity_review',
        np.where(documented,'incomplete_evidence',run.documented_stop_type))
    run['conditional_independence_established']=False
    review_save(run,out,'censor_protocol_audit')
    review_save(run.groupby(['source_id','documented_stop_type','audit_status'],dropna=False).size().rename('n_records').reset_index(),out,'censor_evidence_gaps')
    if not (out/'censor_protocol_reviewed_TEMPLATE.csv').exists():review_save(run,out,'censor_protocol_reviewed_TEMPLATE')
    revision_json(out/'censor_evidence_status.json',dict(n_runout=len(run),n_documented_with_fields=int((complete&documented).sum()),
        n_unresolved=int((~(complete&documented)).sum()),evidence_sha256=evidence_hash,
        source_content_automatically_verified=False,conditional_independence_established=False))


def revision_interpretation_audit(df,tables,out):
    out=Path(out);path=out/'sensitivity_predictions.csv'
    lines=['# 返修证据与未完成事项','本报告区分计算完成与科学主张得到支持；不按外层分数改选模型。']
    if path.exists():
        p=pd.read_csv(path);a=p.loc[p.variant.eq('selected_pipeline')];b=p.loc[p.variant.eq('water_1.5')]
        if len(a) and len(b):
            d=a.merge(b[['row_id','pred_logN']],on='row_id',suffixes=('_new','_fixed15'),validate='one_to_one')
            d=d.merge(df[['row_id','env_class']],on='row_id',validate='one_to_one');d=d.loc[d.is_exact].copy()
            d['MSE_gain']=(d.pred_logN_fixed15-d.logN)**2-(d.pred_logN_new-d.logN)**2
            counts=d.groupby('campaign_id').row_id.transform('size');ng=d.campaign_id.nunique()
            d['weighted_MSE_gain']=d.MSE_gain/(counts*ng)
            total=float(d.weighted_MSE_gain.sum());fit=pd.read_csv(out/'water_parameter_by_fold.csv').query("variant=='selected_pipeline'").set_index('fold')
            d['water_training_records']=d['fold'].map(fit.water_training_records)
            d['water_test_without_water_training']=d.env_class.eq('water_oxygen')&d.water_training_records.eq(0)
            d['net_improvement_share']=d.weighted_MSE_gain/total if abs(total)>1e-12 else np.nan
            review_save(d,out,'environment_improvement_attribution_rows')
            summary=d.groupby(['campaign_id','env_class','fold'],dropna=False).agg(n_exact=('row_id','size'),
                mean_MSE_gain=('MSE_gain','mean'),weighted_MSE_gain=('weighted_MSE_gain','sum'),
                net_improvement_share=('net_improvement_share','sum'),water_training_records=('water_training_records','first')).reset_index()
            review_save(summary,out,'environment_improvement_attribution')
            share=float(d.loc[d.water_test_without_water_training,'weighted_MSE_gain'].sum()/total) if abs(total)>1e-12 else np.nan
            lines.append(f'固定1.5条件对照→当前方案：试验组平衡MSE净改善={total:.8f}；无水氧训练支持的水氧测试记录贡献比例={share:.2%}。这是净MSE变化分解，不是RMSE改善率，不是相对所有基线，更不是机制学习证据。')
            rows=[]
            for f in sorted(fit.index):
                a0=json.loads((out/'checkpoints'/f'selected_pipeline_fold{f}.json').read_text(encoding='utf-8'))
                b0=json.loads((out/'checkpoints'/f'water_1.5_fold{f}.json').read_text(encoding='utf-8'))
                rows.append(dict(fold=f,n_water_train=int(fit.loc[f,'water_training_records']),
                    physical_first5_max_difference=float(np.max(np.abs(np.array(a0['coefficients'][:5])-np.array(b0['coefficients'][:5])))),
                    residual_settings_equal=a0['settings']==b0['settings'],
                    interpretation='no water in training: held-out multiplier is a reference assumption, not an estimated water effect' if fit.loc[f,'water_training_records']==0 else 'training estimate subject to confounding'))
            review_save(pd.DataFrame(rows),out,'water_fallback_attribution')
    if (out/'water_air_counterfactual_order.csv').exists():
        q=pd.read_csv(out/'water_air_counterfactual_order.csv')
        lines.append(f'环境情景检查：{int(q.total_order_reversal.sum())}/{len(q)}个条件在完整预测中出现水氧寿命更长；不是实测错误率。环境共享残差使完整模型保持物理主干的环境顺序；检查发现逆转即报错。')
    ci=tables['Prob_Paired_CI'];ci=ci.loc[ci.scenario.eq('Primary')].copy()
    ci['PIML_minus_comparator']=-ci.estimate;ci['difference_CI95_low']=-ci.CI95_hi;ci['difference_CI95_high']=-ci.CI95_lo
    ci['CI_crosses_zero']=(ci.difference_CI95_low<=0)&(ci.difference_CI95_high>=0)
    review_save(ci,out,'primary_NLL_difference_CI')
    for name in ('WB-CD','WB-Residual','Basquin-Residual'):
        row=ci.loc[ci.comparator.eq(name)].iloc[0]
        lines.append(f'主验证NLL：WB-PIML减{name}={row.PIML_minus_comparator:+.6f}，95% CI [{row.difference_CI95_low:+.6f}, {row.difference_CI95_high:+.6f}]；负值有利于WB-PIML。条件bootstrap不包含反复模型开发不确定性，跨零不证明等效。')
    for _,row in tables['External_PI_Summary'].query("scenario=='LOTO'").iterrows():
        lines.append(f'原温区留出：名义{row.nominal:.0%}经验区间，断裂记录覆盖率{row.coverage:.2%}；不能解释为所有潜在寿命的无条件覆盖率。')
    if (out/'strict_holdout/summary.json').exists():
        v=json.loads((out/'strict_holdout/summary.json').read_text(encoding='utf-8'))
        lines.append(f'严格因素留出：重新拟合{v["n_refitted"]}项，复用相同隔离划分{v["n_reused"]}项，条件不足{v["n_not_estimable"]}项，数值失败{v["n_fit_failed"]}项。严格/原方案仅在相同可评估测试记录上比较，不以减少评价样本冒充性能提升。')
    c=json.loads((out/'censor_evidence_status.json').read_text(encoding='utf-8'))
    lines.append(f'runout停止规则：{c["n_runout"]}条，填写完整证据字段{c["n_documented_with_fields"]}条，未完成{c["n_unresolved"]}条。该信息表为可选元数据；无此表时全部runout仍按右删失信息参与拟合。分布及损失权重分析评估敏感性，不直接证明删失独立性。')
    lines.append('主模型采用环境共享残差，水氧系数仍在训练内估计；本次模型结构修改后全部结果重新生成。不会把固定1.5或学习倍率宣布为已验证物理常数。严格留出也不自动等于独立材料系统实验。')
    (out/'返修关键结论与缺口.md').write_text('\n\n'.join(lines),encoding='utf-8')


STRICT_GUIDE='''严格因素留出说明
主模型采用环境共享残差；权重训练估计、候选池及主验证划分保留。本目录是额外的因素+试验组隔离对照。
先确定原因素测试集，再从训练中排除这些测试试验组的所有记录。
有重叠且可评估任务重新内层选型、拟合并以剩余训练OOF校准区间；原本隔离的相同任务复用保存预测。
所有八个比较模型共用任务和训练/测试划分。不得仅删除旧结果中的重叠记录当成重训。
本额外对照要求至少四个训练试验组，并检查四折内层都有断裂记录；不足时明确不估计。
任务失败单独保存错误，不会自动放宽隔离或切换算法，也不伪造结果。
metrics_matched_population中的原/严格指标使用相同可评估测试记录。汇总不包含不可评估任务，必须一起报告覆盖范围。
外层同源异工况记录被排除后，差异包含训练量减少和材料支持变化，不能解释为纯粹的泄漏偏差估计。
本次验证设计为看到旧结果后的返修分析，不称为预注册。
停止规则证据可使用主脚本同目录或数据同目录的censor_protocol_reviewed.csv；也可--censor-evidence指定。
证据类别：unreviewed、unknown、administrative_cycle_limit、damage_threshold、equipment_or_interruption、other_documented。
若提供可选来源信息表，其字段包括row_id、documented_stop_type、source_page_or_figure、verbatim_stop_rule、evidence_document、evidence_scope。
记录页码和适用样本范围，不能由整百万循环数反推停止协议；文件字段完整并不代表原文已认证。
无该信息表时仍完整计算并保留全部runout；可选元数据的缺失不构成训练失败。条件独立删失作为模型假设说明。
'''

if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    single_main()

