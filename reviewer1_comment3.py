"""Isolated held-out zone-Shapley stability and reference sensitivity analysis."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.metadata,io,json,math,platform,time,warnings,zipfile
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata,spearmanr
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

VERSION='r1c3-1.0.1'
BASE_APP_BLOB='0e67a5bf9ab52c98c0f2296544da30a8595827bd'
ZONES=(('Amide I',1600,1700),('Amide II',1500,1600),('CH (1490-1430)',1430,1490),('CH/CO3',1400,1430),('CH (1350-1330)',1330,1350),('Amide III',1200,1300),('HPO4',1100,1200),('PO4',930,1100),('CO3',860,890))
GAPS='Unassigned (gaps)'
MEAN='Training mean';MEDIAN='Training median';EXTENDED='Training mean extended'
@dataclass(frozen=True)
class Config:
    spectra_per_animal:int=250
    seed:int=42
    explained_per_animal:int=10
    extended_per_animal:int=50
    donor_panels:int=3
    sheets:tuple[str,...]=('Raw',)
    models:tuple[str,...]=('Logistic Regression','XGBoost')
    def validate(self):
        if not 1<=self.explained_per_animal<=self.extended_per_animal<=self.spectra_per_animal:raise ValueError('Require 1 <= primary explained <= extended explained <= sampled spectra per animal.')
        if not 0<=self.seed<2**32:raise ValueError('Seed outside supported range.')
        if not 1<=self.donor_panels<=5:raise ValueError('Use 1–5 donor panels.')
        if not self.sheets or set(self.sheets)-{'Raw','2nd derivative'}:raise ValueError('Select Raw and/or 2nd derivative.')
        if not self.models or set(self.models)-{'Logistic Regression','XGBoost'}:raise ValueError('Select the manuscript LR and/or XGBoost models.')
        if len(set(self.models))!=len(self.models) or len(set(self.sheets))!=len(self.sheets):raise ValueError('Duplicate selections.')

def source_hash():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def identity(data,c):return hashlib.sha256(json.dumps({'input':hashlib.sha256(data).hexdigest(),'config':asdict(c),'engine':source_hash()},sort_keys=True).encode()).hexdigest()
def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.generic):return jsonable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    return repr(x)

def assign_zones(features):
    mapping={n:[] for n,_,_ in ZONES};mapping[GAPS]=[];audit=[]
    for i,w in enumerate(features):
        name=next((n for n,lo,hi in ZONES if lo<=float(w)<=hi),GAPS)
        mapping[name].append(i);audit.append({'Wavenumber':float(w),'Zone':name})
    names=[n for n in mapping if mapping[n]];idx=[np.array(mapping[n],int) for n in names]
    assert sorted(np.concatenate(idx).tolist())==list(range(len(features)))
    return names,idx,pd.DataFrame(audit)

def prepare(frame,c):
    c.validate();norm={str(k).strip().lower():k for k in frame}
    idc=next((norm[k] for k in ('sample id','animal id','mouse id') if k in norm),None)
    lbc=next((norm[k] for k in ('type','phenotype','class','label') if k in norm),None)
    if idc is None or lbc is None:raise ValueError('Require Sample ID and Type/phenotype columns.')
    if frame[[idc,lbc]].isna().any().any():raise ValueError('Missing animal ID or phenotype.')
    d=frame.copy();d[idc]=d[idc].astype(str).str.strip();d[lbc]=d[lbc].astype(str).str.strip().str.upper()
    if (d[idc]=='').any() or set(d[lbc])!={'WT','OIM'}:raise ValueError('Require nonblank animal IDs and WT/OIM labels.')
    if d.groupby(idc)[lbc].nunique().ne(1).any():raise ValueError('Conflicting phenotype within an animal.')
    if d.drop_duplicates(idc)[lbc].value_counts().min()<2:raise ValueError('Require at least two animals per phenotype.')
    fc=[]
    for k in d:
        if k in (idc,lbc):continue
        try:w=float(str(k).strip())
        except ValueError:continue
        if 800<=w<=1800:fc.append(k)
    if not fc or len({float(str(k)) for k in fc})!=len(fc):raise ValueError('Missing or duplicate fingerprint wavenumbers.')
    values=d[fc].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    if not np.isfinite(values).all():raise ValueError('Invalid fingerprint features; no imputation or row exclusion performed.')
    d[fc]=values;d=d.reset_index(drop=True);parts=[]
    for a,g in d.groupby(idc,sort=True):
        if len(g)<c.spectra_per_animal:raise ValueError(f'Animal {a} has too few spectra.')
        rs=(c.seed+int.from_bytes(hashlib.blake2b(str(a).encode(),digest_size=4).digest(),'big'))%(2**32-1)
        part=g.sample(n=c.spectra_per_animal,random_state=rs).copy();part['_source_row']=part.index+2;parts.append(part)
    sampled=pd.concat(parts,ignore_index=True).sample(frac=1,random_state=c.seed).reset_index(drop=True)
    return {'X':sampled[fc].to_numpy(float),'y':(sampled[lbc]=='OIM').to_numpy(int),'groups':sampled[idc].to_numpy(str),'rows':sampled['_source_row'].to_numpy(int),'features':[str(x) for x in fc]},pd.DataFrame({'Animal':sampled[idc],'Phenotype':sampled[lbc],'Source Excel row':sampled['_source_row']})

class ExactGame:
    def __init__(self,idx):
        self.idx=idx;self.n=len(idx)
        if self.n>16:raise ValueError('Too many zones for exact enumeration.')
        self.ids=np.arange(2**self.n);self.masks=((self.ids[:,None]>>np.arange(self.n))&1).astype(bool)
        self.contrasts=[]
        for j in range(self.n):
            without=self.ids[(self.ids&(1<<j))==0]
            size=np.array([int(v).bit_count() for v in without])
            weight=np.array([1/(self.n*math.comb(self.n-1,int(s))) for s in size])
            self.contrasts.append((without,without|(1<<j),weight))
    def values(self,model,x,bg,fast_linear=True):
        if fast_linear and isinstance(model,LogisticRegression):
            if list(model.classes_)!=[0,1]:raise ValueError('Unexpected class encoding.')
            base=float(model.intercept_[0]+bg@model.coef_[0])
            delta=np.array([(x[ii]-bg[ii])@model.coef_[0,ii] for ii in self.idx])
            return expit(base+self.masks@delta)
        variants=np.repeat(bg[None,:],len(self.ids),axis=0)
        for j,ii in enumerate(self.idx):variants[np.ix_(self.masks[:,j],ii)]=x[ii]
        pos=int(np.flatnonzero(model.classes_==1)[0])
        return np.asarray(model.predict_proba(variants)[:,pos],float)
    def explain(self,model,x,background):
        bg=np.atleast_2d(np.asarray(background,float));f=np.zeros(len(self.ids))
        for row in bg:f+=self.values(model,x,row)/len(bg)
        phi=np.array([np.dot(w,f[yes]-f[no]) for no,yes,w in self.contrasts])
        residual=abs(phi.sum()-(f[-1]-f[0]))
        if not np.isfinite(phi).all() or residual>1e-6:raise ValueError('Shapley efficiency check failed.')
        return phi,float(f[-1]),float(f[0]),float(residual)

def donor_panels(X,groups,rows,c,fold):
    panels=[];records=[]
    for p in range(c.donor_panels):
        rng=np.random.default_rng(np.random.SeedSequence([c.seed,fold,p,8191]));positions=[]
        for a in sorted(set(groups)):
            k=int(rng.choice(np.flatnonzero(groups==a)));positions.append(k)
            records.append({'Panel':p+1,'Training animal':a,'Source Excel row':int(rows[k])})
        panels.append(X[positions])
    return panels,pd.DataFrame(records)

def _fit(model,X,y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(X,y)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise ValueError('A model failed to converge.')
    return [{'Category':w.category.__name__,'Message':str(w.message)} for w in caught]

def analyze_frames(frames,c,progress=None):
    c.validate();contrib=[];games=[];donors=[];fits=[];selected=[];assignments=[];cohort=None;warning_records=[]
    prepared=[]
    for sheet in c.sheets:
        if sheet not in frames:raise ValueError(f'Missing sheet {sheet}.')
        d,s=prepare(frames[sheet],c);newcohort=dict(zip(d['groups'],d['y']))
        if cohort is not None and cohort!=newcohort:raise ValueError('Animal cohorts differ across sheets.')
        cohort=newcohort;s.insert(0,'Sheet',sheet);selected.append(s);prepared.append((sheet,d))
    total=sum(len(set(d['groups']))*len(c.models) for _,d in prepared);done=0
    with threadpool_limits(limits=2):
        for sheet,d in prepared:
            X,y,g,rows=d['X'],d['y'],d['groups'],d['rows'];names,idx,audit=assign_zones(d['features']);audit.insert(0,'Sheet',sheet);assignments.append(audit);game=ExactGame(idx)
            # Restart the legacy explanation RNG for each model; independent donor RNGs never consume it.
            selections=[];rng=np.random.RandomState(c.seed)
            splits=list(LeaveOneGroupOut().split(X,y,g))
            for fold,(tr,te) in enumerate(splits):
                core=rng.choice(len(te),size=c.explained_per_animal,replace=False)
                extra_rng=np.random.default_rng(np.random.SeedSequence([c.seed,fold,104729]));remaining=np.setdiff1d(np.arange(len(te)),core)
                extra=extra_rng.choice(remaining,size=c.extended_per_animal-len(core),replace=False)
                selections.append(np.r_[core,extra])
            for name in c.models:
                for fold,((tr,te),explain_positions) in enumerate(zip(splits,selections)):
                    animal=str(g[te][0]);sc=StandardScaler().fit(X[tr]);Xt,Xv=sc.transform(X[tr]),sc.transform(X[te])
                    if name=='Logistic Regression':model=LogisticRegression(max_iter=1000,random_state=c.seed)
                    else:
                        from xgboost import XGBClassifier
                        model=XGBClassifier(eval_metric='logloss',random_state=c.seed,n_jobs=1)
                    for w in _fit(model,Xt,y[tr]):warning_records.append({'Sheet':sheet,'Model':name,'Animal':animal,**w})
                    panels,dr=donor_panels(Xt,g[tr],rows[tr],c,fold);dr.insert(0,'Held-out animal',animal);dr.insert(0,'Model',name);dr.insert(0,'Sheet',sheet);donors.append(dr)
                    refs=[(MEAN,Xt.mean(axis=0)),(MEDIAN,np.median(Xt,axis=0))]+[(f'Empirical panel {i+1}',p) for i,p in enumerate(panels)]
                    fit={'Sheet':sheet,'Model':name,'Animal':animal,'Training animals':sorted(set(g[tr])),'Training rows':len(tr),'Test rows':len(te),'Scaler mean SHA256':hashlib.sha256(sc.mean_.tobytes()).hexdigest(),'Parameters':jsonable(model.get_params()),'OOF accuracy':float((model.predict(Xv)==y[te]).mean())}
                    if name=='XGBoost':fit['Resolved booster configuration']=json.loads(model.get_booster().save_config())
                    fits.append(fit)
                    for position,sel in enumerate(explain_positions):
                        x=Xv[sel];context={'Sheet':sheet,'Model':name,'Animal':animal,'Phenotype':'OIM' if y[te][sel] else 'WT','Source Excel row':int(rows[te][sel]),'Explanation order':position}
                        for baseline,bg in refs:
                            if position>=c.explained_per_animal and baseline!=MEAN:continue
                            phi,fx,ref,error=game.explain(model,x,bg)
                            labels=[baseline] if baseline!=MEAN else ([MEAN,EXTENDED] if position<c.explained_per_animal else [EXTENDED])
                            for label in labels:
                                games.append({**context,'Reference':label,'P(OIM)':fx,'Reference P(OIM)':ref,'Sum Shapley':float(phi.sum()),'Efficiency residual':error})
                                contrib.extend({**context,'Reference':label,'Zone':zone,'Shapley value':float(val),'Channels':len(ii)} for zone,ii,val in zip(names,idx,phi))
                    done+=1
                    if progress:progress(done/total,f'{sheet} — {name} — held-out {animal}')
    out={'contributions':pd.DataFrame(contrib),'games':pd.DataFrame(games),'donors':pd.concat(donors,ignore_index=True),'fitted_models':fits,'selected_rows':pd.concat(selected,ignore_index=True),'zone_assignment':pd.concat(assignments,ignore_index=True),'warnings':warning_records}
    out.update(summarize(out['contributions']))
    return out

def summarize(contributions):
    keys=['Sheet','Model','Reference','Animal','Phenotype','Zone'];d=contributions.copy();d['Absolute Shapley']=d['Shapley value'].abs()
    fold=d.groupby(keys,sort=False).agg(**{'Mean absolute Shapley':('Absolute Shapley','mean'),'Mean signed Shapley':('Shapley value','mean'),'Explained spectra':('Shapley value','size'),'Channels':('Channels','first')}).reset_index()
    grouping=['Sheet','Model','Reference','Animal'];fold['Rank']=fold.groupby(grouping)['Mean absolute Shapley'].rank(method='average',ascending=False)
    fold['Top 3']=(fold.groupby(grouping)['Mean absolute Shapley'].rank(method='min',ascending=False)<=3) & (fold['Mean absolute Shapley']>1e-12)
    fold['Importance share']=fold['Mean absolute Shapley']/fold.groupby(grouping)['Mean absolute Shapley'].transform('sum').replace(0,np.nan)
    fold['Mean absolute per channel']=fold['Mean absolute Shapley']/fold['Channels']
    summary=fold.groupby(['Sheet','Model','Reference','Zone'],sort=False).agg(**{'Mean absolute Shapley':('Mean absolute Shapley','mean'),'Animal SD':('Mean absolute Shapley','std'),'Minimum animal importance':('Mean absolute Shapley','min'),'Maximum animal importance':('Mean absolute Shapley','max'),'Median rank':('Rank','median'),'Minimum rank':('Rank','min'),'Maximum rank':('Rank','max'),'Top 3 folds':('Top 3','sum'),'Animal folds':('Animal','nunique'),'Channels':('Channels','first')}).reset_index()
    paired=[]
    for (sheet,model,animal),f in fold.groupby(['Sheet','Model','Animal']):
        base=f[f.Reference==MEAN].set_index('Zone')
        if base.empty:continue
        for ref in f.Reference.unique():
            if ref==MEAN:continue
            other=f[f.Reference==ref].set_index('Zone').reindex(base.index)
            a,b=base['Mean absolute Shapley'].to_numpy(),other['Mean absolute Shapley'].to_numpy()
            rho=float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else np.nan
            ta=set(base.index[base['Top 3']]);tb=set(other.index[other['Top 3']]);union=ta|tb
            variation=float(.5*np.abs(base['Importance share']-other['Importance share']).sum()) if a.sum()>0 and b.sum()>0 else np.nan
            paired.append({'Sheet':sheet,'Model':model,'Animal':animal,'Reference compared':ref,'Rank Spearman rho':rho,'Top 3 intersection':len(ta&tb),'Top 3 union':len(union),'Top 3 Jaccard':len(ta&tb)/len(union) if union else np.nan,'Importance-share total variation':variation,'Mean-reference total importance':float(a.sum()),'Compared total importance':float(b.sum())})
    return {'animal_importance':fold,'zone_summary':summary,'reference_pairs':pd.DataFrame(paired)}

def run(data,c=Config(),name='uploaded.xlsx',progress=None):
    start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    with pd.ExcelFile(io.BytesIO(data)) as b:frames={s:b.parse(s) for s in c.sheets}
    out=analyze_frames(frames,c,progress)
    versions={}
    for p in ['numpy','pandas','scipy','scikit-learn','xgboost','streamlit','threadpoolctl']:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]='not installed'
    out['manifest']={'version':VERSION,'engine_sha256':source_hash(),'original_app_blob':BASE_APP_BLOB,'input_name':name,'input_sha256':hashlib.sha256(data).hexdigest(),'configuration':asdict(c),'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),'analysis_elapsed_seconds':time.perf_counter()-start,'python':platform.python_version(),'versions':versions,'positive_class':'OIM','positive_code':1,'output_scale':'P(OIM), probability units','reference_definition':f'Mean and coordinate-wise median of outer-training spectra; {c.donor_panels} empirical panels, one complete training spectrum per training animal. Coalition probabilities averaged over donors, not predictions at an averaged spectrum.','sampling':'Legacy zone-SHAP BLAKE2b animal-specific seed, then seed-based row shuffle. Primary explanation RNG is independent of donor RNG.','historical_upstream_preprocessing_reconstructed':False,'new_analysis':True,'top3_ties':'Include all zones tied at the third-largest value, excluding numerical zero importance (<=1e-12).','fold_interpretation':'Each fold corresponds to one explained animal; fitted-model and animal variation are not separately identified.','conditional_distribution_or_causal_effect':False,'zone_endpoints':'First matching zone wins; all remaining channels assigned to gaps.','runtime_controls':{'BLAS_threads':2,'classifier_n_jobs':1}}
    return out

def methods(out):
    c=out['manifest']['configuration']
    return f'''# Additional analysis for Reviewer 1 Comment 3

Models are trained using animal-grouped LOGO validation and StandardScaler fitted only on training animals. The prepared input is used without additional contamination/outlier filtering or derivative generation. Positive output is P(OIM), with OIM=1 and WT=0. Each spectral channel is assigned exactly once to the nine chemical zones or an unassigned-gaps group. Shared endpoints go to the first listed zone.

For each fixed fitted model and held-out spectrum, all 2^K zone coalitions are enumerated. Missing zones are filled from a training-only reference. Training mean is compared with the coordinate-wise training median and {c['donor_panels']} empirical panels; each panel uses one complete spectrum from each training animal. For empirical panels, coalition probabilities are averaged over donor spectra before calculating Shapley values. This is a limited background sensitivity analysis, not conditional SHAP. The same {c['explained_per_animal']} held-out spectra per animal are used for each reference. The mean reference is also evaluated on {c['extended_per_animal']} spectra per animal, containing the original explained subset, to assess explanation-sampling sensitivity. No classifier is retrained between references.

Importance is the mean absolute zone contribution within each held-out animal, followed by equal-weight summaries across animals. Sample SD, rank ranges and top-three frequency are reported. Ties at the third value are included; numerical zero importance (<=1e-12) is excluded. Reference comparisons use within-animal Spearman rank correlation, top-three Jaccard similarity and total variation between normalized importance shares. Fold and animal variability are paired by LOGO, and overlapping training sets prevent treating folds as independent model experiments. Grouped masking preserves each donor's values within a replaced zone but can break dependencies across zones and create hybrid spectra outside the observed distribution. Mean and median references are not guaranteed to represent a physically realizable specimen. Attributions describe model behavior relative to the chosen background, not causal biochemical effects.

Numerical efficiency is checked for every explanation: sum(phi)=P(OIM|full spectrum)-reference output. The reference output is predicted by the fitted fold model; it is never assumed to be 0.5. Exact coalition enumeration does not remove background-choice uncertainty. Existing archived values and this new reanalysis are kept separate.
'''

def bundle(out):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for k,v in out.items():
            if isinstance(v,pd.DataFrame):z.writestr(k+'.csv',v.to_csv(index=False))
        for k in ['manifest','fitted_models','warnings']:z.writestr(k+'.json',json.dumps(jsonable(out[k]),indent=2,allow_nan=False))
        z.writestr('Methods.md',methods(out))
    return b.getvalue()

def render_panel(st):
    st.header('Reviewer 1 Comment 3: zone-SHAP stability and reference sensitivity')
    st.caption('Separate analysis. All references are derived from training animals only. The working application is unchanged.')
    uploaded=st.file_uploader('Input spectral workbook',type=['xlsx'],key='r1c3_upload')
    sheets=tuple(st.multiselect('Spectral representations',['Raw','2nd derivative'],default=['Raw'],key='r1c3_sheets'))
    models=tuple(st.multiselect('Manuscript models',['Logistic Regression','XGBoost'],default=['Logistic Regression','XGBoost'],key='r1c3_models'))
    n=int(st.number_input('Training spectra per animal',1,2000,250,key='r1c3_n'))
    cap=int(st.number_input('Primary explained spectra per animal',1,n,min(10,n),key='r1c3_cap'))
    ext=int(st.number_input('Extended mean-reference explained spectra per animal',cap,n,max(cap,min(50,n)),key='r1c3_extended'))
    seed=int(st.number_input('Random seed',0,10000,42,key='r1c3_seed'))
    c=Config(n,seed,cap,ext,3,sheets,models)
    st.info('Ten zone groups imply 1,024 coalitions per spectrum and reference. This can take several minutes. Empirical backgrounds are a limited sensitivity check; they do not establish physical realism or causality.')
    if uploaded is None:return
    data=uploaded.getvalue();key=identity(data,c)
    if st.button('Run Comment 3 analysis',type='primary',key='r1c3_run'):
        st.session_state.pop('r1c3_completed',None);bar=st.progress(0.0);status=st.empty()
        try:
            def update(fraction,message):bar.progress(fraction);status.write(message)
            out=run(data,c,uploaded.name,update);st.session_state['r1c3_completed']={'identity':key,'result':out,'zip':bundle(out)};status.success('Analysis completed.')
        except Exception as e:st.error(f'Analysis not accepted: {e}')
    completed=st.session_state.get('r1c3_completed')
    if not completed or completed['identity']!=key:
        st.info('Run with the current workbook and settings to enable results and downloads.');return
    out=completed['result'];st.dataframe(out['zone_summary'],hide_index=True,width='stretch');st.dataframe(out['reference_pairs'],hide_index=True,width='stretch')
    st.download_button('Download Comment 3 tables and audit package',completed['zip'],'Reviewer1_Comment3_Reanalysis.zip','application/zip',key='r1c3_zip')
    st.download_button('Download run-specific Methods',methods(out),'Reviewer1_Comment3_Methods.md','text/markdown',key='r1c3_methods')
