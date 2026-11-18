"""Recompute stability from an existing Method B workbook; no refitting.

python research/analyze_archived_zone_shap.py input.xlsx --representation Raw --output output_dir
"""
from pathlib import Path
import argparse,sys
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr,friedmanchisquare
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reviewer1_comment3 import summarize,MEAN

def analyze(path,representation):
    book=pd.ExcelFile(path);readme=pd.read_excel(book,sheet_name='README')
    if not readme.astype(str).apply(lambda c:c.str.contains('Zone perturbation',case=False,regex=False)).any().any():raise ValueError('This reader requires Method B zone-perturbation output, not Method A.')
    zones=pd.read_excel(book,sheet_name='Zone definitions').set_index('Zone');records=[]
    names={'RandomForest':'Random Forest','LogisticRegressi':'Logistic Regression','XGBoost':'XGBoost'}
    for tab in book.sheet_names:
        if not tab.endswith(' per-spectrum'):continue
        prefix=tab.removesuffix(' per-spectrum');model=names.get(prefix,prefix)
        frame=pd.read_excel(book,sheet_name=tab);features=[c for c in frame if c not in ['Animal','True class']]
        if not set(features)<=set(zones.index) or not np.isfinite(frame[features].to_numpy(float)).all():raise ValueError('Invalid zone data.')
        for i,row in frame.iterrows():
            records.extend({'Sheet':representation,'Model':model,'Reference':MEAN,'Animal':str(row.Animal),'Phenotype':str(row['True class']),'Explanation record':i+1,'Zone':z,'Shapley value':float(row[z]),'Channels':int(zones.loc[z,'Channels'])} for z in features)
    values=pd.DataFrame(records);out=summarize(values);out['contributions']=values;agreement=[]
    for (sheet,model),g in out['animal_importance'].groupby(['Sheet','Model']):
        mat=g.pivot(index='Animal',columns='Zone',values='Mean absolute Shapley');m,n=mat.shape
        ranks=np.vstack([rankdata(-a) for a in mat.to_numpy()]);S=np.square(ranks.sum(axis=0)-m*(n+1)/2).sum();T=sum(sum(c**3-c for c in np.unique(v,return_counts=True)[1]) for v in mat.to_numpy());den=m*m*(n**3-n)-m*T;W=12*S/den if den else np.nan
        if den:
            independent=float(friedmanchisquare(*mat.to_numpy().T).statistic)/(m*(n-1))
            if not np.isclose(W,independent,rtol=0,atol=1e-12):raise AssertionError('Rank concordance check failed.')
        rho=[spearmanr(mat.iloc[i],mat.iloc[j]).statistic for i in range(m) for j in range(i)]
        agreement.append({'Sheet':sheet,'Model':model,'Kendall W':W,'Median pairwise Spearman':float(np.median(rho)),'Minimum pairwise Spearman':float(np.min(rho)),'Maximum pairwise Spearman':float(np.max(rho))})
    out['rank_agreement']=pd.DataFrame(agreement);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workbook');p.add_argument('--representation',required=True);p.add_argument('--output',required=True);a=p.parse_args();dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    for name,table in analyze(a.workbook,a.representation).items():table.to_csv(dest/(name+'.csv'),index=False)
