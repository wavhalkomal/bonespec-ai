from pathlib import Path
import hashlib,io,itertools,json,sys,unittest
from unittest.mock import patch
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reviewer1_comment3 as e

def fixture(n=6):
 rng=np.random.RandomState(19);rows=[]
 for animal in range(6):
  for _ in range(n):rows.append({'Sample ID':f'Mouse_{animal}','Type':'OIM' if animal>=3 else 'WT',**dict(zip([1800,1700,1650,1600,1550,1500,1430,1400,1338,1250,1150,1050,875,800],rng.normal(size=14)+(animal>=3)*.3))})
 return pd.DataFrame(rows)
def small():
 c=e.Config(spectra_per_animal=4,explained_per_animal=2,extended_per_animal=3,donor_panels=1,models=('Logistic Regression',));return c,fixture()

class Games(unittest.TestCase):
 def test_zones_cover_every_channel_once_and_endpoint_priority(self):
  names,idx,a=e.assign_zones(list(range(1800,798,-2)))
  self.assertEqual(len(names),10);self.assertEqual(len(np.concatenate(idx)),501);self.assertEqual(len(idx[names.index(e.GAPS)]),141)
  lookup=a.set_index('Wavenumber').Zone
  for w,n in [(1600,'Amide I'),(1430,'CH (1490-1430)'),(1200,'Amide III'),(1100,'HPO4')]:self.assertEqual(lookup[w],n)
 def test_exact_shapley_matches_independent_permutation_definition(self):
  rng=np.random.RandomState(2);X=rng.normal(size=(30,3));y=(X.sum(axis=1)>0).astype(int);m=LogisticRegression().fit(X,y);x=np.array([.5,-.2,.7]);bg=np.array([-.3,.1,-.4]);game=e.ExactGame([np.array([i]) for i in range(3)])
  phi,fx,ref,error=game.explain(m,x,bg);manual=np.zeros(3)
  for perm in itertools.permutations(range(3)):
   v=bg.copy();previous=m.predict_proba(v[None,:])[0,1]
   for j in perm:
    v[j]=x[j];current=m.predict_proba(v[None,:])[0,1];manual[j]+=(current-previous)/6;previous=current
  np.testing.assert_allclose(phi,manual,atol=1e-14);self.assertLess(error,1e-14)
 def test_linear_fast_path_matches_direct_probability_evaluation(self):
  rng=np.random.RandomState(4);X=rng.normal(size=(50,5));y=(X[:,0]+X[:,2]>0).astype(int);m=LogisticRegression().fit(X,y);game=e.ExactGame([np.array([0,1]),np.array([2,3]),np.array([4])]);x=X[0];bg=X.mean(axis=0)
  np.testing.assert_allclose(game.values(m,x,bg),game.values(m,x,bg,fast_linear=False),atol=1e-14)
 def test_empirical_game_averages_probabilities_not_reference_spectrum(self):
  rng=np.random.RandomState(1);X=rng.normal(size=(40,3));m=LogisticRegression().fit(X,(X.sum(axis=1)>0).astype(int));game=e.ExactGame([np.array([0]),np.array([1]),np.array([2])]);refs=np.array([[-3.,-1.,-2.],[1.,2.,2.]])
  phi,fx,ref,_=game.explain(m,X[0],refs);single=[game.explain(m,X[0],b) for b in refs]
  np.testing.assert_allclose(phi,np.mean([t[0] for t in single],axis=0),atol=1e-14)
  self.assertAlmostEqual(ref,float(m.predict_proba(refs)[:,1].mean()));self.assertGreater(abs(ref-float(m.predict_proba(refs.mean(axis=0)[None,:])[0,1])),1e-3)
 def test_invalid_game_outputs_rejected(self):
  game=e.ExactGame([np.array([0])]);model=object()
  with patch.object(game,'values',return_value=np.array([np.nan,np.nan])):
   with self.assertRaises(ValueError):game.explain(model,np.array([1.]),np.array([0.]))

class Input(unittest.TestCase):
 def test_sampling_reproducible_and_source_rows_traceable(self):
  c,d=small();a,rows=e.prepare(d,c);b,other=e.prepare(d,c);np.testing.assert_array_equal(a['rows'],b['rows']);self.assertTrue(rows.groupby('Animal').size().eq(4).all());self.assertEqual(set(a['groups']),set(d['Sample ID']))
  np.testing.assert_array_equal(d.iloc[a['rows']-2]['Sample ID'],a['groups'])
 def test_invalid_features_labels_and_shortfalls_rejected(self):
  c,d=small()
  for mode in ['missing','label','shortfall']:
   a=d.copy()
   if mode=='missing':a.loc[0,1800]=np.nan
   if mode=='label':a.loc[0,'Type']='OIM'
   if mode=='shortfall':a=a.iloc[4:]
   with self.subTest(mode=mode),self.assertRaises(ValueError):e.prepare(a,c)
 def test_donors_are_only_training_animals_and_reproducible(self):
  c,d=small();v,_=e.prepare(d,c);keep=v['groups']!='Mouse_0';p,a=e.donor_panels(v['X'][keep],v['groups'][keep],v['rows'][keep],c,0);q,b=e.donor_panels(v['X'][keep],v['groups'][keep],v['rows'][keep],c,0)
  self.assertNotIn('Mouse_0',set(a['Training animal']));self.assertEqual(len(a),5);pd.testing.assert_frame_equal(a,b);np.testing.assert_array_equal(p[0],q[0])

class Workflow(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c,cls.d=small();cls.r=e.analyze_frames({'Raw':cls.d},cls.c)
 def test_no_refit_between_references_and_outer_group_isolation(self):
  self.assertEqual(len(self.r['fitted_models']),6)
  for fit in self.r['fitted_models']:self.assertNotIn(fit['Animal'],fit['Training animals']);self.assertEqual(fit['Training rows'],20)
  for _,row in self.r['donors'].iterrows():self.assertNotEqual(row['Held-out animal'],row['Training animal'])
 def test_same_spectra_used_for_reference_comparisons(self):
  g=self.r['games']
  for animal,f in g.groupby('Animal'):
   primary=set(f[f.Reference==e.MEAN]['Source Excel row']);extended=set(f[f.Reference==e.EXTENDED]['Source Excel row']);self.assertEqual(len(primary),2);self.assertEqual(len(extended),3);self.assertTrue(primary<=extended)
   for ref in [e.MEDIAN,'Empirical panel 1']:self.assertEqual(set(f[f.Reference==ref]['Source Excel row']),primary)
 def test_prediction_unchanged_across_reference_choices_and_reference_not_assumed_half(self):
  g=self.r['games'];self.assertLess(g.groupby(['Animal','Source Excel row'])['P(OIM)'].agg(lambda x:x.max()-x.min()).max(),1e-12)
  self.assertTrue((abs(g['Reference P(OIM)']-.5)>1e-4).any());self.assertLess(g['Efficiency residual'].max(),1e-12)
 def test_test_features_do_not_change_its_training_or_background(self):
  d=self.d.copy();mask=d['Sample ID']=='Mouse_0';d.loc[mask,d.columns[2:]]+=10
  r=e.analyze_frames({'Raw':d},self.c)
  left=self.r['games'].query('Animal=="Mouse_0"')['Reference P(OIM)'].to_numpy();right=r['games'].query('Animal=="Mouse_0"')['Reference P(OIM)'].to_numpy();np.testing.assert_allclose(left,right,atol=1e-14)
  a=next(f for f in self.r['fitted_models'] if f['Animal']=='Mouse_0');b=next(f for f in r['fitted_models'] if f['Animal']=='Mouse_0');self.assertEqual(a['Scaler mean SHA256'],b['Scaler mean SHA256'])
 def test_mean_absolute_contributions_do_not_cancel_and_animals_equal_weight(self):
  rows=[]
  for animal,values in [('a',[1.,-1.]),('b',[3.])]:
   for x in values:rows.append({'Sheet':'Raw','Model':'LR','Reference':e.MEAN,'Animal':animal,'Phenotype':'WT','Zone':'Zone','Shapley value':x,'Channels':1})
  r=e.summarize(pd.DataFrame(rows));self.assertEqual(r['zone_summary']['Mean absolute Shapley'].iloc[0],2.)
 def test_ties_and_zero_importance_are_reported_without_false_stability(self):
  rows=[{'Sheet':'Raw','Model':'LR','Reference':ref,'Animal':'a','Phenotype':'WT','Zone':str(z),'Shapley value':0.,'Channels':1} for ref in [e.MEAN,e.MEDIAN] for z in range(5)]
  r=e.summarize(pd.DataFrame(rows));self.assertFalse(r['animal_importance']['Top 3'].any());self.assertTrue(r['reference_pairs']['Rank Spearman rho'].isna().all());self.assertTrue(r['reference_pairs']['Importance-share total variation'].isna().all())
 def test_working_script_is_unchanged(self):
  p=Path(__file__).resolve().parents[1]/'app.py';b=p.read_bytes();self.assertEqual(hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),e.BASE_APP_BLOB)

if __name__=='__main__':unittest.main()
