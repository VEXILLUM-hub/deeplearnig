from pathlib import Path
import csv,json,hashlib,warnings,time,sys
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parent
ROIS={'Cam0':(.24,.32,.38,.94),'Cam1':(.33,.52,.74,.70),'Cam2':(.12,.28,.77,.83)}
SPLITS={'operating_point_028':'train_normal','operating_point_031':'calibration_normal','operating_point_033':'test_normal'}
def read(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def vector(path,camera):
 im=Image.open(path).convert('L');w,h=im.size;x0,y0,x1,y1=ROIS[camera]
 return np.asarray(im.crop((int(x0*w),int(y0*h),int(x1*w),int(y1*h))).resize((24,24),Image.Resampling.BILINEAR),dtype=np.float32).reshape(-1)/255

def predict(x,weights):
 x=np.asarray(x,dtype=np.float32)
 for i in range(4):
  x=x@weights['w'+str(i)]+weights['b'+str(i)]
  if i<3:x=np.maximum(x,0)
 return x

def alarms(times,scores,threshold,recovery):
 """Causal two-frame persistence; reset on gaps and recovery. No centered smoothing."""
 streak=0;active=[];onsets=[]
 for i,(t,s) in enumerate(zip(times,scores)):
  if i and (t-times[i-1]>5 or t<=times[i-1]):streak=0
  previous=streak>=2
  if recovery[i]:streak=0
  else:streak=streak+1 if s>threshold else 0
  current=streak>=2;active.append(current);onsets.append(current and not previous)
 return active,onsets

def train(source,output):
 from sklearn.neural_network import MLPRegressor
 from sklearn.exceptions import ConvergenceWarning
 from threadpoolctl import threadpool_limits
 import sklearn
 source=Path(source);output=Path(output);models=output/'models';models.mkdir(parents=True,exist_ok=True)
 audit=read(source/'window_audit.csv');selection={j['qc']['video_id']:j for j in json.loads((source/'selection.json').read_text(encoding='utf-8'))};manifest=[];data={};image_hashes=[]
 for a in audit:
  vid=a['video_id'];rows=read(source/vid/'frames.csv');op=a['run_id'].split('|')[1];role=SPLITS.get(op,'test_event')
  manifest.append(dict(video_id=vid,run_id=a['run_id'],camera=a['camera'],split=role,frames=len(rows),comparison_valid=not (op=='operating_point_033' and a['camera']=='Cam0')))
  x=[]
  for r in rows:
   path=source/vid/r['image'];x.append(vector(path,a['camera']));image_hashes.append(dict(file=vid+'/'+r['image'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  data[vid]=(rows,np.array(x,dtype=np.float32),a,role)
 # Every Run and all its cameras occupy exactly one split.
 for rid in set(r['run_id'] for r in manifest):assert len(set(r['split'] for r in manifest if r['run_id']==rid))==1
 scores=[];model_records=[];run_summary=[]
 with threadpool_limits(limits=1):
  for camera in ['Cam0','Cam1','Cam2']:
   fitid=next(r['video_id'] for r in manifest if r['camera']==camera and r['split']=='train_normal');calid=next(r['video_id'] for r in manifest if r['camera']==camera and r['split']=='calibration_normal')
   fit=data[fitid][1];cal=data[calid][1];start=time.time()
   # Same prespecified capacity/budget for all cameras. No event-based model selection.
   model=MLPRegressor(hidden_layer_sizes=(64,16,64),activation='relu',solver='adam',alpha=.001,batch_size=32,learning_rate_init=.001,max_iter=80,early_stopping=False,n_iter_no_change=81,random_state=7,shuffle=True)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning);model.fit(fit,fit)
   weights={**{'w'+str(i):v for i,v in enumerate(model.coefs_)},**{'b'+str(i):v for i,v in enumerate(model.intercepts_)}}
   np.savez_compressed(models/(camera+'_autoencoder.npz'),**weights)
   assert np.allclose(model.predict(cal),predict(cal,weights),atol=1e-5)
   cal_scores=np.mean((predict(cal,weights)-cal)**2,axis=1);threshold=float(np.quantile(cal_scores,.99))
   meta=dict(camera=camera,model='dense_autoencoder',architecture=[576,64,16,64,576],input='24x24 grayscale fixed ROI /255',roi=ROIS[camera],train_video=fitid,calibration_video=calid,train_run=data[fitid][2]['run_id'],calibration_run=data[calid][2]['run_id'],train_frames=len(fit),calibration_frames=len(cal),threshold=threshold,threshold_rule='calibration-normal 99th percentile; causal 2-frame persistence',seed=7,epochs=model.n_iter_,training_loss=float(model.loss_),loss_curve=[float(x) for x in model.loss_curve_],training_seconds=time.time()-start,optimizer='adam',training_warning=[str(w.message) for w in caught])
   (models/(camera+'_metadata.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8');model_records.append(meta)
   print(camera,'trained',model.n_iter_,'epochs; threshold',threshold,flush=True)
   for vid,(rows,x,a,role) in data.items():
    if a['camera']!=camera:continue
    mse=np.mean((predict(x,weights)-x)**2,axis=1);times=[float(r['sensor_elapsed_seconds']) for r in rows];recover=[r['label3_recovery']=='True' for r in rows];active,onsets=alarms(times,mse,threshold,recover)
    job=selection[vid];cause=job['cause'];observed=job['observed'];before=0 if cause is None else sum(t<cause and v for t,v in zip(times,active));new_between=[]
    for i,r in enumerate(rows):
     row=dict(video_id=vid,run_id=a['run_id'],camera=camera,split=role,frame_index=int(r['frame_index']),image=r['image'],sensor_elapsed_seconds=times[i],relative_to_sensor_seconds=times[i]-observed,segment=r['segment'],label3_recovery=recover[i],active_source_annotation_ids=r['active_source_annotation_ids'],reconstruction_mse=float(mse[i]),threshold=threshold,above_threshold=bool(mse[i]>threshold),persistent_flag=active[i],flag_onset=onsets[i],gap_before=i>0 and times[i]-times[i-1]>5)
     scores.append(row)
     if cause is not None and cause<=times[i]<observed and onsets[i]:new_between.append(times[i]-observed)
    eligible=sum(not r for r in recover);summary=dict(video_id=vid,run_id=a['run_id'],camera=camera,split=role,comparison_valid=not('operating_point_033' in a['run_id'] and camera=='Cam0'),frames=len(rows),nonrecovery_frames=eligible,score_median=float(np.median(mse)),score_p95=float(np.quantile(mse,.95)),flagged_nonrecovery_frames=sum(active),flagged_frame_fraction=sum(active)/eligible if eligible else None,flag_episodes=sum(onsets),flagged_frames_before_cause=before,new_flag_onsets_cause_to_sensor=len(new_between),earliest_new_flag_relative_sensor='' if not new_between else min(new_between),interpretation='exploratory pixel reconstruction error; not bubble detection or future symptom prediction')
    run_summary.append(summary)
 write(output/'scores.csv',scores);write(output/'run_summary.csv',run_summary);write(output/'split_manifest.csv',manifest);write(output/'input_frame_sha256.csv',image_hashes)
 status=dict(model_trained=True,models=3,architecture='576-64-16-64-576 dense autoencoder; 3 hidden layers, not a CNN',total_scored_frames=len(scores),train_runs=1,calibration_runs=1,held_out_normal_runs=1,held_out_event_runs=3,sklearn_version=sklearn.__version__,numpy_version=np.__version__,protocol='fixed roles OP028 train / OP031 calibration / OP033 normal test / OP022 OP023 OP026 event test; all cameras kept in same Run split',causal_flag='2 consecutive above-threshold frames; reset after >5s gap, no recovery flags',gate='not_validated',bubble_detection='not_validated',future_symptom_prediction='not_trained',limitations=['Only one normal Run used for training and one for calibration; same-stage process matching not verified.','ROIs inherited from preceding exploratory review; this is not a blinded confirmatory study.','OP033 Cam0 view mismatch: scores shown as out-of-view comparison and excluded from normal comparison claims.','Event annotations are air-ingress interventions, not bubble onset labels.','A new flag before a sensor annotation is not validated early detection.','Frame windows and cameras are correlated; do not count frames as independent trials.','80 epochs is a fixed exploration budget; warning may report optimizer did not converge.','No raw sensor values or measured camera/sensor synchronization uncertainty supplied.'])
 (output/'experiment_summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
 (output/'model_training.json').write_text(json.dumps(model_records,indent=2),encoding='utf-8');return status
if __name__=='__main__':print(json.dumps(train(sys.argv[1],sys.argv[2]),ensure_ascii=False,indent=2))
