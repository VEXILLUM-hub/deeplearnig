from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import streamlit as st
import altair as alt
from train_experiment import vector,predict,ROIS
BASE=Path(__file__).resolve().parent
st.set_page_config(page_title='배치 증류 영상 탐색 실험',layout='wide')
@st.cache_data
def load():
 scores=pd.read_csv(BASE/'experiment/scores.csv');summary=pd.read_csv(BASE/'experiment/run_summary.csv');splits=pd.read_csv(BASE/'experiment/split_manifest.csv')
 jobs={j['qc']['video_id']:j for j in json.loads((BASE/'data/selection.json').read_text(encoding='utf-8'))}
 return scores,summary,splits,jobs
scores,summary,splits,jobs=load()
st.title('배치 증류 · 영상 변화 탐색')
st.info('학습된 모델의 탐색 결과입니다. 점수는 정상 학습 화면을 재구성한 오차이며 기포 판정·발생 확률·미래 센서 증상 예측이 아닙니다.')
st.warning('이번 모델은 평가용 정상 Run에서도 플래그가 많았습니다. 공기 유입 조작→센서 반응 사이에 새로 시작된 플래그는 0건입니다. 조기 탐지 성공으로 해석할 수 없습니다.')
labels={r: r.split('|')[1].replace('operating_point_','OP')+' / '+r.split('|')[2] for r in scores.run_id.unique()}
run=st.sidebar.selectbox('실험 선택',list(labels),format_func=lambda r:labels[r])
camera=st.sidebar.selectbox('카메라',['Cam0','Cam1','Cam2'])
rows=scores[(scores.run_id==run)&(scores.camera==camera)].sort_values('sensor_elapsed_seconds').reset_index(drop=True)
vid=rows.video_id.iloc[0];job=jobs[vid];meta=json.loads((BASE/f'experiment/models/{camera}_metadata.json').read_text());one=summary[summary.video_id==vid].iloc[0];event=job['role']=='event'
st.caption('선택 Run 역할: '+str(rows.split.iloc[0])+' · 학습 OP028 / 기준 설정 OP031 / 정상 평가 OP033 / 사건 평가 OP022·023·026')
if not bool(one.comparison_valid):st.warning('OP033 Cam0은 촬영 각도가 다릅니다. 이 점수는 구도 차이가 섞인 참고값이며 정상 대조 성능으로 평가하지 않습니다.')
cols=st.columns(3)
cols[0].metric('지속 플래그 프레임 비율',f'{100*one.flagged_frame_fraction:.1f}%')
cols[1].metric('조작→센서 기록 간격',f'{job["observed"]-job["cause"]:.0f}초' if event else '사건 없음')
cols[2].metric('미래 증상 예측','미학습')
st.caption('플래그 비율은 Label3 회복 제외 프레임 기준입니다. 프레임 비율은 독립 사건 탐지율이 아닙니다. 센서 표시선은 원본 주석 시점이며 실제 측정값 그래프가 아닙니다.')
# Break connecting lines at recorded gaps; no clock interpolation.
chartdata=rows.copy();chartdata['gap_group']=chartdata.gap_before.astype(int).cumsum().astype(str)
x='relative_to_sensor_seconds:Q'
line=alt.Chart(chartdata).mark_line().encode(x=alt.X(x,title='센서 주석 기준 경과시간 (초)' if event else '비교 기준 경과시간 (초)'),y=alt.Y('reconstruction_mse:Q',title='재구성 오차 MSE'),detail='gap_group:N',tooltip=['frame_index','relative_to_sensor_seconds','reconstruction_mse','persistent_flag','label3_recovery'])
flag=alt.Chart(chartdata[chartdata.persistent_flag]).mark_point(color='#dc2626',size=30).encode(x=x,y='reconstruction_mse:Q')
limit=alt.Chart(pd.DataFrame({'threshold':[meta['threshold']]})).mark_rule(color='#d97706',strokeDash=[6,4]).encode(y='threshold:Q')
chart=line+flag+limit
if event:
 marks=pd.DataFrame({'t':[job['cause']-job['observed'],0.],'event':['공기 유입 조작 기록','PDI702 증상 주석']})
 chart+=alt.Chart(marks).mark_rule(strokeDash=[3,3]).encode(x='t:Q',color=alt.Color('event:N',title='원본 주석'),tooltip=['event','t'])
st.altair_chart(chart.properties(height=300),width='stretch')
st.caption('파란선: 모델 점수 · 주황선: OP031 정상 기준 99백분위 · 빨간점: 2개 연속 프레임 초과. 5초 초과 촬영 공백에서 연결선과 지속 상태를 끊습니다.')
idx=st.slider('실제 프레임 선택',0,len(rows)-1,min(len(rows)-1,int(np.abs(rows.relative_to_sensor_seconds.to_numpy()).argmin())),key=f'frame_{vid}')
r=rows.iloc[idx];path=BASE/'data'/vid/r.image
im=Image.open(path).convert('RGB');w,h=im.size;box=ROIS[camera];draw=ImageDraw.Draw(im);draw.rectangle((int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h)),outline='#ff3030',width=3)
weights=np.load(BASE/f'experiment/models/{camera}_autoencoder.npz',allow_pickle=False);v=vector(path,camera);reconstruction=predict(v[None],weights)[0];live=float(np.mean((reconstruction-v)**2))
left,right=st.columns([2,1]);left.image(im,caption=f'frame {int(r.frame_index)} · 센서 경과 {r.sensor_elapsed_seconds:.1f}초 · 기준 대비 {r.relative_to_sensor_seconds:+.1f}초',width='stretch')
with right:
 st.write(f'실제 모델 추론 오차: {live:.6f}')
 st.write('회복 Label3: '+('해당' if r.label3_recovery else '해당 없음'))
 st.write('탐색 플래그: '+('기준 초과 지속' if r.persistent_flag else '지속 플래그 없음'))
 st.caption('빨간 사각형은 탐색 관심영역입니다. 아래는 24×24 모델 입력과 재구성입니다. 재구성 출력은 보기용으로만 0~1 범위를 잘라 표시합니다.')
 crop=Image.fromarray(np.uint8(v.reshape(24,24)*255)).resize((192,192),Image.Resampling.NEAREST)
 recon=Image.fromarray(np.uint8(np.clip(reconstruction.reshape(24,24),0,1)*255)).resize((192,192),Image.Resampling.NEAREST)
 st.image([crop,recon],caption=['실제 입력','모델 재구성'],width=192)
 st.caption('기포 위치 표시나 기포 정답은 제공하지 않습니다.')
with st.expander('정상 평가 · 모델 한계',expanded=False):
 st.dataframe(summary[summary.split=='test_normal'][['camera','comparison_valid','flagged_frame_fraction','flag_episodes']],hide_index=True,width='stretch')
 st.write('학습·기준 설정 각각 정상 Run 1개뿐입니다. 같은 장치라도 공정 상태·반사·촬영 구도 차이에 민감할 수 있습니다. 일반화 성능은 충분히 검증하지 못했습니다.')
 st.write('576→64→16→64→576 전결합 오토인코더, 은닉층 3개, 카메라별 80 epoch 고정 학습. CNN·기포 검출기·시계열 미래 예측 모델은 아닙니다.')
 st.write('사건 주변 화면을 본 뒤 정한 ROI를 재사용한 탐색 실험입니다. 정상/이상 전체 데이터셋에 대한 확증 평가가 아닙니다.')
 st.dataframe(splits,hide_index=True,width='stretch')
 st.line_chart(pd.DataFrame({'training_loss':meta['loss_curve']}))
st.download_button('현재 실험 점수 CSV',rows.to_csv(index=False).encode('utf-8-sig'),file_name=f'{vid}_scores.csv',mime='text/csv')
