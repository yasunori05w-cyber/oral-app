import streamlit as st
import numpy as np
import librosa
import parselmouth
from sklearn.ensemble import RandomForestClassifier
import joblib
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="学習ツール", layout="centered")
st.title("🧠 [oral] スマホ用学習ツール")

with st.sidebar:
    st.header("⚙️ 検知チューニング")
    onset_delta = st.slider("検知感度", 0.01, 0.30, 0.08)

if 'features' not in st.session_state:
    st.session_state.features = []
if 'labels' not in st.session_state:
    st.session_state.labels = []
if 'counts' not in st.session_state:
    st.session_state.counts = {"パ": 0, "タ": 0, "カ": 0}
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None

target_label = st.radio("録音する音節", ["パ", "タ", "カ"])
label_map = {"パ": 0, "タ": 1, "カ": 2}

audio_bytes = audio_recorder(text="タップして録音", recording_color="#ff4b4b", neutral_color="#4b4bff")

if audio_bytes and audio_bytes != st.session_state.last_audio:
    st.session_state.last_audio = audio_bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        sr_target = 16000
        y, sr = librosa.load(tmp_path, sr=sr_target)
        y = librosa.util.normalize(y)
        y_pre = librosa.effects.preemphasis(y)
        
        wait_frames = int(sr * 0.1 / 512)
        onset_frames = librosa.onset.onset_detect(
            y=y_pre, sr=sr, wait=wait_frames, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=onset_delta
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        if len(onset_times) > 0:
            extracted_features = []
            snd = parselmouth.Sound(tmp_path)
            formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)
            
            for t in onset_times:
                start_idx = max(0, int((t - 0.02) * sr))
                end_idx = min(len(y), int((t + 0.02) * sr))
                segment_consonant = y_pre[start_idx:end_idx]
                zcr = np.mean(librosa.feature.zero_crossing_rate(segment_consonant)) if len(segment_consonant) > 0 else 0
                
                target_time = t + 0.02
                f2 = formants.get_value_at_time(2, target_time)
                f3 = formants.get_value_at_time(3, target_time)
                
                if np.isnan(f2) or np.isnan(f3):
                    f2 = np.nanmean([formants.get_value_at_time(2, target_time + i*0.01) for i in range(5)])
                    f3 = np.nanmean([formants.get_value_at_time(3, target_time + i*0.01) for i in range(5)])
                
                f2 = 0 if np.isnan(f2) else f2
                f3 = 0 if np.isnan(f3) else f3
                
                extracted_features.append([zcr, f2, f3])
            
            if len(extracted_features) > 0:
                st.session_state.features.extend(extracted_features)
                st.session_state.labels.extend([label_map[target_label]] * len(extracted_features))
                st.session_state.counts[target_label] += len(extracted_features)
                st.success(f"「{target_label}」を {len(extracted_features)} 件抽出しました！")
        else:
            st.warning("発音が検出されませんでした。")
    except Exception as e:
        st.error(f"エラー: {e}")
    finally:
        os.remove(tmp_path)

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("パ", f"{st.session_state.counts['パ']} 件")
c2.metric("タ", f"{st.session_state.counts['タ']} 件")
c3.metric("カ", f"{st.session_state.counts['カ']} 件")

if all(c >= 10 for c in st.session_state.counts.values()):
    if st.button("🚀 スマホ環境に適応したAIを学習"):
        X = np.array(st.session_state.features)
        y = np.array(st.session_state.labels)
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, "oral_model_hybrid.pkl")
        st.success("学習完了！")
        with open("oral_model_hybrid.pkl", "rb") as f:
            st.download_button("💾 モデルをスマホにダウンロード", f, file_name="oral_model_hybrid.pkl")
