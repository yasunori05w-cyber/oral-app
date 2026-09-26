import streamlit as st
import numpy as np
import librosa
import parselmouth
import matplotlib.pyplot as plt
import joblib
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(page_title="[oral] Hybrid AI Analyzer", page_icon="👄", layout="centered")

st.title("👄 [oral] Hybrid AI")
st.markdown("### ZCR × Praat 融合型オーラルディアドコキネシス")
st.markdown("マイクに向かって「パ・タ・カ」と連続して発音してください。")

# ==========================================
# サイドバー: パラメータ設定
# ==========================================
with st.sidebar:
    st.header("⚙️ [oral] Engine Params")
    onset_delta = st.slider("検知感度 (小さいほど敏感)", 0.01, 0.30, 0.10)
    min_distance_sec = st.slider("最小発声間隔 (秒)", 0.05, 0.30, 0.10)
    st.divider()
    
    # AIモデルの読み込み
    model_path = "oral_model_hybrid.pkl"
    clf = None
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        st.success("✅ ハイブリッドAIモデルをロード済")
    else:
        st.error("❌ `oral_model_hybrid.pkl` が見つかりません。GitHubにアップロードしてください。")

# ==========================================
# メイン画面
# ==========================================
st.write("👇 マイクアイコンをタップして検査開始 / 停止")
audio_bytes = audio_recorder(text="タップして録音", recording_color="#ff4b4b", neutral_color="#4b4bff")

if audio_bytes:
    st.success("録音が完了しました。ハイブリッドAIエンジンで解析を開始します...")
    st.audio(audio_bytes, format="audio/wav")
    st.divider()
    
    if clf is None:
        st.error("AIモデルがないため解析を中止しました。")
    else:
        with st.spinner("Onset検知・特徴抽出(ZCR+F2/F3)・AI分類を実行中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            try:
                # 1. 音声読み込みとカウント（Onset Detection）
                sr_target = 16000
                y, sr = librosa.load(tmp_path, sr=sr_target)
                
                wait_frames = int(sr * min_distance_sec / 512)
                onset_frames = librosa.onset.onset_detect(
                    y=y, sr=sr, wait=wait_frames, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=onset_delta
                )
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                count = len(onset_times)
                
                if count < 3:
                    st.error("検出された発音が少なすぎます。もう少し大きな声で録音してください。")
                else:
                    # 2. 特徴量抽出 (ZCR + Praat)
                    snd = parselmouth.Sound(tmp_path)
                    formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)
                    y_pre = librosa.effects.preemphasis(y)
                    
                    valid_times = []
                    features_for_ai = []
                    f2_list = []
                    f3_list = []
                    
                    for t in onset_times:
                        # ZCR (破裂の鋭さ)
                        start_idx = max(0, int((t - 0.02) * sr))
                        end_idx = min(len(y), int((t + 0.02) * sr))
                        segment_consonant = y_pre[start_idx:end_idx]
                        zcr = np.mean(librosa.feature.zero_crossing_rate(segment_consonant)) if len(segment_consonant) > 0 else 0
                        
                        # Praat F2/F3 (調音位置の響き)
                        target_time = t + 0.02
                        f2 = formants.get_value_at_time(2, target_time)
                        f3 = formants.get_value_at_time(3, target_time)
                        
                        if not np.isnan(f2) and not np.isnan(f3):
                            valid_times.append(t)
                            f2_list.append(f2)
                            f3_list.append(f3)
                            features_for_ai.append([zcr, f2, f3])

                    # 3. あなたのAIによる絶対判定
                    if features_for_ai:
                        X_infer = np.array(features_for_ai)
                        predictions = clf.predict(X_infer)
                        
                        syllable_counts = {"パ": np.sum(predictions == 0), 
                                           "タ": np.sum(predictions == 1), 
                                           "カ": np.sum(predictions == 2)}
                        
                        color_map = {0: 'blue', 1: 'orange', 2: 'green'}
                        colors = [color_map[p] for p in predictions]
                        
                        # ==========================================
                        # 結果の表示
                        # ==========================================
                        st.subheader("📊 [oral] 判定結果")
                        st.metric("🗣️ 総検知回数 (Onset)", f"{count} 回")
                        
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("👄 パ (AI判定)", f"{syllable_counts['パ']} 回")
                        sc2.metric("👅 タ (AI判定)", f"{syllable_counts['タ']} 回")
                        sc3.metric("🗣️ カ (AI判定)", f"{syllable_counts['カ']} 回")
                        st.caption("※学習モデルに基づく絶対評価 (青:パ / 橙:タ / 緑:カ)")

                        # ==========================================
                        # グラフ表示
                        # ==========================================
                        st.subheader("📈 調音空間マップ & 波形")
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
                        
                        # フォルマント空間マップ
                        ax1.scatter(f2_list, f3_list, c=colors, s=80, alpha=0.7, edgecolors='black')
                        ax1.set_title("Formant Space (AI Classified)")
                        ax1.set_xlabel("F2 (Hz)")
                        ax1.set_ylabel("F3 (Hz)")
                        ax1.grid(True, linestyle='--', alpha=0.5)
                        
                        # 波形と判定タイミング
                        time_axis = np.arange(len(y)) / sr
                        ax2.plot(time_axis, y, color='lightgray')
                        
                        env = np.abs(y)
                        for t, p_color in zip(valid_times, colors):
                            ax2.axvline(x=t, color=p_color, linestyle='--', alpha=0.7)
                            ax2.plot(t, np.max(env), "v", color=p_color, markersize=10)
                            
                        ax2.set_title("Waveform & AI Detection Timing")
                        ax2.set_xlabel("Time (sec)")
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("解析可能な特徴量が抽出できませんでした。")

            except Exception as e:
                st.error(f"[oral] エンジン処理中にエラーが発生しました: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
else:
    st.info("上のマイクアイコンをタップして検査を開始してください。")
