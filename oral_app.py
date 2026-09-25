import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(page_title="[oral] Diadochokinesis Analyzer", page_icon="👄", layout="centered")

st.title("👄 [oral]")
st.markdown("### Oral Diadochokinesis Analysis System")
st.markdown("スマートフォンのマイクに向かって「パ・タ・カ」と連続して発音してください。")

# ==========================================
# サイドバー: [oral] エンジン パラメータ設定
# ==========================================
with st.sidebar:
    st.header("⚙️ [oral] Engine Params")
    st.markdown("解析アルゴリズムの微調整を行います。")
    cutoff_hz = st.slider("ローパスフィルタ (Hz)", 5.0, 30.0, 15.0)
    threshold_ratio = st.slider("音量閾値の割合", 0.05, 0.50, 0.15)
    min_distance_sec = st.slider("最小発声間隔 (秒)", 0.05, 0.30, 0.10)
    st.info("※波形がうまく検出されない場合のみ調整してください。")

# ==========================================
# メイン: スマホマイクからの録音
# ==========================================
st.write("👇 下のマイクアイコンをタップして検査開始 / 停止")
audio_bytes = audio_recorder(text="タップして録音", recording_color="#ff4b4b", neutral_color="#4b4bff")

# ==========================================
# 解析ロジックと結果表示
# ==========================================
if audio_bytes:
    st.success("録音が完了しました。[oral] エンジンで解析を開始します...")
    st.audio(audio_bytes, format="audio/wav")
    st.divider()
    
    with st.spinner("音声波形を解析中..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # 1. 音声の読み込みとエンベロープ抽出
            sr_target = 16000
            y, sr = librosa.load(tmp_path, sr=sr_target)
            abs_y = np.abs(y)
            nyq = 0.5 * sr
            b, a = butter(3, cutoff_hz / nyq, btype='low')
            envelope = filtfilt(b, a, abs_y)
            envelope = np.maximum(envelope, 0)
            
            # 2. ピーク（発音）検出
            threshold_height = np.max(envelope) * threshold_ratio
            min_distance = int(min_distance_sec * sr)
            
            # 【改良版】プロミネンスを追加し、独立した破裂音だけを正確に拾う
            peaks, _ = find_peaks(
                envelope, 
                height=threshold_height, 
                distance=min_distance, 
                prominence=threshold_height * 0.5
            )
            count = len(peaks)
            
            if count < 3:
                st.error("検出された発音が少なすぎます。もう少し大きな声でハッキリと録音してください。")
            else:
                # 3. リズムと減衰の計算
                intervals_sec = np.diff(peaks) / sr
                cv_interval = (np.std(intervals_sec) / np.mean(intervals_sec)) * 100
                
                peak_amplitudes = envelope[peaks]
                peak_numbers = np.arange(1, count + 1)
                slope, intercept = np.polyfit(peak_numbers, peak_amplitudes, 1)
                
                n_samples = min(3, max(1, count // 2))
                first_mean = np.mean(peak_amplitudes[:n_samples])
                last_mean = np.mean(peak_amplitudes[-n_samples:])
                decay_rate = ((first_mean - last_mean) / first_mean) * 100 if first_mean > 0 else 0

                # ==========================================
                # 指標の表示
                # ==========================================
                st.subheader("📊 [oral] 解析結果")
                
                col1, col2 = st.columns(2)
                col1.metric("🗣️ 発音回数", f"{count} 回")
                col2.metric("⏱️ リズムCV値", f"{cv_interval:.1f} %", "15%未満が目安", delta_color="off")
                
                col3, col4 = st.columns(2)
                col3.metric("📉 音量減衰率", f"{decay_rate:.1f} %", "20%未満が目安", delta_color="off")
                col4.metric("⏱️ 測定時間目安", f"{(len(y)/sr):.1f} 秒")

                # ==========================================
                # 波形グラフ表示
                # ==========================================
                st.subheader("📈 波形・トレンドグラフ")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                
                # 波形グラフ
                time_axis = np.arange(len(y)) / sr
                ax1.plot(time_axis, y, label='Voice', color='lightgray')
                ax1.plot(time_axis, envelope, label='Envelope', color='blue')
                ax1.plot(peaks / sr, peak_amplitudes, "x", color='red', markersize=8)
                ax1.axhline(y=threshold_height, color='green', linestyle=':', label='Threshold')
                ax1.set_title("[oral] Waveform & Peak Detection (Prominence Applied)")
                ax1.set_xlabel("Time (sec)")
                ax1.set_ylabel("Amplitude")
                
                # トレンドグラフ
                ax2.bar(peak_numbers, peak_amplitudes, color='orange', alpha=0.5)
                ax2.plot(peak_numbers, slope * peak_numbers + intercept, color='red', label='Trend')
                ax2.set_title(f"[oral] Amplitude Decay Trend (Slope: {slope:.4f})")
                ax2.set_xlabel("N-th Pronunciation")
                ax2.set_ylabel("Amplitude")
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"[oral] エンジン処理中にエラーが発生しました: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
else:
    st.info("上のマイクアイコンをタップして検査を開始してください。")
