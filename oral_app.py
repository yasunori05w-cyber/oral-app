import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.cluster import KMeans
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(page_title="[oral] Advanced DDK Analyzer", page_icon="👄", layout="centered")

st.title("👄 [oral] Advanced")
st.markdown("### AI音響分析・オーラルディアドコキネシス")
st.markdown("スマートフォンのマイクに向かって**「パ・タ・カ」の順番で**連続して発音してください。")

# ==========================================
# サイドバー: パラメータ設定
# ==========================================
with st.sidebar:
    st.header("⚙️ [oral] Engine Params")
    st.markdown("**波形検出チューニング**")
    cutoff_hz = st.slider("ローパスフィルタ (Hz)", 5.0, 30.0, 15.0)
    threshold_ratio = st.slider("音量閾値の割合", 0.05, 0.50, 0.15)
    min_distance_sec = st.slider("最小発声間隔 (秒)", 0.05, 0.30, 0.10)
    st.divider()
    st.markdown("**測定モード**")
    test_mode = st.radio("検査タイプ", ["パタカ交互反復 (SMR)", "単音反復 (AMR)"])

# ==========================================
# メイン画面
# ==========================================
st.write("👇 マイクアイコンをタップして検査開始 / 停止")
audio_bytes = audio_recorder(text="タップして録音", recording_color="#ff4b4b", neutral_color="#4b4bff")

if audio_bytes:
    st.success("録音が完了しました。[oral] AIエンジンで高度音響解析を開始します...")
    st.audio(audio_bytes, format="audio/wav")
    st.divider()
    
    with st.spinner("MFCC抽出・クラスター解析を実行中..."):
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
            
            # 2. ピーク（発音）検出 (プロミネンス適用)
            threshold_height = np.max(envelope) * threshold_ratio
            min_distance = int(min_distance_sec * sr)
            peaks, _ = find_peaks(
                envelope, 
                height=threshold_height, 
                distance=min_distance, 
                prominence=threshold_height * 0.5
            )
            count = len(peaks)
            
            if count < 3:
                st.error("検出された発音が少なすぎます。もう少し大きな声で録音してください。")
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
                
                # 4. 【高度化】MFCCによる音響特徴量の抽出
                mfcc_features = []
                distortions = []
                display_y_axis = [] # グラフ表示用の直感的な指標（周波数重心）
                
                # プレエンファシス処理（高域強調）
                y_pre = librosa.effects.preemphasis(y)
                
                for p in peaks:
                    # ピークの直前から直後（子音＋母音の移行部）を切り出す
                    start_idx = max(0, p - int(0.04 * sr))
                    end_idx = min(len(y), p + int(0.02 * sr))
                    segment = y_pre[start_idx:end_idx]
                    
                    if len(segment) > 0:
                        # 分類用：MFCC（13次元）の抽出
                        n_fft_size = min(512, len(segment)) 
                        if n_fft_size > 0:
                            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=n_fft_size, hop_length=n_fft_size//4)
                            mfcc_features.append(np.mean(mfcc, axis=1))
                        else:
                            mfcc_features.append(np.zeros(13))

                        # グラフ表示用：周波数の重心
                        cent = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
                        display_y_axis.append(cent)

                        # ひずみ量：スペクトル平坦度
                        start_dist = max(0, p - int(0.02 * sr))
                        end_dist = min(len(y), p + int(0.05 * sr))
                        flat = np.mean(librosa.feature.spectral_flatness(y=y[start_dist:end_dist]))
                        distortions.append(flat * 1000)
                    else:
                        mfcc_features.append(np.zeros(13))
                        display_y_axis.append(0)
                        distortions.append(0)
                
                feature_matrix = np.array(mfcc_features)
                display_y_axis = np.array(display_y_axis)
                distortions = np.array(distortions)
                
                # 5. K-Means法による パ・タ・カ 自動分類 (多次元MFCC対応)
                syllable_counts = {"パ": 0, "タ": 0, "カ": 0, "合計": count}
                labels = np.zeros(count, dtype=int)
                colors = ['gray'] * count
                
                if test_mode == "パタカ交互反復 (SMR)" and count >= 3:
                    # 13次元のMFCCデータをもとにAIが3つのグループに分ける
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(feature_matrix)

                    # 【ヒューリスティック・ラベリング】
                    # 患者が「パ・タ・カ」の順番で発音し始めたと仮定し、出現順にラベルを割り当てる
                    unique_clusters = []
                    for l in labels:
                        if l not in unique_clusters:
                            unique_clusters.append(l)
                        if len(unique_clusters) == 3:
                            break
                    
                    if len(unique_clusters) == 3:
                        pa_label, ta_label, ka_label = unique_clusters
                    else:
                        pa_label, ta_label, ka_label = 0, 1, 2 # フォールバック

                    syllable_counts["パ"] = np.sum(labels == pa_label)
                    syllable_counts["タ"] = np.sum(labels == ta_label)
                    syllable_counts["カ"] = np.sum(labels == ka_label)
                    
                    # カラーマップ（パ:青, タ:橙, カ:緑）
                    color_map = {pa_label: 'blue', ta_label: 'orange', ka_label: 'green'}
                    colors = [color_map[l] for l in labels]

                avg_distortion = np.mean(distortions)

                # ==========================================
                # 指標の表示
                # ==========================================
                st.subheader("📊 [oral] AI 解析結果")
                
                col1, col2 = st.columns(2)
                col1.metric("🗣️ 総発音回数", f"{count} 回")
                col2.metric("⏱️ リズムCV値 (ばらつき)", f"{cv_interval:.1f} %", "15%未満が目安", delta_color="off")
                
                col3, col4 = st.columns(2)
                col3.metric("⚠️ 平均ひずみ率 (ノイズ量)", f"{avg_distortion:.1f}", "高いほど息漏れ/不明瞭", delta_color="inverse")
                col4.metric("⏱️ 測定時間目安", f"{(len(y)/sr):.1f} 秒")
                
                if test_mode == "パタカ交互反復 (SMR)":
                    st.markdown("**音節別の推測回数（AI/MFCCクラスタリング）**")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("👄 パ", f"{syllable_counts['パ']} 回")
                    sc2.metric("👅 タ", f"{syllable_counts['タ']} 回")
                    sc3.metric("🗣️ カ", f"{syllable_counts['カ']} 回")
                    st.caption("※患者が「パ→タ→カ」の順で発音開始したと仮定して推測しています")

                # ==========================================
                # AI分析グラフ表示
                # ==========================================
                st.subheader("📈 音響特性マップ & 波形")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
                
                # トレンドグラフ（時間 vs 周波数重心・ひずみ）
                time_points = peaks / sr
                ax1.scatter(time_points, display_y_axis, c=colors, s=distortions*10 + 20, alpha=0.7)
                ax1.plot(time_points, display_y_axis, color='gray', linestyle=':', alpha=0.5)
                ax1.set_title("Syllable Map (Y: Spectral Centroid, Size: Distortion)")
                ax1.set_ylabel("Spectral Centroid (Hz)")
                
                # 波形グラフ
                time_axis = np.arange(len(y)) / sr
                ax2.plot(time_axis, y, label='Voice', color='lightgray')
                ax2.plot(time_axis, envelope, label='Envelope', color='darkblue', alpha=0.6)
                
                # 分類された色でピークをプロット
                for t, a, c in zip(time_points, envelope[peaks], colors):
                    ax2.plot(t, a, "x", color=c, markersize=8, markeredgewidth=2)
                    
                ax2.set_title("Waveform & Detected Syllables")
                ax2.set_xlabel("Time (sec)")
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"[oral] エンジン処理中にエラーが発生しました: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
else:
    st.info("上のマイクアイコンをタップして検査を開始してください。")
