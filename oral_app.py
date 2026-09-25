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
st.markdown("スマートフォンのマイクに向かって「パ・タ・カ」と連続して発音してください。")

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
    st.success("録音が完了しました。[oral] AIエンジンで音響解析を開始します...")
    st.audio(audio_bytes, format="audio/wav")
    st.divider()
    
    with st.spinner("波形ピーク・周波数特性・ひずみを解析中..."):
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
                
                # 4. 【新規】音響特徴量の抽出（各ピーク周辺の周波数重心とノイズ率）
                centroids = []
                distortions = []
                
                for p in peaks:
                    # ピークの少し前（子音部）からピーク後（母音部）までを切り出す
                    start_idx = max(0, p - int(0.02 * sr))
                    end_idx = min(len(y), p + int(0.05 * sr))
                    segment = y[start_idx:end_idx]
                    
                    if len(segment) > 0:
                        # 周波数の重心（高いほど「タ」寄り、低いほど「パ」寄り）
                        cent = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
                        centroids.append(cent)
                        # スペクトル平坦度（ノイズ成分＝構音のひずみ率として代用）
                        flat = np.mean(librosa.feature.spectral_flatness(y=segment))
                        distortions.append(flat * 1000) # 見やすいようにスケール調整
                    else:
                        centroids.append(0)
                        distortions.append(0)
                
                centroids = np.array(centroids)
                distortions = np.array(distortions)
                
                # 5. 【新規】K-Means法による パ・タ・カ 自動分類
                syllable_counts = {"パ": 0, "タ": 0, "カ": 0, "合計": count}
                labels = np.zeros(count)
                colors = ['red'] * count
                
                if test_mode == "パタカ交互反復 (SMR)" and count >= 3:
                    # 重心データをもとに3つのクラスターに分類
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(centroids.reshape(-1, 1))
                    cluster_centers = kmeans.cluster_centers_.flatten()
                    
                    # 重心の低い順に並べ替え（0:パ, 1:カ, 2:タ に該当）
                    sorted_idx = np.argsort(cluster_centers)
                    
                    pa_label = sorted_idx[0]
                    ka_label = sorted_idx[1]
                    ta_label = sorted_idx[2]
                    
                    syllable_counts["パ"] = np.sum(labels == pa_label)
                    syllable_counts["カ"] = np.sum(labels == ka_label)
                    syllable_counts["タ"] = np.sum(labels == ta_label)
                    
                    # グラフ描画用のカラーマップ
                    color_map = {pa_label: 'blue', ka_label: 'green', ta_label: 'orange'}
                    colors = [color_map[l] for l in labels]

                avg_distortion = np.mean(distortions)

                # ==========================================
                # 指標の表示
                # ==========================================
                st.subheader("📊 [oral] AI 解析結果")
                
                col1, col2 = st.columns(2)
                col1.metric("🗣️ 総発音回数", f"{count} 回")
                col2.metric("⏱️ リズムCV値 (ばらつき)", f"{cv_interval:.1f} %")
                
                col3, col4 = st.columns(2)
                col3.metric("⚠️ 平均ひずみ率 (ノイズ量)", f"{avg_distortion:.1f}", "高いほど息漏れ/不明瞭", delta_color="inverse")
                col4.metric("⏱️ 測定時間目安", f"{(len(y)/sr):.1f} 秒")
                
                if test_mode == "パタカ交互反復 (SMR)":
                    st.markdown("**音節別の推測回数（周波数特性による分類）**")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("👄 パ (両唇音)", f"{syllable_counts['パ']} 回")
                    sc2.metric("👅 タ (歯茎音)", f"{syllable_counts['タ']} 回")
                    sc3.metric("🗣️ カ (軟口蓋音)", f"{syllable_counts['カ']} 回")
                    st.caption("※青:パ相当(低域) / 橙:タ相当(高域) / 緑:カ相当(中域)")

                # ==========================================
                # AI分析グラフ表示
                # ==========================================
                st.subheader("📈 音響特性マップ & 波形")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
                
                # トレンドグラフ（時間 vs 周波数重心・ひずみ）
                time_points = peaks / sr
                scatter = ax1.scatter(time_points, centroids, c=colors, s=distortions*10 + 20, alpha=0.7)
                ax1.plot(time_points, centroids, color='gray', linestyle=':', alpha=0.5)
                ax1.set_title("Syllable Map (Y: Frequency Centroid, Size: Distortion)")
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
