"""
PAKAI INI UNTUK REKOMENDASINYA TAPI PERBARUI LAGI TAMPILANNYA

COBA PAKAI LATIHANNYA HANYA SYNOPSIS
"""
import streamlit as st
import requests
import numpy as np
import re
import unicodedata
import pickle
import ast
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import time

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(page_title="Anime Recommender", layout="centered")

# ============================
# Load Model dan Encoder
# ============================
@st.cache_resource
def load_resources():
    encoder_model = load_model("anime_encoder_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer_ = pickle.load(f)
    with open("studio_encoder.pkl", "rb") as f:
        studio_enc = pickle.load(f)
    with open("genre_list.txt", "r") as f:
        genre_lst = [line.strip().lower() for line in f.readlines()]
    return encoder_model, tokenizer_, studio_enc, genre_lst

encoder, tokenizer, studio_encoder, genre_list = load_resources()
MAX_SEQ_LEN = 1000

# ============================
# Fungsi Membersihkan Teks
# ============================
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r'story: \d+\/\d+', ' ', text)
    text = re.sub(r'[“”"’]', '"', text)
    text = re.sub(r'\.\.\.+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r"[^a-zA-Z.!?'\s]", ' ', text)
    text = re.sub(r'(?<!\w)\.(?!\w)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize("NFKD", text)
    return text.strip()

def clean_studio(s):
    try:
        studios = ast.literal_eval(str(s)) if isinstance(s, str) else s
        if isinstance(studios, list) and studios:
            return studios[0].get("name", "unknown").strip().lower()
        elif isinstance(studios, dict):
            return studios.get("name", "unknown").strip().lower()
        else:
            return "unknown"
    except:
        return "unknown"

# ============================
# Fungsi Preprocess Input
# ============================
def preprocess_input(text, score, studio, genres):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_text = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    score_array = np.array([[score]])

    if studio in studio_encoder.classes_:
        studio_encoded = studio_encoder.transform([studio])
    else:
        studio_encoded = studio_encoder.transform(["unknown"])
    studio_cat = np.zeros((1, len(studio_encoder.classes_)))
    studio_cat[0, studio_encoded] = 1

    genres = [g.strip().lower() for g in genres]
    genre_vector = np.zeros((1, len(genre_list)))
    for g in genres:
        if g in genre_list:
            idx = genre_list.index(g)
            genre_vector[0, idx] = 1

    return [padded_text, studio_cat, genre_vector, score_array]

# ============================
# Ambil Data dari Jikan API
# ============================
def get_anime_info(mal_id):
    anime_url = f"https://api.jikan.moe/v4/anime/{mal_id}"
    review_url = f"https://api.jikan.moe/v4/anime/{mal_id}/reviews"

    anime_resp = requests.get(anime_url).json()
    if "data" not in anime_resp or not anime_resp["data"]:
        raise ValueError("Anime tidak ditemukan. Pastikan MAL ID yang dimasukkan benar.")

    review_resp = requests.get(review_url).json()

    data = anime_resp.get("data", {})
    synopsis = data.get("synopsis", "")
    score = float(data.get("score", 0)) / 10.0
    studio_raw = data.get("studios", [])
    studio = clean_studio(studio_raw)

    genre_names = [g.get("name", "").lower() for g in data.get("genres", [])]
    theme_names = [t.get("name", "").lower() for t in data.get("themes", [])]
    demo_names = [d.get("name", "").lower() for d in data.get("demographics", [])]
    genres = list(set(genre_names + theme_names + demo_names))

    reviews = [rev["review"] for rev in review_resp.get("data", [])]
    combined_review = " ".join(reviews) if reviews else ""
    combined_text = synopsis + " " + combined_review
    title = data.get("title", "Unknown")

    return combined_text, score, studio, genres, title

# ============================
# Fungsi Detail Anime Lengkap
# ============================
def tampilkan_detail_anime(mal_id):
    url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
    response = requests.get(url)
    if response.status_code != 200:
        st.warning("Gagal mengambil data anime.")
        return

    data = response.json().get("data", {})
    title = data.get("title_english") or data.get("title")
    title_jp = data.get("title_japanese", "")
    st.markdown(f"### {title} ({title_jp})")

    image_url = data.get("images", {}).get("jpg", {}).get("image_url")
    if image_url:
        st.image(image_url, width=200)

    synopsis = data.get("synopsis", "Tidak ada sinopsis.")
    st.markdown(f"**Sinopsis:** {synopsis}")

    studios = data.get("studios", [])
    studio_names = ", ".join([studio["name"] for studio in studios]) if studios else "Tidak diketahui"
    st.markdown(f"**Studio:** {studio_names}")

    trailer_url = data.get("trailer", {}).get("url")
    if trailer_url:
        st.video(trailer_url)

    all_genres = []
    for key in ["genres", "themes", "demographics", "explicit_genres"]:
        all_genres.extend([g["name"] for g in data.get(key, [])])
    genres_str = ", ".join(all_genres) if all_genres else "Tidak ada genre."
    st.markdown(f"**Genre:** {genres_str}")

    rank = data.get("rank", "Tidak diketahui")
    st.markdown(f"**Peringkat:** {rank}")

    episodes = data.get("episodes", "Tidak diketahui")
    st.markdown(f"**Jumlah Episode:** {episodes}")

    streaming_links = data.get("streaming", [])
    if streaming_links:
        for stream in streaming_links:
            st.markdown(f"[Tonton di {stream['name']}]({stream['url']})")
    else:
        st.markdown("_Tidak ada link streaming yang tersedia._")

# ============================
# Load Embeddings
# ============================
@st.cache_data
def load_anime_data():
    embeddings = np.load("anime_embeddings.npy")
    with open("anime_info.pkl", "rb") as f:
        info = pickle.load(f)
    return embeddings, info

anime_embeddings, anime_info = load_anime_data()
anime_ids = np.array([anime['mal_id'] for anime in anime_info])

def get_top_recommendations(embedding, embeddings, info, similarity_threshold=0.85, initial_results=5, max_results=50):
    similarities = cosine_similarity(embedding, embeddings)[0]
    top_indices = np.where(similarities >= similarity_threshold)[0]
    sorted_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    results = []
    total = min(max_results, len(sorted_indices))
    progress_bar = st.progress(0, text="Mengambil rekomendasi anime...")

    # Hanya mengambil initial_results pertama untuk ditampilkan
    for i, idx in enumerate(sorted_indices[:initial_results]):
        if len(results) >= initial_results:
            break

        sim_score = similarities[idx]
        if sim_score < similarity_threshold:
            continue

        anime = info[idx]
        mal_id = anime["mal_id"]

        try:
            synopsis, score, studio, genres, title = get_anime_info(mal_id)
            detail_url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
            detail_resp = requests.get(detail_url)
            if detail_resp.status_code != 200:
                continue
            detail_data = detail_resp.json().get("data", {})

            image = detail_data.get("images", {}).get("jpg", {}).get("image_url", "")
            trailer_url = detail_data.get("trailer", {}).get("url")
            rank = detail_data.get("rank", "Tidak diketahui")
            episodes = detail_data.get("episodes", "Tidak diketahui")
            streaming_links = [
                {"name": stream["name"], "url": stream["url"]}
                for stream in detail_data.get("streaming", [])
            ]

            anime_data = {
                "title": title,
                "score": score * 10,
                "studio": studio,
                "genres": genres,
                "synopsis": synopsis,
                "similarity": round(sim_score * 100, 2),
                "image": image,
                "trailer_url": trailer_url,
                "rank": rank,
                "episodes": episodes,
                "streaming_links": streaming_links,
                "mal_id": mal_id
            }

            results.append(anime_data)

            # Update progress
            progress_bar.progress((len(results)) / initial_results, text=f"Mengambil rekomendasi... ({len(results)}/{initial_results})")

            time.sleep(0.7)  # Delay antar request untuk menghindari rate limit

        except Exception as e:
            print(f"Gagal mengambil data untuk mal_id {mal_id}: {e}")
            continue

    progress_bar.empty()  # Hapus progress bar setelah selesai
    
    # Simpan semua indeks yang memenuhi syarat untuk digunakan nanti
    st.session_state.all_recommendation_indices = sorted_indices
    st.session_state.all_similarities = similarities
    st.session_state.current_results = results
    st.session_state.current_page = 0
    
    return results

def load_more_recommendations(page, per_page=5):
    start_idx = page * per_page
    end_idx = start_idx + per_page
    
    results = []
    progress_bar = st.progress(0, text="Memuat lebih banyak rekomendasi...")
    
    for i, idx in enumerate(st.session_state.all_recommendation_indices[start_idx:end_idx]):
        sim_score = st.session_state.all_similarities[idx]
        
        anime = anime_info[idx]
        mal_id = anime["mal_id"]

        try:
            synopsis, score, studio, genres, title = get_anime_info(mal_id)
            detail_url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
            detail_resp = requests.get(detail_url)
            if detail_resp.status_code != 200:
                continue
            detail_data = detail_resp.json().get("data", {})

            image = detail_data.get("images", {}).get("jpg", {}).get("image_url", "")
            trailer_url = detail_data.get("trailer", {}).get("url")
            rank = detail_data.get("rank", "Tidak diketahui")
            episodes = detail_data.get("episodes", "Tidak diketahui")
            streaming_links = [
                {"name": stream["name"], "url": stream["url"]}
                for stream in detail_data.get("streaming", [])
            ]

            anime_data = {
                "title": title,
                "score": score * 10,
                "studio": studio,
                "genres": genres,
                "synopsis": synopsis,
                "similarity": round(sim_score * 100, 2),
                "image": image,
                "trailer_url": trailer_url,
                "rank": rank,
                "episodes": episodes,
                "streaming_links": streaming_links,
                "mal_id": mal_id
            }

            results.append(anime_data)
            progress_bar.progress((i+1)/per_page, text=f"Memuat rekomendasi... ({i+1}/{per_page})")
            time.sleep(0.7)
            
        except Exception as e:
            print(f"Gagal mengambil data untuk mal_id {mal_id}: {e}")
            continue
            
    progress_bar.empty()
    return results

# ============================
# Fungsi Pencarian Anime berdasarkan Judul
# ============================
def search_anime_by_title(title):
    url = f"https://api.jikan.moe/v4/anime?q={title}&limit=10"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("data", [])
        return [{"mal_id": anime["mal_id"], "title": anime["title"]} for anime in results]
    else:
        return []

# ============================
# Streamlit UI
# ============================


st.title("🎌 Rekomendasi Anime Berdasarkan MAL ID")
st.markdown(
    "Masukkan **MAL ID** dari anime yang ingin kamu cari rekomendasinya.\n\n"
    "Aplikasi ini akan mengambil data sinopsis dan review dari Jikan API,\n"
    "menggabungkannya, dan mencari anime paling mirip berdasarkan model encoder.\n\n"
    "**Contoh ID:** 5114 (Fullmetal Alchemist: Brotherhood), 9253 (Steins;Gate)"
)

# Input judul anime
query_input = st.text_input("Cari anime berdasarkan judul", "")

if query_input.strip():
    with st.spinner("Mencari anime..."):
        search_results = search_anime_by_title(query_input)
        
        if search_results:
            st.subheader("🔍 Hasil Pencarian:")
            for i, result in enumerate(search_results):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"- **{result['title']}**")
                with col2:
                    st.code(result['mal_id'], language='text')  # untuk memudahkan menyalin
        else:
            st.warning("Tidak ada anime yang ditemukan dengan judul tersebut.")

mal_id_input = st.text_input("Masukkan MAL ID anime", "")

if mal_id_input.strip().isdigit():
    mal_id = int(mal_id_input)
    with st.spinner("Mengambil dan memproses data dari Jikan API..."):
        try:
            anime_data = get_anime_info(mal_id)
            combined_text, score, studio, genres, title = anime_data
            st.subheader(f"📌 Anime Referensi: {title}")
            input_data = preprocess_input(combined_text, score, studio, genres)
            embedding = encoder.predict(input_data)

            st.success("✅ Embedding berhasil dibuat!")

            with st.expander("📖 Lihat Detail Anime"):
                tampilkan_detail_anime(mal_id)

            st.subheader("🎯 Rekomendasi Anime Mirip:")
            if "recommendations" not in st.session_state or st.session_state.get("mal_id") != mal_id:
                recommendations = get_top_recommendations(embedding, anime_embeddings, anime_info, initial_results=5)
                recommendations = [rec for rec in recommendations if rec['mal_id'] != mal_id]
                st.session_state.recommendations = recommendations
                st.session_state.page = 0
                st.session_state.mal_id = mal_id
            else:
                recommendations = st.session_state.recommendations
            recommended_count = len(st.session_state.all_recommendation_indices) if "all_recommendation_indices" in st.session_state else len(recommendations)
            st.markdown(f"**Jumlah rekomendasi yang ditemukan: {recommended_count} anime.**")

            if recommended_count == 0:
                st.warning("Tidak ada rekomendasi ditemukan dengan kemiripan > 85%.")
            else:
                # Tampilkan rekomendasi yang sudah dimuat
                for rec in recommendations:
                    # Batasi sinopsis hanya 500 kata
                    synopsis_words = rec['synopsis'].split()
                    limited_synopsis = ' '.join(synopsis_words[:200])
                    st.markdown(f"### {rec['title']}")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(rec["image"], width=150)
                    with col2:
                        st.markdown(f"⭐ **Rating:** {rec['score']:.2f} &nbsp;&nbsp;&nbsp; 🔁 **Kemiripan:** {rec['similarity']}%")
                        st.markdown(f"**Studio:** {rec['studio']}")
                        st.markdown(f"**Genre:** {', '.join(rec['genres'])}")
                        st.markdown(f"**Peringkat:** {rec['rank']}")
                        st.markdown(f"**Jumlah Episode:** {rec['episodes']}")
                    with st.expander("Lihat Synopsis"):
                        st.markdown(f"{limited_synopsis}...")
                    if rec["trailer_url"]:
                        st.video(rec["trailer_url"])
                    if rec["streaming_links"]:
                        st.markdown("**Streaming:**")
                        for stream in rec["streaming_links"]:
                            st.markdown(f"- [{stream['name']}]({stream['url']})")
                    else:
                        st.markdown("_Tidak ada link streaming yang tersedia._")
                    st.markdown("---")

                # Tombol untuk memuat lebih banyak
                if st.button("Tampilkan Lebih Banyak") and "all_recommendation_indices" in st.session_state:
                    st.session_state.page += 1
                    new_recommendations = load_more_recommendations(st.session_state.page)
                    st.session_state.recommendations.extend(new_recommendations)
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mengambil data: {e}")

# ============================
# Tambahan Styling
# ============================
st.markdown("""
    <style>
        .main {padding-top: 2rem;}
        img {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)