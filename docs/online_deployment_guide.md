# Panduan Deployment Online SINYAL (IRDSC 2026 — Tim Bismillah Bisa)

Dokumen ini berisi panduan langkah demi langkah untuk menjalankan aplikasi web **SINYAL** secara **online** menggunakan berkas konfigurasi yang telah disiapkan di proyek ini ([`Dockerfile`](../Dockerfile), [`render.yaml`](../render.yaml), dan [`Procfile`](../Procfile)).

---

## Opsi 1: Link Publik Instan dari Laptop (Tanpa Upload Cloud — 1 Menit)

Gunakan opsi ini saat Anda sedang presentasi/demo langsung di hadapan juri dan ingin membagikan link HTTPS yang bisa dibuka di HP atau laptop juri.

### A. Menggunakan Port Forwarding Editor (VS Code / Antigravity)
1. Pastikan server lokal sedang berjalan di port `8000`:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
2. Buka panel bawah editor (di sebelah tab **TERMINAL**), klik tab **PORTS**.
3. Klik tombol **Forward a Port** lalu ketik **`8000`** dan tekan **Enter**.
4. Klik kanan pada baris port `8000` $\rightarrow$ pilih **Port Visibility** $\rightarrow$ ubah menjadi **Public**.
5. Salin alamat **Forwarded Address** (contoh: `https://xxxx-8000.asse.devtunnels.ms`) dan bagikan ke juri.

### B. Menggunakan Cloudflare Quick Tunnel (Tanpa Akun)
1. Biarkan server Uvicorn menyala di port `8000`.
2. Buka terminal PowerShell baru, lalu jalankan:
   ```powershell
   winget install --id Cloudflare.cloudflared
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. Salin link `https://xxxx.trycloudflare.com` yang muncul di kotak informasi terminal.

---

## Opsi 2: Deploy Permanen Gratis 24 Jam di Render.com (Via GitHub)

Proyek ini sudah dilengkapi dengan berkas [`render.yaml`](../render.yaml) dan pengecualian `!models/sinyal_model.joblib` pada [`.gitignore`](../.gitignore) sehingga model terlatih ikut ter-upload ke GitHub tanpa perlu *training* ulang di server cloud.

1. **Push Proyek ke Repository GitHub Anda:**
   Repository target: `https://github.com/deakusuma05/SINYAL-AI-Humanitarian`
   - Klik tombol **Deploy to Render** langsung di halaman utama GitHub Anda:
     https://render.com/deploy?repo=https://github.com/deakusuma05/SINYAL-AI-Humanitarian
2. **Atau Deploy Manual di Render.com:**
   - Masuk ke [https://dashboard.render.com](https://dashboard.render.com).
   - Klik **New +** $\rightarrow$ pilih **Blueprint** (atau **Web Service**).
   - Hubungkan repository GitHub Anda.
   - Jika memilih **Blueprint**, Render akan otomatis membaca [`render.yaml`](../render.yaml).
   - Jika memilih **Web Service** manual, isi:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - Klik **Create Web Service**. Dalam 2–3 menit, aplikasi Anda akan aktif di `https://sinyal-humanitarian-triage.onrender.com`.

---

## Opsi 3: Deploy Permanen Gratis di Hugging Face Spaces (Sangat Direkomendasikan untuk AI/ML)

Hugging Face Spaces menyediakan spesifikasi gratis yang sangat besar (**2 vCPU + 16 GB RAM**) dan mendukung [`Dockerfile`](../Dockerfile) yang sudah dibuat di proyek ini:

1. Buka [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Isi **Space name** (misalnya: `sinyal-triage`), pilih **License:** `cc-by-4.0`, lalu pada bagian **Select the Space SDK** pilih **Docker** $\rightarrow$ **Blank**.
3. Pilih **Public** lalu klik **Create Space**.
4. Upload file/folder berikut ke Space tersebut (bisa lewat tombol **Files** $\rightarrow$ **+ Add file** $\rightarrow$ **Upload files** di web Hugging Face atau lewat `git push`):
   - `Dockerfile`
   - `.dockerignore`
   - `requirements.txt`
   - `README.md`
   - Folder `src/`
   - Folder `web/`
   - Folder `models/` (`sinyal_model.joblib`, `model_manifest.json`, `decision_policy.json`)
   - Folder `reports/phase2/` (`final_test_metrics.json`, `final_evaluation_lock.json`, `model_comparison_validation.csv`)
5. Hugging Face Spaces akan otomatis mem-build [`Dockerfile`](../Dockerfile) pada port `7860` dan menampilkan dashboard **SINYAL** secara online!

