# Agentic & LLM Fine-Tuning Pipeline Simulation

Repositori ini memuat simulasi dan implementasi asisten AI untuk sistem kasir **PocketPOS**:
- **Interactive Q&A & Automatic Agent Routing**: CLI tanya-jawab interaktif yang secara otomatis memilih rute agen (**General Support Agent** atau **Technical Specialist Agent**) menggunakan **Google Gemini (`gemini-3.1-flash-lite`)**, dilengkapi konteks *knowledge base* dari [`data/raw_tickets.json`](data/raw_tickets.json) serta *Safety Guardrail*.
- **Dataset Pipeline for Fine-Tuning**: Konversi data percakapan tiket menjadi format JSONL (*chat template*) untuk pelatihan LLM.

## Persiapan & Instalasi

1. Instal dependensi yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```

2. Konfigurasi API Key di berkas `.env` (salin dari `.env.example` jika belum ada):
   ```env
   GEMINI_API_KEY=masukkan_api_key_gemini_anda
   GEMINI_MODEL=gemini-3.1-flash-lite
   ```

## Cara Menjalankan

1. **Jalankan Tanya-Jawab Interaktif (Otomatis Pilih Rute Agen)**:
   ```bash
   python src/orchestrator.py
   ```
   - Ketik pertanyaan umum (misal: *"Halo, jam berapa layanan support buka?"*) → otomatis diarahkan ke **General Support Agent**.
   - Ketik kendala teknis (misal: *"Kasir saya gagal sinkron waktu offline"* atau *"Printer kasir macet"*) → otomatis diarahkan ke **Technical Specialist Agent** dengan solusi berbasis tiket dan catatan keamanan.
   - Ketik `reset` untuk menghapus riwayat percakapan, atau `exit` / `keluar` untuk mengakhiri sesi.

2. **Jalankan Mode Demo Singkat (Tanpa Input Manual)**:
   ```bash
   python src/orchestrator.py --demo
   ```

3. **Jalankan Konversi Data Tiket ke JSONL (Fine-Tuning Dataset)**:
   ```bash
   python src/dataset_formatter.py
   ```

   ```
   python src/dataset_formatter.py