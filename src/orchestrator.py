import json
import os
import sys
from typing import Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Muat variabel lingkungan dari berkas .env
load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def get_gemini_client() -> genai.Client:
    """Inisialisasi klien Google GenAI menggunakan GEMINI_API_KEY dari environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY belum diatur. Silakan isi GEMINI_API_KEY di berkas .env"
        )
    return genai.Client(api_key=api_key)


def load_ticket_knowledge() -> str:
    """Membaca data/raw_tickets.json sebagai referensi knowledge base agen teknis."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tickets_path = os.path.join(base_dir, "data", "raw_tickets.json")

    if not os.path.exists(tickets_path):
        return "Tidak ada data tiket referensi."

    try:
        with open(tickets_path, "r", encoding="utf-8") as f:
            tickets = json.load(f)

        lines = []
        for t in tickets:
            lines.append(
                f"- [{t.get('ticket_id', 'TCK')}] Kendala: {t.get('issue', '')} | "
                f"Solusi Standar: {t.get('solution', '')}"
            )
        return "\n".join(lines)
    except Exception:
        return "Tidak ada data tiket referensi."


def format_history_context(history: Optional[List[Dict[str, str]]]) -> str:
    """Memformat riwayat percakapan singkat untuk menjaga konteks tanya-jawab."""
    if not history:
        return ""
    recent = history[-4:]  # Ambil maksimal 4 giliran terakhir
    formatted = ["Riwayat Percakapan Sebelumnya:"]
    for item in recent:
        formatted.append(f"Pengguna: {item['query']}")
        formatted.append(f"Agen ({item['routed_to']}): {item['response']}")
    return "\n".join(formatted) + "\n\n"


def call_general_assistant(
    query: str, history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Menghasilkan jawaban untuk pertanyaan umum menggunakan General Support Agent."""
    system_instruction = (
        "Kamu adalah General Support Agent untuk aplikasi kasir PocketPOS. "
        "Tugasmu menjawab pertanyaan umum, sapaan, jadwal operasional (layanan dukungan pelanggan "
        "aktif 24 jam setiap hari kerja Senin-Jumat), informasi fitur dasar, dan panduan layanan pelanggan. "
        "Jawablah dalam bahasa Indonesia yang ramah, ringkas, dan jelas."
    )
    prompt = f"{format_history_context(history)}Pertanyaan Pengguna Saat Ini: {query}"

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )
        return (response.text or "").strip()
    except Exception as e:
        return (
            "Informasi operasional: Layanan dukungan pelanggan aktif 24 jam setiap hari kerja. "
            f"(Catatan: Gagal menghubungi LLM: {e})"
        )


def call_technical_model(
    query: str, history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Menghasilkan solusi teknis menggunakan Technical Specialist Agent + konteks tiket."""
    knowledge_base = load_ticket_knowledge()
    system_instruction = (
        "Kamu adalah Technical Specialist Agent otomatis untuk sistem kasir PocketPOS. "
        "Tugasmu mendiagnosis kendala teknis (seperti gagal sinkronisasi offline, masalah IndexedDB, "
        "printer thermal/Bluetooth macet, koneksi socket/jaringan, atau bug aplikasi) dan memberikan "
        "langkah troubleshooting yang akurat, terstruktur, dan ringkas dalam bahasa Indonesia.\n\n"
        "Gunakan referensi solusi tiket berikut jika relevan dengan masalah pengguna:\n"
        f"{knowledge_base}"
    )
    prompt = f"{format_history_context(history)}Keluhan/Pertanyaan Teknis Pengguna: {query}"

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return (response.text or "").strip()
    except Exception as e:
        return (
            "Analisis sistem: Ditemukan kegagalan koneksi socket lokal. "
            "Lakukan reset cache peramban atau muat ulang Service Worker. "
            f"(Catatan: Gagal menghubungi LLM: {e})"
        )


def apply_safety_guardrail(response_text: str) -> str:
    """Menambahkan peringatan keamanan wajib pada setiap respon teknis."""
    disclaimer = "\n[Catatan Keamanan: Pastikan cadangan basis data lokal telah dibuat sebelum menjalankan reset data.]"
    return response_text + disclaimer


def _fallback_keyword_route(query: str) -> str:
    """Fallback klasifikasi berbasis kata kunci jika pemanggilan API terkendala."""
    technical_keywords = [
        "error",
        "sinkron",
        "offline",
        "bug",
        "gagal",
        "printer",
        "macet",
        "bluetooth",
        "database",
        "indexeddb",
        "reset",
        "koneksi",
        "lambat",
        "crash",
    ]
    if any(kwd in query.lower() for kwd in technical_keywords):
        return "technical"
    return "general"


def route_intent(
    query: str, history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Menentukan rute agen ('technical' atau 'general') secara otomatis menggunakan Gemini."""
    router_instruction = (
        "Kamu adalah pengklasifikasi rute agen (Intent Router) untuk layanan pelanggan sistem kasir PocketPOS.\n"
        "Klasifikasikan pesan pengguna ke dalam tepat SATU kategori berikut:\n"
        "- 'technical': Jika pengguna melaporkan masalah teknis, error aplikasi, gagal sinkronisasi, "
        "mode offline, printer kasir/Bluetooth bermasalah, database/cache, bug, atau menanyakan tindak lanjut dari solusi teknis.\n"
        "- 'general': Jika pengguna hanya menyapa, menanyakan jam operasional, info layanan umum, "
        "harga/langganan umum, atau hal non-teknis lainnya.\n\n"
        "Jawab HANYA dengan satu kata: technical atau general."
    )
    prompt = f"{format_history_context(history)}Pesan Pengguna: {query}"

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=router_instruction,
                temperature=0.0,
            ),
        )
        label = (response.text or "").strip().lower()
        if "technical" in label:
            return "technical"
        if "general" in label:
            return "general"
        return _fallback_keyword_route(query)
    except Exception:
        return _fallback_keyword_route(query)


def run_orchestrator(
    query: str, history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, str]:
    """Menjalankan orkestrasi pemilihan rute agen otomatis dan pembuatan respon."""
    intent = route_intent(query, history=history)

    if intent == "technical":
        raw_output = call_technical_model(query, history=history)
        final_output = apply_safety_guardrail(raw_output)
        agent_type = "Technical Specialist Agent"
    else:
        final_output = call_general_assistant(query, history=history)
        agent_type = "General Support Agent"

    return {
        "query": query,
        "intent": intent,
        "routed_to": agent_type,
        "response": final_output,
    }


def run_demo_samples():
    """Menjalankan contoh pertanyaan statis untuk pengujian cepat."""
    samples = [
        "Bagaimana jadwal operasional tim support?",
        "Transaksi POS saya mendadak error waktu mode offline.",
    ]
    history: List[Dict[str, str]] = []
    for prompt in samples:
        res = run_orchestrator(prompt, history=history)
        history.append(res)
        print(f"\nPermintaan: {res['query']}")
        print(f"Rute Agen : {res['routed_to']}")
        print(f"Respon    : {res['response']}")


def start_interactive_cli():
    """Menjalankan sesi tanya-jawab interaktif di terminal."""
    print("=" * 65)
    print("  POCKETPOS AI ASSISTANT - INTERACTIVE Q&A (AUTO AGENT ROUTING)")
    print(f"  Model Aktif : {DEFAULT_MODEL}")
    print("  Agen Tersedia: General Support Agent | Technical Specialist Agent")
    print("-" * 65)
    print("  Ketik pertanyaan Anda lalu tekan Enter.")
    print("  Perintah khusus: 'reset' (hapus riwayat) | 'exit' / 'keluar' (berhenti)")
    print("=" * 65)

    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("\nAnda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSesi tanya-jawab diakhiri. Sampai jumpa!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "keluar", "quit", "q"}:
            print("\nTerima kasih telah menggunakan layanan PocketPOS. Sampai jumpa!")
            break

        if user_input.lower() == "reset":
            history.clear()
            print("[Sistem] Riwayat percakapan berhasil direset.")
            continue

        print("Menganalisis rute agen & menyusun jawaban...", end="\r", flush=True)
        result = run_orchestrator(user_input, history=history)
        history.append(result)

        print(" " * 50, end="\r")  # Bersihkan teks loading
        print(f"[Rute Agen : {result['routed_to']}]")
        print(f"Respon     :\n{result['response']}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo_samples()
    else:
        start_interactive_cli()



def start_interactive_cli():
    """Menjalankan sesi tanya-jawab interaktif di terminal."""
    print("=" * 65)
    print("  POCKETPOS AI ASSISTANT - INTERACTIVE Q&A (AUTO AGENT ROUTING)")
    print(f"  Model Aktif : {DEFAULT_MODEL}")
    print("  Agen Tersedia: General Support Agent | Technical Specialist Agent")
    print("-" * 65)
    print("  Ketik pertanyaan Anda lalu tekan Enter.")
    print("  Perintah khusus: 'reset' (hapus riwayat) | 'exit' / 'keluar' (berhenti)")
    print("=" * 65)

    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("\nAnda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSesi tanya-jawab diakhiri. Sampai jumpa!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "keluar", "quit", "q"}:
            print("\nTerima kasih telah menggunakan layanan PocketPOS. Sampai jumpa!")
            break

        if user_input.lower() == "reset":
            history.clear()
            print("[Sistem] Riwayat percakapan berhasil direset.")
            continue

        print("Menganalisis rute agen & menyusun jawaban...", end="\r", flush=True)
        result = run_orchestrator(user_input, history=history)
        history.append(result)

        print(" " * 50, end="\r")  # Bersihkan teks loading
        print(f"[Rute Agen : {result['routed_to']}]")
        print(f"Respon     :\n{result['response']}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo_samples()
    else:
        start_interactive_cli()
        print(f"Respon    : {res['response']}")