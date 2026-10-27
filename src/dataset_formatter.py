import json
import os

SYSTEM_PROMPT = (
    "Kamu adalah asisten teknis otomatis untuk sistem kasir PocketPOS. "
    "Berikan instruksi troubleshooting yang akurat, ringkas, dan jelas."
)

def convert_tickets_to_jsonl(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"Berkas masukan {input_file} tidak ditemukan.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        raw_tickets = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f:
        for ticket in raw_tickets:
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ticket["issue"]},
                    {"role": "assistant", "content": ticket["solution"]}
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Berhasil menghasilkan dataset fine-tuning di: {output_file}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_dir, "data", "raw_tickets.json")
    output_path = os.path.join(base_dir, "data", "train_data.jsonl")
    convert_tickets_to_jsonl(input_path, output_path)