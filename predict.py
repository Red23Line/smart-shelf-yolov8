from ultralytics import YOLO

# 1. Load model hasil training terbaik
model = YOLO(r"C:\Games\smart_shelf\runs\few_shot_experiment-2\weights\best.pt")

# 2. Jalankan deteksi dengan filter ketat (menghilangkan kotak ganda & berantakan)
results = model.predict(
    source=r"C:\Games\smart_shelf\dataset\images\train\shelf_empty.jpg",
    conf=0.65,      # Hanya tampilkan deteksi dengan tingkat keyakinan minimal 65%
    iou=0.30,       # Batasi tumpang tindih kotak maksimal 30% (menghapus kotak double)
    save=True,      # Simpan file gambar hasil deteksi
    project=r"C:\Games\smart_shelf\runs",
    name="predict_test",
    exist_ok=True   # Menimpa file lama di folder predict_test agar tidak membuat folder baru
)

print("\n--- DETEKSI BERSIH SELESAI ---")
print("Cek gambar hasil deteksi terbaru di: C:\\Games\\smart_shelf\\runs\\predict_test\\shelf_empty.jpg")