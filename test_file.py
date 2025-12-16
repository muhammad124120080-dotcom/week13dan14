# test_file.py
import os
import segyio

print("=== CHECKING FILE & FOLDER ===")
print(f"Current directory: {os.getcwd()}")
print(f"Data folder exists: {os.path.exists('data')}")

if os.path.exists("data"):
    print(f"\nFiles in 'data' folder:")
    for file in os.listdir("data"):
        print(f"  - {file}")
        
    # Cek file SEG-Y spesifik
    file_path = "data/Test_Post_Stack.sgy"
    if os.path.exists(file_path):
        print(f"\n✅ File ditemukan: {file_path}")
        print(f"   File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        # Coba baca dengan segyio
        try:
            with segyio.open(file_path, "r") as f:
                print(f"✅ SEG-Y bisa dibuka!")
                print(f"   Traces: {f.tracecount}")
                print(f"   Samples per trace: {f.bin[segyio.BinField.Samples]}")
                print(f"   Sample interval: {f.bin[segyio.BinField.Interval]} μs")
        except Exception as e:
            print(f"❌ Error membaca SEG-Y: {e}")
    else:
        print(f"\n❌ File TIDAK ditemukan: {file_path}")
        print("   Pastikan nama file tepat: 'Test_Post_Stack.sgy'")
else:
    print("❌ Folder 'data' TIDAK ADA!")
    print("   Buat folder: mkdir data")