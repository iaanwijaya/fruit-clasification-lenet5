import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# 1. KONFIGURASI HALAMAN WEB
st.set_page_config(
    page_title="Klasifikasi Kesegaran Buah - LeNet-5",
    page_icon="🍎",
    layout="centered"
)


# 2. DEFINISI ARSITEKTUR MODEL (HARUS SAMA PERSIS)
class LeNet5Custom(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5Custom, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 3. FUNGSI MEMUAT MODEL & PREPROCESSING
@st.cache_resource
def load_model():
    device = torch.device("cpu") # Gunakan CPU untuk inferensi web yang ringan
    model = LeNet5Custom(num_classes=2)
    # Muat bobot yang sudah dilatih
    model.load_state_dict(torch.load('lenet_buah_model.pth', map_location=device))
    model.eval() # Wajib set ke mode evaluasi!
    return model

def preprocess_image(image):
    # Transformasi harus sama persis dengan tahap training/validasi
    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0) # Tambahkan dimensi batch: (1, 3, 64, 64)
    return tensor

# 4. ANTARMUKA PENGGUNA (UI STREAMLIT)
st.title("🍎 Klasifikasi Kesegaran Buah")
st.markdown(
    "Aplikasi berbasis **Deep Learning (Modifikasi Arsitektur LeNet-5)** untuk mendeteksi "
    "apakah buah dalam kondisi **Segar (Fresh)** atau **Busuk (Rotten)**."
)
st.divider()

# Muat model
try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ File model `'lenet_buah_model.pth'` tidak ditemukan! Pastikan file tersebut berada di folder yang sama dengan `app.py`.")
    model_loaded = False

# Widget Upload Gambar
uploaded_file = st.file_uploader(
    "Unggah foto buah (Format: JPG, JPEG, PNG, WEBP)", 
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None and model_loaded:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Citra yang Diunggah", use_container_width=True)
        
    with col2:
        st.subheader("Hasil Analisis Model")
        with st.spinner("Sedang menganalisis pola citra..."):
            # Preprocessing & Prediksi
            input_tensor = preprocess_image(image)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                # Hitung probabilitas menggunakan Softmax
                probabilities = F.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            # Pemetaan indeks kelas (sesuai urutan alfabet folder: 0 = fresh, 1 = rotten)
            class_names = ['Fresh (Segar)', 'Rotten (Busuk)']
            predicted_label = class_names[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            # Tampilkan hasil dengan visualisasi warna berbeda
            if predicted_idx.item() == 0:
                st.success(f"**Status:** {predicted_label}")
            else:
                st.error(f"**Status:** {predicted_label}")
                
            st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence_score:.2f}%")
            
            # Tampilkan rincian probabilitas kedua kelas
            st.write("**Rincian Probabilitas:**")
            st.progress(probabilities[0].item(), text=f"Fresh: {probabilities[0].item()*100:.1f}%")
            st.progress(probabilities[1].item(), text=f"Rotten: {probabilities[1].item()*100:.1f}%")

st.divider()
st.caption("Proyek UAS Pengenalan Pola • Arsitektur LeNet-5 (Resolusi Input 64x64)")