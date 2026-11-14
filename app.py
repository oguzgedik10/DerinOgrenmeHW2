import gradio as gr
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

def nesne_tespiti(gorsel, guven_esigi):
    if gorsel is None:
        return None, "Lütfen bir fotoğraf yükleyin."
    sonuclar = model.predict(source=gorsel, conf=guven_esigi)
    cizimli_gorsel = sonuclar[0].plot()[:, :, ::-1]
    
    tespitler = []
    for box in sonuclar[0].boxes:
        sinif_id = int(box.cls[0])
        sinif_adi = model.names[sinif_id]
        skor = float(box.conf[0])
        tespitler.append(f"{sinif_adi}: %{skor*100:.1f}")
        
    ozet = "\n".join(tespitler) if tespitler else "Belirlenen eşik değerinde nesne tespit edilemedi."
    return Image.fromarray(cizimli_gorsel), ozet

arayuz = gr.Interface(
    fn=nesne_tespiti,
    inputs=[
        gr.Image(type="pil", label="Fotoğraf Yükle"),
        gr.Slider(minimum=0.10, maximum=0.90, value=0.35, step=0.05, label="Güven Eşiği (Confidence)")
    ],
    outputs=[
        gr.Image(type="pil", label="Model Tespit Çıktısı"),
        gr.Textbox(label="Tespit Edilen Sınıflar ve Güven Skorları")
    ],
    title="YOLOv8n - Pascal VOC Nesne Tespiti Uygulaması",
    description="Pascal VOC (20 Sınıf) veri seti ile eğitilmiş YOLOv8n modeli kullanılarak geliştirilen nesne tespiti arayüzü."
)

if __name__ == "__main__":
    arayuz.launch()