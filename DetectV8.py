import cv2
from ultralytics import YOLO
from sort.sort import Sort
import numpy as np

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')

# SORT nesne takip algoritmasını başlat
tracker = Sort()

# Çizgi koordinatları ve sayaçları
line_coords = {
    "Gelen Arac": [(767, 462), (1203, 462)],  # Sol çizgi (Mavi)
    "Giden Arac": [(1281, 462), (1616, 462)]  # Sağ çizgi (Yeşil)
}

# Araç sayacı
vehicle_count = {
    "Gelen Arac": 0,
    "Giden Arac": 0
}

# Geçiş yapılan araçların ID'lerini saklayan yapı
already_counted = {
    "Gelen Arac": set(),
    "Giden Arac": set()
}

# Trafik yoğunluğunu hesaplayan fonksiyon
def calculate_traffic_percentage(vehicle_count):
    """
    Trafik yoğunluğunu yüzde olarak hesaplar.
    """
    total_vehicles = sum(vehicle_count.values())
    if total_vehicles == 0:
        return {"Gelen Arac": 0, "Giden Arac": 0}

    percentages = {
        line_name: (count / total_vehicles) * 100
        for line_name, count in vehicle_count.items()
    }
    return percentages

# Araçların ağırlık merkezinin bir çizgiden geçip geçmediğini kontrol eden fonksiyon
def is_passing_line(center, line_start, line_end):
    """
    Bir aracın merkezi bir çizgiyi geçiyor mu kontrol eder.
    """
    cx, cy = center
    x1, y1 = line_start
    x2, y2 = line_end

    # Çizgi yatay olduğu için Y ekseni kontrol edilir
    if y1 == y2:  # Yatay çizgi
        if min(x1, x2) <= cx <= max(x1, x2) and abs(cy - y1) <= 5:
            return True
    return False

# Video işleme
video_path = 'traffic_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Sabit çözünürlüğe ayarla
    frame = cv2.resize(frame, (1920, 1080))

    # YOLO tahmini
    results = model(frame)

    detections = []  # SORT için tespit listesi
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box koordinatları
        conf = box.conf[0].cpu().numpy()  # Güven skoru
        cls = int(box.cls[0].cpu().numpy())  # Sınıf etiketi

        # Sadece araç sınıflarını takip et
        if model.names[cls] in ['car', 'truck', 'bus'] and conf > 0.5:
            detections.append([x1, y1, x2, y2, conf])  # 5 değerli liste

    # SORT algoritmasını kullanarak tespitleri takip edin
    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))
    else:
        tracked_objects = []

    # Takip edilen araçları çiz ve say
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)  # Takip edilen obje bilgileri
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center = (center_x, center_y)

        # Araç dikdörtgenini çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Ağırlık merkezine küçük bir işaret koy
        cv2.circle(frame, center, 2, (0, 0, 255), -1)  # Küçük kırmızı daire

        # Çizgilerden geçiş kontrolü
        for line_name, (line_start, line_end) in line_coords.items():
            if is_passing_line(center, line_start, line_end) and obj_id not in already_counted[line_name]:
                vehicle_count[line_name] += 1
                already_counted[line_name].add(obj_id)

    # Çizgileri çiz ve üzerine yazıyı ekle
    for line_name, (start, end) in line_coords.items():
        color = (0, 255, 0) if line_name == "Giden Arac" else (255, 0, 0)
        cv2.line(frame, start, end, color, 2)

        # Çizgiye sayaç yazısı ekle, 50 piksel sola kaydır
        text_position = (start[0] - 50, start[1] - 10)  # Çizginin hemen üstüne yazı koy
        cv2.putText(frame, f"{line_name}: {vehicle_count[line_name]}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Trafik yoğunluğunu hesapla
    traffic_percentages = calculate_traffic_percentage(vehicle_count)

    # Sol üstte yazılar için arka plan (beyaz dikdörtgen)
    cv2.rectangle(frame, (30, 20), (400, 200), (255, 255, 255), -1)  # Beyaz arka plan

    # Trafik yoğunluğunu ekranda göster
    cv2.putText(frame, "Trafik Yogunlugu", (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Gelen Arac: {traffic_percentages['Gelen Arac']:.1f}%", (50, 110), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 0), 2)
    cv2.putText(frame, f"Giden Arac: {traffic_percentages['Giden Arac']:.1f}%", (50, 160), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 255, 0), 2)

    # Video penceresi
    cv2.imshow('Traffic Detection with Lines', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
