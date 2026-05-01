# ĐỀ TÀI 8.3: Hệ thống Nhận diện Hành động và Tương tác VR bằng AI (Action Recognition)

Dự án này tích hợp AI để phân tích chuyển động cơ thể từ camera, từ đó nhận diện hành động và điều khiển tương tác trong môi trường thực tế ảo (VR) sử dụng Unity.

## 🚀 Tính Năng Chính
- **Full-body Pose Estimation**: Sử dụng **MediaPipe Tasks API** để trích xuất 33 điểm neo (landmarks) trên cơ thể theo thời gian thực.
- **Action Recognition (LSTM)**: Nhận diện các hành động phức tạp theo chuỗi thời gian sử dụng mô hình Deep Learning LSTM (TensorFlow/Keras). Các hành động mặc định: `Idle`, `Walk Forward`, `Walk Backward`, `Turn Left`, `Turn Right`, `Stop`, `Jump`.
- **Anomaly Detection**: Hệ thống theo dõi vận tốc và sự thay đổi đột ngột của cơ thể để phát hiện các hành vi bất thường (rất hữu ích trong giám sát/an ninh).
- **Action Prediction**: Dự đoán hành động tiếp theo của người dùng dựa trên chuỗi lịch sử hành động.
- **VR Interaction Bridge**: Kết nối thời gian thực với Unity thông qua giao thức UDP (Peaceful-pie), ánh xạ trực tiếp hành động AI sang chuyển động trong game/VR.

## 🛠 Cài Đặt & Sử Dụng

### Yêu Cầu Hệ Thống
- Python 3.11+
- Unity 6000.4.4f1 (hoặc tương đương)
- Webcam

### Khởi Chạy Nhanh
Chỉ cần chạy file tự động:
```powershell
start_game.bat
```
File script sẽ tự động tìm Unity Editor, khởi động Project và bật AI Controller.

*(Lưu ý: Bạn phải nhấn nút Play trong Unity trước khi hệ thống AI có thể kết nối thành công).*

### Huấn Luyện AI Thủ Công (Thu Thập Dữ Liệu)
Nếu bạn muốn hệ thống nhận diện chính xác chuyển động cơ thể của riêng bạn:
1. Chạy `python data_collector.py` để ghi hình các cử chỉ.
2. Chạy `python train_lstm.py` để huấn luyện lại mô hình LSTM dựa trên dữ liệu vừa thu thập.

## 🧠 Công Nghệ Sử Dụng
- **Python**: TensorFlow, Keras, OpenCV, MediaPipe, Scikit-learn
- **Unity (C#)**: UDP Sockets, NavMesh, VR Simulation

## 📌 Điểm Mới & Ứng Dụng
Hệ thống này tăng cường tính **"Natural Interaction"** (Tương tác tự nhiên) trong VR, cho phép người chơi sử dụng toàn bộ cơ thể thay vì Controller truyền thống. Rất tiềm năng cho các ứng dụng:
- VR Gaming & Fitness (Thể dục thực tế ảo)
- VR Training (Huấn luyện kỹ năng)
- Hệ thống cảnh báo an ninh thông minh (Anomaly Detection)
