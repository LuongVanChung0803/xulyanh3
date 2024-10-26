import cv2
import numpy as np
from collections import Counter

# Hàm để tải dữ liệu huấn luyện từ file
def load_training_data(file_path='train_data.txt'):
    features = []  # Mảng chứa các đặc trưng của ảnh huấn luyện
    labels = []    # Mảng chứa nhãn của ảnh huấn luyện
    
    # Mở file và đọc từng dòng
    with open(file_path, 'r') as f:
        for line in f:
            # Chia các giá trị đặc trưng và nhãn, sau đó thêm vào danh sách
            *feature_values, label = line.strip().split(',')
            features.append(list(map(float, feature_values)))  # Chuyển đặc trưng thành float
            labels.append(label)  # Thêm nhãn vào danh sách
    
    return np.array(features), np.array(labels)  # Trả về mảng đặc trưng và nhãn

# Hàm để trích xuất đặc trưng từ ảnh đầu vào
def extract_features(image_path):
    # Đọc ảnh ở chế độ xám
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Thay đổi kích thước ảnh về 64x64
    image = cv2.resize(image, (64, 64))
    # Làm phẳng ảnh thành một mảng 1 chiều để dễ xử lý
    features = image.flatten()
    return features  # Trả về đặc trưng của ảnh

# Hàm tính khoảng cách Euclid giữa hai vector
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))  # Trả về khoảng cách Euclid

# Hàm để dự đoán nhãn của ảnh đầu vào sử dụng thuật toán k-NN
def predict_label(image_path, training_features, training_labels, k=3):
    # Trích xuất đặc trưng của ảnh đầu vào
    features = extract_features(image_path)
    distances = []  # Danh sách để lưu khoảng cách đến từng ảnh huấn luyện
    
    # Tính khoảng cách giữa ảnh đầu vào và mỗi ảnh huấn luyện
    for i, train_feature in enumerate(training_features):
        distance = euclidean_distance(features, train_feature)
        distances.append((distance, training_labels[i]))  # Lưu khoảng cách và nhãn tương ứng
    
    # Sắp xếp khoảng cách tăng dần để tìm các điểm gần nhất
    distances.sort(key=lambda x: x[0])
    # Lấy k nhãn gần nhất
    k_nearest_labels = [label for _, label in distances[:k]]
    # Tìm nhãn phổ biến nhất trong k nhãn gần nhất
    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_label  # Trả về nhãn dự đoán

# Hàm chính của chương trình
def main():
    # Tải dữ liệu huấn luyện từ file
    features, labels = load_training_data('train_data.txt')
    
    # Đường dẫn ảnh đầu vào
    input_image_path = '/XLA/NEW/input1.jpg'
    # Dự đoán nhãn của ảnh đầu vào với k=15
    predicted_label = predict_label(input_image_path, features, labels, k=15)
    
    # Đọc ảnh màu để thêm nhãn dự đoán lên ảnh
    image = cv2.imread(input_image_path)
    
    # Thêm nhãn dự đoán lên ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font chữ cho nhãn
    cv2.putText(image, f'Du doan: {predicted_label}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Hiển thị ảnh với nhãn dự đoán
    cv2.imshow('Du doan anh', image)
    cv2.waitKey(0)  # Đợi người dùng nhấn phím bất kỳ để đóng cửa sổ hiển thị
    cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị
    
    # Lưu ảnh với nhãn dự đoán vào file mới
    output_image_path = '/XLA/NEW/output_1.jpg'  # Đường dẫn nơi ảnh sẽ được lưu
    cv2.imwrite(output_image_path, image)  # Lưu ảnh vào file
    print(f"Ảnh với nhãn dự đoán đã được lưu tại: {output_image_path}")

# Kiểm tra nếu chương trình đang chạy chính
if __name__ == "__main__":
    main()  # Gọi hàm main để thực thi chương trình
