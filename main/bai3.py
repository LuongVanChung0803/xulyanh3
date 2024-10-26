import cv2
import numpy as np
from collections import Counter

def load_training_data(file_path='train_data.txt'):
    features = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            *feature_values, label = line.strip().split(',')
            features.append(list(map(float, feature_values)))
            labels.append(label)
    
    return np.array(features), np.array(labels)

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    features = image.flatten()
    return features

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def predict_label(image_path, training_features, training_labels, k=3):
    features = extract_features(image_path)
    distances = []
    
    for i, train_feature in enumerate(training_features):
        distance = euclidean_distance(features, train_feature)
        distances.append((distance, training_labels[i]))
    
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_label

def main():
    # Tải dữ liệu huấn luyện
    features, labels = load_training_data('train_data.txt')
    
    # Đường dẫn ảnh đầu vào
    input_image_path = '/XLA/NEW/input1.jpg'
    predicted_label = predict_label(input_image_path, features, labels, k=15)
    
    # Đọc ảnh màu để hiển thị
    image = cv2.imread(input_image_path)
    
    # Thêm nhãn dự đoán lên ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Du doan: {predicted_label}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Hiển thị ảnh với nhãn dự đoán
    cv2.imshow('Du doan anh', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Lưu ảnh với nhãn dự đoán vào file mới
    output_image_path = '/XLA/NEW/output_1.jpg'  # Đường dẫn nơi ảnh sẽ được lưu
    cv2.imwrite(output_image_path, image)
    print(f"Ảnh với nhãn dự đoán đã được lưu tại: {output_image_path}")

if __name__ == "__main__":
    main()
