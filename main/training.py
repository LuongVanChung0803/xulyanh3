import cv2
import os
import numpy as np

# Hàm trích xuất đặc trưng của ảnh
def extract_features(image_path):
    # Đọc ảnh và chuyển đổi sang ảnh xám
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc được ảnh: {image_path}")
        return None
    
    # Chuyển sang ảnh xám nếu ảnh chưa là ảnh grayscale
    if len(image.shape) == 3:  # Kiểm tra nếu ảnh là RGB hoặc BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thay đổi kích thước ảnh về 64x64 để giảm kích thước dữ liệu và chuẩn hóa đầu vào
    image = cv2.resize(image, (64, 64))
    # Chuyển ảnh thành vector đặc trưng một chiều
    features = image.flatten()
    return features

# Hàm lưu đặc trưng của ảnh cùng nhãn vào file
def save_features_to_file(features, label, file_path='train_data.txt'):
    # Mở file để ghi, nếu file tồn tại thì thêm dữ liệu mới vào cuối file
    with open(file_path, 'a') as f:
        # Ghi vector đặc trưng, ngăn cách các giá trị bằng dấu phẩy và thêm nhãn ở cuối dòng
        f.write(','.join(map(str, features)) + f',{label}\n')

# Hàm chính thực hiện dán nhãn tự động cho ảnh
def main():
    # Đường dẫn thư mục chứa các thư mục ảnh theo nhãn
    train_dir = 'D:/XLA/NEW/Animals'  # Thay thế bằng đường dẫn thực tế
    
    # Duyệt qua tất cả các thư mục con trong train_dir
    for folder_name in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder_name)
        
        # Kiểm tra nếu folder_path là thư mục con
        if os.path.isdir(folder_path):
            label = folder_name  # Tên thư mục con sẽ được sử dụng làm nhãn
            
            # Duyệt qua các file ảnh trong thư mục con
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                # Kiểm tra nếu image_path là một file ảnh hợp lệ (png, jpg, jpeg)
                if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Trích xuất đặc trưng của ảnh
                    features = extract_features(image_path)
                    # Kiểm tra nếu việc trích xuất đặc trưng thành công
                    if features is not None:
                        # Lưu đặc trưng và nhãn vào file
                        save_features_to_file(features, label)
                    else:
                        print(f"Không thể trích xuất đặc trưng cho ảnh: {image_path}")

# Gọi hàm chính để thực hiện việc gán nhãn và lưu đặc trưng
if __name__ == "__main__":
    main()
