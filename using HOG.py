import cv2
import numpy as np
from sklearn import svm

# Định nghĩa các tham số cho HOG descriptor
win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

# Chuẩn bị dữ liệu huấn luyện
# Đọc các hình ảnh chứa đối tượng cần theo dõi
pos_images = []
for i in range(1,9):
    img = cv2.imread(f'pic_00{i}.jpg')
    pos_images.append(img)

# Đọc các hình ảnh không có đối tượng (background)
neg_images = []
for i in range(1,9):
    img = cv2.imread(f'train ({i}).jpg')
    neg_images.append(img)

# Tạo vector đặc trưng (feature vector) cho các hình ảnh
features = []
labels = []
for img in pos_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = hog.compute(gray)
    features.append(feat.flatten())
    labels.append(1)

for img in neg_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = hog.compute(gray)
    features.append(feat.flatten())
    labels.append(0)
# Chuyển đổi features sang kiểu numpy array và thêm một chiều để tạo mảng 2 chiều
features = np.array(features)
features = np.reshape(features, (features.shape[0], -1))
features = np.vstack(features)

# Tạo labels
labels = np.concatenate((np.onesxs(len(pos_images)), np.zeros(len(neg_images))))

# Huấn luyện mô hình SVM
clf = svm.SVC(kernel='linear')
clf.fit(features, labels)

# Lấy kích thước khung hình của video
cap = cv2.VideoCapture('highway.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Đọc khung hình đầu tiên và chọn vùng quan tâm chứa đối tượng cần theo dõi
ret, frame = cap.read()
bbox = cv2.selectRoi(frame,False)

# Vòng lặp xử lý các khung hình
while True:
    # Đọc khung hình tiếp theo
    ret, frame = cap.read()

    # Nếu không còn khung hình nào, thoát khỏi vòng lặp
    if not ret:
        break

    # Lấy vùng quan tâm của khung hình hiện tại
    x, y, w, h = bbox
    roi = frame[y:y + h, x:x + w]

    # Trích xuất đặc trưng HOG cho vùng quan tâm
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    feat = hog.compute(gray).flatten()

    # Dự đoán xem vùng quan tâm có chứa đối tượng hay không bằng mô hình SVM đã huấn luyện
    prediction = clf.predict(feat.flatten().reshape(1, -1))

    # Nếu vùng quan tâm chứa đối tượng, hiển thị khung hình với đường bao quanh đối tượng
    if prediction == 1:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị khung hình kết quả
    cv2.imshow('Tracking', frame)

    # Nhấn phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()