# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- \**MSSV:*2A202604109\*
- \**Họ và tên:*Nguyễn Trọng Khiêm\*
- **SEED** (in ở ô đầu notebook): 2164

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | ☐   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐   |
| [`hand_calc1.jpg`](hand_calc1.jpg) — ảnh tính tay có các bước        | ☐   |
| [`hand_calc2.jpg`](hand_calc2.jpg) — ảnh tính tay có các bước        | ☐   |

## 2. Kết quả chính

Bảng so sánh kết quả các mô hình:

| Mô hình           | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
| ----------------- | ------: | --------: | -----------: | ---------: |
| BoW + LogReg      |   0.886 |     0.857 |        0.080 |          0 |
| BoW + MLP         |   0.936 |     0.919 |        0.224 |          0 |
| MiniLM + kNN      |   0.920 |     0.919 |      135.950 |          0 |
| MiniLM + LogReg   |   0.960 |     0.923 |       37.911 |          0 |
| MiniLM + MLP head |   0.969 |     0.895 |       35.477 |          0 |

**Nhận xét:**

- MiniLM + MLP head đạt F1 test cao nhất (0.969).
- MiniLM + LogReg đạt F1 cao nhất trên SUB 40 (0.923).
- BoW + LogReg có độ trễ thấp nhất.
- MiniLM + kNN vượt ngưỡng SLA 50 ms.
- Chi phí API bằng 0 vì các mô hình trên chạy local/Colab, chưa tính chi phí hạ tầng.
- LLM chưa có kết quả thực tế do chưa cấu hình API key.

## 3. Dự đoán và thực tế

### 3.1. Learning rate (ml_lr)

- **Dự đoán:** `lr=50` sẽ khiến loss dao động mạnh và không hội tụ.
- **Thực tế:** Loss tăng mạnh ban đầu nhưng sau đó giảm xuống khoảng 0.003.
- **Giải thích:** Learning rate lớn không phải lúc nào cũng khiến mô hình thất bại, mà còn phụ thuộc dữ liệu và quá trình tối ưu.

### 3.2. Bỏ ReLU (dl_relu)

- **Dự đoán:** F1 sẽ giảm về gần Logistic Regression (0.886).
- **Thực tế:** MLP không có ReLU vẫn đạt F1 = 0.930, gần với MLP có ReLU (0.936).
- **Giải thích:** Với dữ liệu nhỏ và đặc trưng BoW, mô hình tuyến tính đã có thể học khá tốt nên lợi ích từ tính phi tuyến không quá lớn.

### 3.3. Stress test (hard_eval)

- **Dự đoán:** Embedding + kNN đạt accuracy khoảng 62%.
- **Thực tế:** Mô hình đạt accuracy 72%.
- **Giải thích:** Embedding có khả năng biểu diễn ngữ nghĩa nên xử lý tốt hơn dự đoán đối với một số tin nhắn khó.

## 4. Phân tích 50 case khó

**BoW + LogReg tụt nhiều nhất**, F1 giảm từ 0.886 xuống 0.286, tức giảm 0.600.

Nhóm khó nhất là `natural_spam`. Cả hai mô hình BoW đều dự đoán sai toàn bộ 10 tin thuộc nhóm này.

**Hai trường hợp cụ thể:**

1. **Case #12:** Tin nhắn mở đầu lịch sự, nói được bạn bè giới thiệu rồi dụ người nhận truy cập trang hẹn hò. Mô hình dễ nhầm thành ham vì thiếu các từ khóa spam quen thuộc.

2. **Case #14:** Tin nhắn kể chuyện làm việc tại nhà, kiếm £300/ngày và không cần kinh nghiệm. Cách viết giống chia sẻ cá nhân nên mô hình không nhận ra dấu hiệu lừa đảo.

**Kết luận:** F1 cao trên tập test thông thường chưa đảm bảo mô hình xử lý tốt spam ngụy trang. Cần đánh giá thêm bằng những bộ dữ liệu khó để phát hiện điểm yếu.

## 5. Thử thách cá nhân

### Đề số 5 — Embedding

Viết 4 tin nhắn gồm 2 spam né từ khóa và 2 ham chứa `free` hoặc `win`. So sánh BoW + LogReg với Embedding + LogReg.

**Cách thực hiện:**

- Chuẩn bị 4 tin nhắn với nhãn thật.
- Dùng BoW + LogReg để dự đoán.
- Dùng MiniLM + LogReg để dự đoán.
- So sánh kết quả và giải thích các trường hợp sai.

**Bốn ví dụ tham khảo:**

| Tin nhắn                                                                        | Nhãn thật |
| ------------------------------------------------------------------------------- | --------- |
| Hi, I have a private opportunity for you. Send your bank details to join today. | Spam      |
| Your delivery cannot be completed. Confirm your identity through this link.     | Spam      |
| Are you free to study together this evening?                                    | Ham       |
| We managed to win the football match yesterday!                                 | Ham       |

**Kết quả:** Chưa có kết quả chạy thực tế trên 4 tin nhắn mới, cần bổ sung sau khi thực hiện thí nghiệm.

**Giải thích:** BoW dễ bị nhầm khi gặp từ khóa spam trong ngữ cảnh vô hại hoặc spam cố tình né từ khóa. Embedding có thể hiểu ngữ cảnh tốt hơn, nhưng vẫn cần kiểm chứng bằng kết quả thực nghiệm.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:**

- Xử lý 10 triệu SMS/ngày (~116 tin/giây).
- p95 < 50 ms/tin.
- Đảm bảo chất lượng phân loại và tối ưu chi phí.

**Phương án lựa chọn: Hybrid Router**

Tôi đề xuất kết hợp **BoW + MLP** và **MiniLM + LogReg**.

- BoW + MLP xử lý ban đầu vì có F1 = 0.936, p95 = 0.224 ms và tốc độ khoảng 39.482 tin/giây.
- Các tin có xác suất phân loại không chắc chắn được chuyển sang MiniLM + LogReg, có F1 = 0.960 và p95 = 37.911 ms.
- Các tin có độ tin cậy cao được xử lý ngay để tiết kiệm tài nguyên.

**Đánh đổi:**

Hệ thống phức tạp hơn và cần hiệu chỉnh ngưỡng phân loại. Ngoài ra, cần benchmark lại toàn bộ pipeline khi triển khai thực tế để đảm bảo throughput và SLA.

Nếu sử dụng LLM để xử lý toàn bộ 10 triệu SMS/ngày, giả sử mỗi tin có 100 input tokens và 1 output token, với giá $0.10/$0.50 cho mỗi triệu token, chi phí khoảng **$3,150/tháng** (30 ngày), chưa tính hạ tầng và chi phí phát sinh.

## 7. AI log và điều đã học

### 7.1. AI log

| Phần       | Hỏi AI gì?                                  | AI hỗ trợ gì?                                     | Sửa / kiểm chứng                                   |
| ---------- | ------------------------------------------- | ------------------------------------------------- | -------------------------------------------------- |
| Phần 1     | BoW, Sigmoid, Gradient Descent, metrics     | Hướng dẫn code NumPy và công thức                 | Test PASS, sửa lỗi xử lý list trong metrics        |
| Phần 2     | Xây MLP và vòng lặp huấn luyện              | Hướng dẫn PyTorch, backpropagation                | Test PASS, kiểm tra train/val loss                 |
| Phần 3     | Cosine Similarity, Top-K, kNN               | Code và giải thích các thuật toán                 | Test PASS, kiểm tra chuẩn hóa vector               |
| Phần 4     | Prompt, parse label, gọi API                | Hướng dẫn code gọi LLM và đo chi phí              | API giả lập PASS, chưa chạy API thật               |
| Thí nghiệm | Dự đoán F1, learning rate, overfitting      | Gợi ý giả thuyết                                  | Đối chiếu với kết quả chạy thực tế                 |
| Phần 5     | Tính tay, stress test, thử thách và báo cáo | Hướng dẫn phép tính, gợi ý code, tổng hợp báo cáo | Đối chiếu số liệu Colab, thử thách chưa kiểm chứng |

### 7.2. Điều đã học

Trước đây tôi nghĩ mô hình càng phức tạp thì độ chính xác càng cao, nhưng thực tế còn phụ thuộc vào dữ liệu và cách biểu diễn đặc trưng.

Tôi hiểu rằng F1 cao không đồng nghĩa mô hình luôn hoạt động tốt, đặc biệt với những tin spam có cách viết khác thường.

Ngoài độ chính xác, khi triển khai sản phẩm thực tế còn phải cân nhắc tốc độ xử lý, khả năng mở rộng và chi phí vận hành.
