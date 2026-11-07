# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- \*MSSV:\*2A202604109\*
- \**Họ và tên:*Nguyễn Trọng Khiêm\*
- **SEED** (in ở ô đầu notebook): 2164

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | Có  |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có  |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | Có  |

## 2. Kết quả chính

Bảng so sánh kết quả các mô hình:

| Mô hình                       | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
| ----------------------------- | ------: | --------: | -----------: | ---------: |
| BoW + LogReg                  |   0.886 |     0.857 |        0.087 |          0 |
| BoW + MLP                     |   0.936 |     0.919 |        0.352 |          0 |
| MiniLM + kNN                  |   0.920 |     0.919 |       92.964 |          0 |
| MiniLM + LogReg               |   0.960 |     0.923 |       35.247 |          0 |
| MiniLM + MLP head             |   0.969 |     0.895 |       35.257 |          0 |
| Groq GPT-OSS 20B (zero-shot)  |       — |     0.878 |     2413.437 |   0.006865 |
| Groq GPT-OSS 120B (zero-shot) |       — |     0.864 |     2436.987 |   0.013730 |
| Groq GPT-OSS 20B (few-shot)   |       — |     0.923 |     2436.125 |   0.014465 |

**Nhận xét:**

- MiniLM + MLP head đạt F1 test cao nhất (0.969).
- MiniLM + LogReg và Groq GPT-OSS 20B few-shot cùng đạt F1 SUB 40 cao nhất (0.923).
- BoW + LogReg có độ trễ thấp nhất (p95 = 0.087 ms/tin).
- MiniLM + kNN và cả ba phương án LLM vượt ngưỡng SLA 50 ms/tin.
- LLM được đánh giá trên SUB 40, chưa có F1 của toàn bộ tập test; các mô hình còn lại được đánh giá trên tập test 775 tin.
- Chi phí API của các mô hình ML/DL/Embedding bằng 0, chưa tính hạ tầng. Chi phí Groq là ước tính theo token trong notebook, không phải số tiền thanh toán thực tế.

## 3. Dự đoán và thực tế

### 3.1. Learning rate (ml_lr)

- **Dự đoán:** `lr=50` sẽ khiến loss dao động mạnh hoặc không hội tụ.
- **Thực tế:** Loss tăng mạnh ban đầu nhưng sau đó giảm xuống khoảng 0.003.
- **Giải thích:** Learning rate lớn có thể khiến quá trình tối ưu mất ổn định, nhưng không phải lúc nào cũng làm mô hình không hội tụ; kết quả còn phụ thuộc vào dữ liệu và hàm loss.

### 3.2. Bỏ ReLU (dl_relu)

- **Dự đoán:** Khi bỏ ReLU, F1 sẽ giảm về gần Logistic Regression (0.886).
- **Thực tế:** MLP không có ReLU đạt F1 = 0.930, gần với MLP có ReLU (0.936).
- **Giải thích:** Hai lớp Linear liên tiếp không có activation vẫn tương đương một phép biến đổi tuyến tính. Trên đặc trưng BoW, mô hình tuyến tính đã học khá tốt nên chênh lệch F1 không lớn.

### 3.3. Stress test (hard_eval)

- **Dự đoán:** Embedding + kNN đạt accuracy khoảng 62%, LLM nhỏ khoảng 70% trên 50 case khó.
- **Thực tế:** Embedding + kNN đạt 72%, Groq GPT-OSS 20B zero-shot đạt 92%.
- **Giải thích:** Embedding biểu diễn ngữ nghĩa tốt hơn BoW trong nhiều trường hợp; LLM cũng nhận diện tốt nội dung spam/ham viết tự nhiên trong bộ 50 case này. Tuy nhiên, kết quả trên bộ dữ liệu nhỏ chưa đủ để kết luận hiệu quả ngoài thực tế.

## 4. Phân tích 50 case khó

**BoW + LogReg tụt nhiều nhất**, F1 giảm từ 0.886 trên tập test thường xuống 0.286 trên tập 50 case khó, tức giảm 0.600.

Nhóm khó nhất với các mô hình BoW là `natural_spam`. Cả BoW + LogReg và BoW + MLP đều dự đoán sai toàn bộ 10 tin thuộc nhóm này. Trong khi đó, Groq GPT-OSS 20B zero-shot đạt F1 = 0.933, accuracy = 92%; khi dùng prompt phòng thủ, F1 = 0.918 và accuracy = 90%.

**Hai trường hợp cụ thể:**

1. **Case #12:** Tin nhắn mở đầu lịch sự, nói được bạn bè giới thiệu rồi dụ người nhận truy cập trang hẹn hò. BoW dễ nhầm thành ham vì không có những từ khóa spam quen thuộc.
2. **Case #14:** Tin nhắn kể chuyện làm việc tại nhà, kiếm £300/ngày và không cần kinh nghiệm. Cách viết giống chia sẻ cá nhân khiến BoW khó nhận ra dấu hiệu spam tuyển dụng/lừa đảo.

**Kết luận:** F1 cao trên tập test thông thường chưa đảm bảo mô hình hoạt động tốt với spam ngụy trang. LLM xử lý bộ case khó này tốt hơn các mô hình truyền thống, nhưng prompt phòng thủ không cải thiện điểm tổng thể trong lần đo: accuracy giảm từ 92% xuống 90%.

## 5. Thử thách cá nhân

### Đề số 5 — Embedding

Viết 4 tin nhắn gồm 2 spam né từ khóa và 2 ham chứa `free` hoặc `win`. So sánh BoW + LogReg với Embedding + LogReg.

**Cách thực hiện:**

- Chuẩn bị 4 tin nhắn với nhãn thật.
- Dùng BoW + LogReg để dự đoán.
- Dùng MiniLM + LogReg để dự đoán.
- So sánh kết quả và giải thích các trường hợp sai.

**Bốn tin nhắn đã kiểm tra:**

| Tin nhắn                                                                        | Nhãn thật | BoW + LogReg | MiniLM + LogReg |
| ------------------------------------------------------------------------------- | --------- | ------------ | --------------- |
| Hi, I have a private opportunity for you. Send your bank details to join today. | Spam      | Ham (sai)    | Spam (đúng)     |
| Your delivery cannot be completed. Confirm your identity through this link.     | Spam      | Ham (sai)    | Ham (sai)       |
| Are you free to study together this evening?                                    | Ham       | Ham (đúng)   | Ham (đúng)      |
| We managed to win the football match yesterday!                                 | Ham       | Ham (đúng)   | Ham (đúng)      |

**Kết quả:** BoW + LogReg đúng 2/4 tin (accuracy = 0.50, F1 = 0.000). MiniLM + LogReg đúng 3/4 tin (accuracy = 0.75, F1 = 0.667).

**Giải thích:** BoW bỏ sót cả hai tin spam né từ khóa. Embedding nhận diện đúng tin yêu cầu thông tin ngân hàng nhưng vẫn bỏ sót tin giả giao hàng. Cả hai mô hình nhận diện đúng hai tin ham có `free` hoặc `win` trong ngữ cảnh bình thường.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:**

- Xử lý 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây).
- p95 < 50 ms/tin.
- Đảm bảo chất lượng phân loại và tối ưu chi phí.

**Phương án lựa chọn: Hybrid Router**

Tôi đề xuất kết hợp **BoW + MLP** và **MiniLM + LogReg**.

- BoW + MLP xử lý ban đầu vì có F1 test = 0.936, p95 = 0.352 ms và throughput đo được khoảng 29.939 tin/giây khi xử lý cả lô.
- Các tin có xác suất phân loại không chắc chắn được chuyển sang MiniLM + LogReg, có F1 test = 0.960 và p95 = 35.247 ms.
- Các tin có độ tin cậy cao được xử lý ngay để tiết kiệm tài nguyên; LLM không nằm trên đường phản hồi đồng bộ vì p95 của các thí nghiệm Groq đều trên 2.400 ms.

**Đánh đổi:**

Hệ thống phức tạp hơn, cần hiệu chỉnh ngưỡng và kiểm soát false positive để tránh chặn nhầm SMS hợp lệ. MiniLM + LogReg đạt khoảng 45.319 tin/giây trong phép đo cả lô nên có thể cần nhiều worker khi số tin chuyển sang tầng này tăng. Phải benchmark lại toàn bộ pipeline ở tải thực tế để xác nhận SLA.

Nếu dùng Groq GPT-OSS 20B zero-shot cho toàn bộ 10 triệu SMS/ngày, theo mức ước tính 0.006865 USD/1.000 tin trong notebook, chi phí khoảng **2,059.50 USD/30 ngày**, chưa tính chi phí phát sinh. Độ trễ p95 = 2,413.437 ms/tin cũng không đáp ứng yêu cầu 50 ms.

## 7. AI log và điều đã học

### 7.1. AI log

| Phần       | Hỏi AI gì?                                          | AI hỗ trợ gì?                                                     | Sửa / kiểm chứng                                                 |
| ---------- | --------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| Phần 1     | BoW, Sigmoid, Gradient Descent, metrics             | Hướng dẫn code NumPy và giải thích công thức                      | Các test PASS; chỉnh cách xử lý dữ liệu trong metrics            |
| Phần 2     | Xây MLP và vòng lặp huấn luyện                      | Hướng dẫn code PyTorch, backpropagation                           | Test PASS; kiểm tra train/validation loss và ReLU                |
| Phần 3     | Cosine Similarity, Top-K, kNN                       | Hướng dẫn code và cách tính độ tương đồng                         | Test PASS; kiểm tra chuẩn hóa và thứ tự kết quả                  |
| Phần 4     | Prompt, parse label, gọi Groq API                   | Hướng dẫn gọi API, xử lý phản hồi, đo latency và ước tính chi phí | Mock test PASS; chạy API thật, đối chiếu zero-shot và few-shot   |
| Thí nghiệm | Dự đoán F1, learning rate, overfitting, stress test | Gợi ý giả thuyết và giải thích kết quả                            | Đối chiếu dự đoán với các số liệu đã chạy trong notebook         |
| Phần 5     | Tính tay, thử thách cá nhân và báo cáo              | Hướng dẫn công thức, gợi ý SMS và tổng hợp báo cáo                | Các test tính tay PASS; chạy 4 SMS và kiểm chứng kết quả dự đoán |

### 7.2. Điều đã học

Trước đây tôi nghĩ mô hình càng phức tạp thì độ chính xác càng cao, nhưng thực tế còn phụ thuộc vào dữ liệu và cách biểu diễn đặc trưng.

Tôi hiểu rằng F1 cao không đồng nghĩa mô hình luôn hoạt động tốt, đặc biệt với spam có cách viết tự nhiên; đồng thời LLM lớn hơn không nhất thiết đạt F1 cao hơn model nhỏ trên SUB 40.

Ngoài độ chính xác, khi triển khai sản phẩm thực tế còn phải cân nhắc tốc độ xử lý, khả năng mở rộng và chi phí vận hành.
