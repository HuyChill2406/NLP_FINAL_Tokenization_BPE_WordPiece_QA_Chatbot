# NLP_FINAL_Tokenization_BPE_WordPiece_QA_Chatbot

## README.md

# NLP Final Assignments: Tokenization (BPE/WordPiece) & Q&A Chatbot

Repo này gồm 2 notebook cho bài tập cuối kỳ NLP:

* **Task 1:** Tìm hiểu **WordPiece** và **BPE (Byte-Pair Encoding)** + so sánh ảnh hưởng của BPE trong một bài toán phân loại văn bản.
* **Task 2:** Xây dựng **trợ lý ảo (chatbot) chăm sóc khách hàng** cho lĩnh vực du lịch (Q&A), từ thu thập dữ liệu → huấn luyện → test → đánh giá.

---

## 1. Nội dung chính

### ✅ Task 1 — So sánh BPE và không BPE trên bài toán SMS Spam

**Notebook:** `NLP_Final_Task1.ipynb`

**Mục tiêu**

* Trình bày khái niệm và ví dụ về **WordPiece** và **BPE**.
* Thực nghiệm so sánh 2 hướng:

  * **Không dùng BPE**: dùng văn bản gốc
  * **Dùng BPE**: token hóa bằng GPT-2 tokenizer (BPE), sau đó ghép token lại để vector hóa

**Dataset**

* **SMS Spam Collection** (Kaggle / UCI)
* File dùng trong notebook: `datasets_task1.csv` (đọc từ Google Drive)

**Pipeline thực nghiệm**

* Tiền xử lý: lowercase (đơn giản)
* Chia train/test: `train_test_split`
* Mô hình: **TF-IDF + Multinomial Naive Bayes**
* So sánh:

  * **Accuracy**
  * **Training time**
  * **Prediction time**
* Trực quan hóa bằng biểu đồ cột (accuracy/time)

**Tokenization được trình bày**

* **WordPiece**: minh họa bằng `BertTokenizer(bert-base-uncased)`
* **BPE**: dùng `GPT2Tokenizer(gpt2)` để token hóa theo BPE

> Lưu ý: Notebook có phần minh họa WordPiece và phần so sánh chính tập trung vào **BPE vs không BPE** trong mô hình TF-IDF + Naive Bayes.

---

### ✅ Task 2 — Chatbot chăm sóc khách hàng (Q&A Du lịch)

**Notebook:** `NLP_Final_Task2.ipynb`

**Mục tiêu**

* Tạo một “agent” chăm sóc khách hàng (trợ lý ảo) cho **du lịch**:

  * Thu thập & xây dựng dữ liệu huấn luyện
  * Chọn mô hình huấn luyện
  * Đánh giá chất lượng

**Dataset**

* Notebook mô tả dataset tự thu thập (lọc từ các trang báo du lịch Việt Nam).
* File dùng trong notebook: `datasets_task2.csv` (đọc từ Google Drive)

**Tiền xử lý dữ liệu**

* Drop các cột không dùng (dạng SQuAD-like), ví dụ:

  * `context`, `Title`, `is_impossible`, `id`, `answer_start`
* Giữ lại cặp chính:

  * `question` (câu hỏi người dùng)
  * `answers` (câu trả lời / nhãn)

**Mô hình**

* Cách tiếp cận: **phân loại câu hỏi → chọn câu trả lời (label)**
* Pipeline:

  * Tokenize bằng `keras.preprocessing.text.Tokenizer`
  * Padding sequence
  * Mô hình neural đơn giản:

    * `Embedding`
    * `Flatten`
    * `Dense(ReLU)`
    * `Dense(Softmax)` theo số lớp (số câu trả lời)
* Huấn luyện: `epochs=50`

**Suy luận (Demo chatbot)**

* Nhập câu hỏi từ terminal:

  * `You: ...`
  * Model dự đoán label và trả về câu trả lời tương ứng

**Đánh giá**

* **Accuracy** trên tập test
* **BLEU Score** (tính BLEU trung bình giữa câu trả lời dự đoán và câu trả lời thật)

> Ghi chú: BLEU trong bài toán chọn câu trả lời theo nhãn có thể không phản ánh đầy đủ chất lượng hội thoại, nhưng hữu ích để tham khảo mức “gần đúng” theo text.

---

## 2. Hướng dẫn chạy (Google Colab)

Hai notebook đang đọc dữ liệu từ đường dẫn Google Drive dạng:

* `/content/drive/MyDrive/datasets_task1.csv`
* `/content/drive/MyDrive/datasets_task2.csv`

Bạn cần:

1. Upload notebook lên Colab
2. Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Đảm bảo file `.csv` nằm đúng đường dẫn như trong notebook (hoặc sửa lại biến `url`)

---

## 3. Yêu cầu thư viện (gợi ý)

Cài các thư viện cần thiết (tùy notebook):

```bash
pip install pandas scikit-learn transformers nltk tensorflow
```

Nếu BLEU lỗi do thiếu NLTK data:

```python
import nltk
nltk.download('stopwords')
```

---

## 4. Kết quả mong đợi

* **Task 1:** in ra Accuracy + thời gian train/predict và vẽ biểu đồ so sánh.
* **Task 2:** in ra Accuracy + BLEU trung bình, và chạy được vòng lặp demo chatbot hỏi–đáp.

---

## 5. Hạn chế & hướng cải tiến

* **Task 1:** TF-IDF + NB là baseline cổ điển; có thể thử thêm Logistic Regression / SVM hoặc fine-tune model Transformer để thấy khác biệt rõ hơn.
* **Task 2:** hiện là mô hình “phân loại câu hỏi”; có thể nâng cấp lên:

  * BiLSTM/GRU (thay Flatten)
  * Transformer encoder
  * Retrieval (semantic search) + reranking
  * Generative chatbot (Seq2Seq / LLM fine-tune)
