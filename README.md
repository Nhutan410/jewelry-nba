# 💎 PNJ · In-Store NBA Dashboard

> **Next Best Action engine** cho nhân viên bán hàng (TVV) tại cửa hàng PNJ.  
> Phân tích hành vi khách hàng, phân loại tệp và tự động sinh **Sales Script 5 bước** bằng AI (GPT-4o hoặc Fallback Template).

---

## 🗂️ Cấu trúc thư mục

```
jewelry_nba/
├── app_instore.py          # Entry point — Streamlit dashboard chính
├── requirements.txt        # Danh sách thư viện Python
├── data/
│   └── customer_data_poc_enhanced.xlsx   # File dữ liệu Excel đầu vào
├── outputs/
│   ├── instore_scripts.json              # JSON cache kết quả phân tích (tự sinh)
│   ├── models/                           # Model LEP đã train (tự sinh)
│   ├── .instore_cache/                   # Cache nội bộ (tự sinh)
│   └── .llm_cache/                       # Cache LLM (tự sinh)
└── src/
    ├── lep_pipeline.py                   # Model Machine Learning (LEP)
    ├── instore_script_engine.py          # Engine phân loại + sinh script
    ├── nba_engine.py / nba_engine_llm.py # NBA logic
    ├── pipeline.py / pipeline_instore.py / pipeline_llm.py
    ├── llm_message_generator.py
    └── feedback_loop.py
```

---

## 🚀 Hướng dẫn cài đặt từ đầu trên **Windows**

> Không cần VSCode. Chỉ cần **Command Prompt (cmd)** hoặc **PowerShell**.

---

### Bước 1 — Cài Python 3.10+

1. Vào **https://www.python.org/downloads/** → tải Python **3.10** hoặc **3.11** (khuyên dùng).
2. Chạy file cài đặt → **bắt buộc tích vào ô `Add Python to PATH`** trước khi nhấn Install.
3. Mở **Command Prompt** (tìm "cmd" trong Start menu) và kiểm tra:

```cmd
python --version
```

Kết quả mẫu: `Python 3.11.x` ✅

---

### Bước 2 — Tải source code từ GitHub

**Cách A — Dùng Git (khuyên dùng):**

```cmd
:: Cài Git nếu chưa có: https://git-scm.com/download/win
git clone https://github.com/Nhutan410/jewelry-nba.git
cd jewelry-nba
```

**Cách B — Tải ZIP:**

1. Vào **https://github.com/Nhutan410/jewelry-nba**
2. Nhấn nút **Code → Download ZIP**
3. Giải nén và mở cmd trong thư mục đó:

```cmd
cd C:\Users\<TenBan>\Downloads\jewelry-nba-main
```

---

### Bước 3 — Tạo môi trường ảo (virtual environment)

```cmd
python -m venv venv
```

Kích hoạt môi trường ảo:

```cmd
venv\Scripts\activate
```

> Bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh — đó là môi trường đã được kích hoạt ✅

---

### Bước 4 — Cài thư viện

```cmd
pip install -r requirements.txt
```

> Quá trình cài đặt mất khoảng **1–3 phút** tuỳ tốc độ internet.

---

### Bước 5 — Chuẩn bị dữ liệu

Đặt file Excel dữ liệu khách hàng vào thư mục `data/`:

```
data\customer_data_poc_enhanced.xlsx
```

> **Lưu ý:** File Excel phải có sheet tên `profiles_enhanced` với cột `c` (customer_id).

---

### Bước 6 — Chạy ứng dụng

```cmd
streamlit run app_instore.py
```

Streamlit sẽ tự động mở trình duyệt tại **http://localhost:8501**

Nếu trình duyệt không tự mở, hãy copy địa chỉ đó dán vào Chrome / Edge.

---

### ✅ Tóm tắt các lệnh (copy chạy một lần)

```cmd
git clone https://github.com/Nhutan410/jewelry-nba.git
cd jewelry-nba
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_instore.py
```

---

## 🔑 Cấu hình OpenAI API Key (tùy chọn)

Ứng dụng hỗ trợ **2 chế độ**:

| Chế độ | Mô tả |
|---|---|
| **GPT-4o mode** | Sinh script cá nhân hóa bằng AI — cần API Key |
| **Fallback Template** | Sinh script từ template cố định — **không cần API Key** |

**Nhập API Key trực tiếp trên giao diện** (sidebar trái) — không cần file cấu hình.

Hoặc đặt biến môi trường trước khi chạy:

```cmd
set OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
streamlit run app_instore.py
```

---

## 🖥️ Truy cập từ máy khác trong cùng mạng LAN

Sau khi chạy `streamlit run`, terminal sẽ hiển thị **Network URL**, ví dụ:

```
Network URL: http://192.168.1.126:8501
```

Mở URL này trên **bất kỳ máy nào trong cùng WiFi/mạng LAN** để truy cập dashboard — không cần cài thêm gì.

---

## 🔄 Lần chạy tiếp theo

Mỗi lần mở lại máy, chỉ cần:

```cmd
cd jewelry-nba
venv\Scripts\activate
streamlit run app_instore.py
```

---

## ⚙️ Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| **OS** | Windows 10/11 (64-bit) |
| **Python** | 3.10 hoặc 3.11 |
| **RAM** | 4 GB (khuyên dùng 8 GB+) |
| **Kết nối Internet** | Cần khi cài đặt lần đầu và khi dùng GPT-4o |

---

## 🧩 Các tab trong Dashboard

| Tab | Chức năng |
|---|---|
| **📊 Dữ liệu thô** | Xem toàn bộ dữ liệu Excel gốc |
| **🤖 Phân tích AI** | Danh sách khách đã phân tích + Sales Script 5 bước |
| **🔍 Chi tiết khách** | Tìm kiếm và xem chi tiết từng khách hàng |
| **📖 Hướng dẫn** | Giải thích chiến lược phân loại tệp khách |

---

## 🐛 Xử lý lỗi thường gặp

**Lỗi `'streamlit' is not recognized`:**
```cmd
pip install streamlit
```

**Lỗi `ModuleNotFoundError`:**
```cmd
pip install -r requirements.txt
```

**Lỗi không đọc được file Excel:**
- Kiểm tra file có đúng tên `customer_data_poc_enhanced.xlsx` trong thư mục `data/` không.
- Đảm bảo file Excel không đang mở (đóng Excel trước).

**Lỗi `(venv) không xuất hiện` khi activate:**
- Chạy PowerShell với quyền Admin và gõ:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Sau đó thử lại `venv\Scripts\activate`.

---

## 📄 License

Internal use · PNJ Digital Team · 2024–2026
