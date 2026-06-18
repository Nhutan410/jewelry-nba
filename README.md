# Jewelry NBA App

Ứng dụng Streamlit để xem dữ liệu, phân tích khách hàng, sinh kịch bản tư vấn và gợi ý ưu đãi phù hợp.

## Tính năng chính

- Đăng nhập đơn giản bằng username/password từ `.env`.
- Xem dữ liệu gốc và kết quả phân tích.
- Sinh sales script bằng OpenAI nếu có API key, hoặc fallback template nếu không có.
- Gợi ý Next Best Offer theo từng khách hàng.
- Hỗ trợ chạy bằng Docker Compose, Docker CLI hoặc Python local.

## Yêu cầu

- Docker Desktop nếu chạy bằng Docker.
- Python 3.11+ nếu chạy local.
- File `.env` ở thư mục gốc dự án.

## Cấu hình `.env`

Tạo file `.env` từ file mẫu:

```bash
cp .env.example .env
```

Điền giá trị thật vào `.env`:

```env
AUTH_USERNAME=your_username
AUTH_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

`OPENAI_API_KEY` có thể để trống nếu chỉ muốn chạy fallback template. Không commit file `.env`.

## Chạy bằng Docker Compose

```bash
docker compose up --build
```

Mở app tại:

```text
http://localhost:8501
```

Nếu port `8501` bị chiếm, sửa trong `docker-compose.yml`:

```yaml
ports:
  - "8502:8501"
```

Sau đó mở `http://localhost:8502`.

## Chạy bằng Docker CLI

Build image:

```bash
docker build -t jewelry-nba:latest .
```

Chạy container:

```bash
docker run --rm --env-file .env --name jewelry-nba -p 8501:8501 jewelry-nba:latest
```

Nếu port `8501` bị chiếm:

```bash
docker run --rm --env-file .env --name jewelry-nba -p 8502:8501 jewelry-nba:latest
```

## Chạy local bằng Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Mở app tại:

```text
http://localhost:8501
```
