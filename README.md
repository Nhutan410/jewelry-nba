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

## Gợi ý sản phẩm thật

App đọc trực tiếp `data/catalog_production_enriched.json`, dùng các trường chuẩn như `product_type`, `gender`, `audience`, `material`, `primary_stone`, `secondary_stone`, `price_tier`, `style_tags`, `occasion_tags`, `display_image_url`, rồi chấm điểm sản phẩm theo thang 100. Logic không dùng LLM để rank.

Rule chấm điểm:

- Ngân sách `25`: loại sản phẩm vượt trần ngân sách, ưu tiên giá nằm đúng khung.
- Loại SP `20`: khớp `product_type`; nhóm gần nhau như bracelet/bangle được điểm một phần.
- Dịp mua `15`: khớp `occasion_tags` hoặc tín hiệu LEP/purpose như cầu hôn, sinh nhật, quà tặng, kỷ niệm.
- Chất liệu/đá `15`: 8 điểm chất liệu/tuổi vàng, 7 điểm đá chính/phụ.
- Style `8`: map style khách sang tag như minimal, youthful, elegant, luxury, bold.
- Phân khúc `8`: RFM/monetary/budget quyết định `entry/mid/premium/luxury` có phù hợp không.
- Bán chạy `6`: ưu tiên số bán/rating nếu catalog có; nếu chưa có thì dùng proxy từ daily/classic/giftable/brand.
- Giới tính `3`: chấm theo người thụ hưởng thật sự, không mặc định theo người mua khi là mua tặng; gồm `gender` và `audience` adult/child.

Với khách walk-in, nếu khách mua cho người khác, TVV cần nhập thêm người thụ hưởng là người lớn/trẻ em và nam/nữ để hệ thống chọn đúng sản phẩm cho người sẽ đeo.

Test nhanh danh sách sản phẩm cho một khách:

```bash
python scripts/recommend_products_for_customer.py CUSTOMER_ID --top-n 5
```

Xuất JSON để kiểm tra hoặc dùng cho bước khác:

```bash
python scripts/recommend_products_for_customer.py CUSTOMER_ID --top-n 5 --json
```

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
