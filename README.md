# Cheatsheet: 
### Phương thức get: 

#### Tổng quan
`@app.get()` là decorator dùng để định nghĩa endpoint HTTP GET. Cú pháp cơ bản:  
`@app.get(path, **kwargs)`  
- **path**: Bắt buộc, đường dẫn URL (ví dụ: `"/items/{item_id}"`).  
- **kwargs**: Các tham số tùy chọn để tùy chỉnh hành vi, tài liệu OpenAPI, và phản hồi.  

#### Tham số chính (Quan trọng nhất)
Dưới đây là bảng tóm tắt các tham số phổ biến nhất, với loại dữ liệu, giá trị mặc định, mô tả ngắn, và ví dụ.

| Tham số              | Loại dữ liệu                  | Mặc định       | Mô tả ngắn gọn                                                                 | Ví dụ ngắn gọn |
|----------------------|-------------------------------|----------------|--------------------------------------------------------------------------------|---------------|
| `path`              | `str` (bắt buộc)             | Không có      | Đường dẫn URL cho endpoint, hỗ trợ path params như `{id}`.                     | `"/users/{id}"` |
| `response_model`    | `Any` (Pydantic model)       | `None`        | Định dạng và xác thực phản hồi (dùng cho docs, serialization).                 | `response_model=UserOut` |
| `status_code`       | `Optional[int]`              | `None` (200)  | Mã trạng thái HTTP mặc định cho phản hồi.                                     | `status_code=201` |
| `tags`              | `Optional[List[str]]`        | `None`        | Nhãn nhóm endpoint trong docs OpenAPI (/docs).                                 | `tags=["users"]` |
| `summary`           | `Optional[str]`              | `None`        | Tóm tắt ngắn cho endpoint trong docs.                                          | `summary="Lấy user"` |
| `description`       | `Optional[str]`              | `None`        | Mô tả chi tiết (hỗ trợ Markdown) cho endpoint.                                 | `description="Lấy user theo ID"` |
| `dependencies`      | `Optional[Sequence[Depends]]`| `None`        | Các phụ thuộc (như auth) chạy trước endpoint.                                  | `dependencies=[Depends(get_user)]` |
| `responses`         | `Optional[Dict[int, Dict]]`  | `None`        | Định nghĩa các phản hồi có thể (bao gồm lỗi) cho docs.                         | `responses={404: {"description": "Not found"}}` |
| `deprecated`        | `Optional[bool]`             | `None`        | Đánh dấu endpoint là lỗi thời.                                                 | `deprecated=True` |
| `include_in_schema` | `bool`                       | `True`        | Bao gồm endpoint trong docs OpenAPI hay không.                                 | `include_in_schema=False` |

#### Tham số nâng cao (Response Model & Khác)
Những tham số này chủ yếu cho tùy chỉnh Pydantic và OpenAPI.

| Tham số                        | Loại dữ liệu | Mặc định | Mô tả ngắn gọn                                      | Ví dụ ngắn gọn |
|--------------------------------|--------------|----------|-----------------------------------------------------|---------------|
| `response_description`         | `str`       | `"Successful Response"` | Mô tả phản hồi mặc định trong docs.                | `"Dữ liệu user"` |
| `operation_id`                 | `Optional[str]` | `None`  | ID duy nhất cho endpoint (dùng cho client gen).     | `"get_user_by_id"` |
| `response_model_exclude_none`  | `bool`      | `False` | Loại bỏ trường `None` trong phản hồi.               | `response_model_exclude_none=True` |
| `response_class`               | `Type[Response]` | `JSONResponse` | Lớp phản hồi tùy chỉnh (ví dụ: HTML).             | `response_class=HTMLResponse` |
| `openapi_extra`                | `Optional[Dict]` | `None` | Metadata thêm cho OpenAPI schema.                   | `openapi_extra={"x-logo": {"url": "/logo.png"}}` |

#### Ví dụ ngắn (Cơ bản)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
class User(BaseModel): id: int; name: str

@app.get("/users/{user_id}", response_model=User, tags=["users"], summary="Lấy user")
def get_user(user_id: int):
    return {"id": user_id, "name": "Alice"}
```
- **Chạy**: `GET /users/1` → Phản hồi: `{"id": 1, "name": "Alice"}` (status 200).  
- **Docs**: Trong `/docs`, có tag "users", summary "Lấy user".

#### Ví dụ đầy đủ (Nâng cao với lỗi & tùy chỉnh)
```python
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
class UserOut(BaseModel): id: int; name: str; status: str

def get_current_user():  # Dependency giả lập auth
    return {"user": "authenticated"}

@app.get(
    "/users/{user_id}",
    response_model=UserOut,
    status_code=200,
    tags=["users"],
    summary="Lấy thông tin user",
    description="Lấy user theo ID, hỗ trợ query brief. **Lưu ý**: Cần auth.",
    dependencies=[Depends(get_current_user)],
    responses={
        200: {"description": "User thành công", "model": UserOut},
        404: {"description": "Không tìm thấy user"}
    },
    deprecated=False,
    operation_id="get_user_details",
    response_model_exclude_none=True  # Loại bỏ trường None
)
def get_user(
    user_id: int,
    brief: Optional[bool] = Query(False, description="Phiên bản ngắn gọn")
):
    if user_id == 999:
        raise HTTPException(status_code=404, detail="Không tìm thấy user")
    data = {"id": user_id, "name": "Bob", "status": "active" if not brief else None}
    return UserOut(**data)
```
- **Chạy ví dụ**:  
  - `GET /users/1` → `{"id": 1, "name": "Bob", "status": "active"}` (status 200).  
  - `GET /users/1?brief=true` → `{"id": 1, "name": "Bob"}` (loại bỏ `status=None`).  
  - `GET /users/999` → Lỗi 404: `{"detail": "Không tìm thấy user"}`.  
- **Docs**: Trong `/docs`, có đầy đủ summary, description, responses (200 & 404), dependency auth, và operation_id.

**Lưu ý**: Các tham số này áp dụng tương tự cho `@app.post()`, `@app.put()`, v.v. Xem docs FastAPI để chi tiết hơn.