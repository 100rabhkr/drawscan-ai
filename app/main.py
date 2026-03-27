"""
DrawScan AI — FastAPI application.
2D Engineering Drawing Data Extraction for United Rubber Industries.
"""

import asyncio
import json
import os
import shutil
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from app.models import (
    EXTRACTION_TIMEOUT_SECONDS,
    MAX_UPLOAD_SIZE_BYTES,
    SLA_LIMITS,
    SLATier,
    authenticate,
    create_extraction,
    create_session,
    get_all_extractions,
    get_all_users,
    get_db,
    get_extraction,
    get_stats,
    get_user_by_session,
    get_user_extractions,
    get_user_monthly_usage,
    hash_password,
    init_db,
    update_extraction,
)

app = FastAPI(title="DrawScan AI", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Template context helpers
# ---------------------------------------------------------------------------

def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get("session_token")
    return get_user_by_session(token)


def base_context(request: Request, user: dict, **extra) -> dict:
    tier = SLATier(user["sla_tier"])
    sla = SLA_LIMITS[tier]
    usage = get_user_monthly_usage(user["id"])
    return {
        "user": user,
        "sla": sla,
        "sla_tier": tier,
        "usage": usage,
        "usage_pct": min(round(usage / sla["max_drawings_per_month"] * 100), 100),
        "now": datetime.utcnow(),
        **extra,
    }


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

def require_auth(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


def require_admin(request: Request) -> dict:
    user = require_auth(request)
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    init_db()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "drawscan-ai", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    user = authenticate(email, password)
    if not user:
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid email or password"})
    token = create_session(user["id"])
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("session_token", token, httponly=True, samesite="lax", max_age=7 * 86400)
    return response


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session_token")
    if token:
        db = get_db()
        db.execute("DELETE FROM sessions WHERE token = ?", (token,))
        db.commit()
        db.close()
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session_token")
    return response


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    stats = get_stats(user["id"])
    extractions = get_user_extractions(user["id"], limit=10)
    ctx = base_context(request, user, stats=stats, extractions=extractions, page="dashboard")
    return templates.TemplateResponse(request, "dashboard.html", ctx)


# ---------------------------------------------------------------------------
# Upload & Extract
# ---------------------------------------------------------------------------

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ctx = base_context(request, user, page="upload")
    return templates.TemplateResponse(request, "upload.html", ctx)


@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Check SLA limit
    tier = SLATier(user["sla_tier"])
    sla = SLA_LIMITS[tier]
    usage = get_user_monthly_usage(user["id"])
    if usage >= sla["max_drawings_per_month"]:
        return JSONResponse({"error": f"Monthly limit reached ({sla['max_drawings_per_month']} drawings). Upgrade your plan."}, status_code=429)

    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported"}, status_code=400)

    # Save file with size check
    timestamp = int(time.time() * 1000)
    safe_name = f"{timestamp}_{file.filename.replace(' ', '_')}"
    upload_path = UPLOAD_DIR / safe_name
    size = 0
    with open(upload_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE_BYTES:
                f.close()
                upload_path.unlink(missing_ok=True)
                return JSONResponse({"error": f"File too large. Max {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."}, status_code=413)
            f.write(chunk)

    # Create extraction record
    extraction_id = create_extraction(user["id"], file.filename, str(upload_path))

    return JSONResponse({"id": extraction_id, "filename": file.filename, "status": "processing"})


@app.post("/api/extract/{extraction_id}")
async def run_extraction(extraction_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    ext = get_extraction(extraction_id)
    if not ext or ext["user_id"] != user["id"]:
        return JSONResponse({"error": "Not found"}, status_code=404)

    try:
        start = time.time()
        update_extraction(extraction_id, status="ocr")

        # Run blocking OCR in thread pool with timeout to keep event loop responsive
        from app.extractor import extract_drawing
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, extract_drawing, ext["upload_path"]),
                timeout=EXTRACTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            update_extraction(extraction_id, status="failed")
            return JSONResponse({"error": f"Extraction timed out after {EXTRACTION_TIMEOUT_SECONDS}s"}, status_code=504)

        elapsed_ms = int((time.time() - start) * 1000)

        # Save preview image (returned by extract_drawing to avoid double rasterization)
        preview_png = result.pop("_preview_png", None)
        if preview_png:
            img_path = OUTPUT_DIR / f"{extraction_id}_preview.png"
            with open(img_path, "wb") as f:
                f.write(preview_png)
            update_extraction(extraction_id, image_path=str(img_path))

        # Accuracy: average confidence across extracted dimensions
        dims = result.get("dimensions", [])
        if dims:
            accuracy = round(sum(d.get("confidence", 90) for d in dims) / len(dims), 1)
        else:
            accuracy = 0.0

        update_extraction(
            extraction_id,
            status="review",
            extraction_json=json.dumps(result),
            accuracy_score=accuracy,
            processing_time_ms=elapsed_ms,
        )

        return JSONResponse({
            "id": extraction_id,
            "status": "review",
            "extraction": result,
            "processing_time_ms": elapsed_ms,
            "accuracy_score": accuracy,
        })

    except Exception as e:
        traceback.print_exc()
        update_extraction(extraction_id, status="failed")
        return JSONResponse({"error": str(e), "status": "failed"}, status_code=500)


@app.get("/api/extraction/{extraction_id}")
async def get_extraction_data(extraction_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    ext = get_extraction(extraction_id)
    if not ext:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if ext["user_id"] != user["id"] and not user["is_admin"]:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    result = dict(ext)
    if result.get("extraction_json"):
        result["extraction"] = json.loads(result["extraction_json"])
    return JSONResponse(result)


@app.get("/api/extraction/{extraction_id}/image")
async def get_extraction_image(extraction_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    ext = get_extraction(extraction_id)
    if not ext or not ext.get("image_path"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    if ext["user_id"] != user["id"] and not user["is_admin"]:
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    return FileResponse(ext["image_path"], media_type="image/png")


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

@app.get("/review/{extraction_id}", response_class=HTMLResponse)
async def review_page(extraction_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ext = get_extraction(extraction_id)
    if not ext:
        return RedirectResponse("/dashboard", status_code=303)
    if ext["user_id"] != user["id"] and not user["is_admin"]:
        return RedirectResponse("/dashboard", status_code=303)

    extraction_data = {}
    if ext.get("extraction_json"):
        extraction_data = json.loads(ext["extraction_json"])

    ctx = base_context(request, user, extraction=ext, extraction_data=extraction_data, page="review")
    return templates.TemplateResponse(request, "review.html", ctx)


@app.post("/api/extraction/{extraction_id}/approve")
async def approve_extraction(extraction_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    ext = get_extraction(extraction_id)
    if not ext:
        return JSONResponse({"error": "Not found"}, status_code=404)

    # Get updated data from request body
    body = await request.json()
    extraction_data = body.get("extraction", json.loads(ext["extraction_json"] or "{}"))

    # Generate report
    from app.report import generate_report
    report_path = str(OUTPUT_DIR / f"{extraction_id}_report.xlsx")
    generate_report(extraction_data, report_path)

    update_extraction(
        extraction_id,
        status="complete",
        extraction_json=json.dumps(extraction_data),
        report_path=report_path,
        completed_at=datetime.utcnow().isoformat(),
    )

    return JSONResponse({"id": extraction_id, "status": "complete", "report_path": report_path})


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    extractions = get_user_extractions(user["id"], limit=50)
    completed = [e for e in extractions if e["status"] == "complete"]
    ctx = base_context(request, user, reports=completed, page="reports")
    return templates.TemplateResponse(request, "reports.html", ctx)


@app.get("/api/report/{extraction_id}/download")
async def download_report(extraction_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    ext = get_extraction(extraction_id)
    if not ext or not ext.get("report_path"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    if ext["user_id"] != user["id"] and not user["is_admin"]:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    filename = f"DrawScan_Report_{ext['filename'].replace('.pdf', '')}.xlsx"
    return FileResponse(ext["report_path"], filename=filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not user["is_admin"]:
        return RedirectResponse("/dashboard", status_code=303)

    users = get_all_users()
    all_extractions = get_all_extractions(limit=30)
    global_stats = get_stats()

    # Per-user usage
    for u in users:
        u["monthly_usage"] = get_user_monthly_usage(u["id"])

    ctx = base_context(request, user, all_users=users, all_extractions=all_extractions, global_stats=global_stats, page="admin")
    return templates.TemplateResponse(request, "admin.html", ctx)


@app.post("/api/admin/users")
async def create_user(request: Request):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    body = await request.json()
    email = body.get("email", "").strip()
    name = body.get("name", "").strip()
    password = body.get("password", "")
    sla_tier = body.get("sla_tier", "starter")
    is_admin = body.get("is_admin", False)

    if not email or not name or not password:
        return JSONResponse({"error": "Email, name, and password are required"}, status_code=400)

    pw_hash, pw_salt = hash_password(password)
    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (email, name, password_hash, password_salt, sla_tier, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (email, name, pw_hash, pw_salt, sla_tier, int(is_admin), datetime.utcnow().isoformat()),
        )
        db.commit()
    except Exception as e:
        db.close()
        return JSONResponse({"error": f"Failed to create user: {e}"}, status_code=400)
    db.close()
    return JSONResponse({"success": True})


@app.put("/api/admin/users/{user_id}")
async def update_user(user_id: int, request: Request):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    body = await request.json()
    db = get_db()
    sets = []
    vals = []
    for field in ("name", "email", "sla_tier", "is_active"):
        if field in body:
            sets.append(f"{field} = ?")
            vals.append(body[field] if field != "is_active" else int(body[field]))
    if body.get("password"):
        pw_hash, pw_salt = hash_password(body["password"])
        sets.extend(["password_hash = ?", "password_salt = ?"])
        vals.extend([pw_hash, pw_salt])

    if sets:
        vals.append(user_id)
        db.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", vals)
        db.commit()
    db.close()
    return JSONResponse({"success": True})


@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, request: Request):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if user_id == user["id"]:
        return JSONResponse({"error": "Cannot delete yourself"}, status_code=400)
    db = get_db()
    db.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
    db.commit()
    db.close()
    return JSONResponse({"success": True})
