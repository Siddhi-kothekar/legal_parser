# Testing the API

Your server is running! Here's how to test it:

## Quick Test URLs

Open these in your browser:

1. **API Root:** http://127.0.0.1:8000/
   - Should show API info (you're seeing this ✅)

2. **API Documentation:** http://127.0.0.1:8000/docs
   - Interactive API testing interface

3. **Health Check:** http://127.0.0.1:8000/health
   - Should return: `{"status": "ok"}`

4. **Frontend (if served):** http://127.0.0.1:8000/static/index.html

## Test Endpoints Using Browser

### 1. Health Check
```
http://127.0.0.1:8000/health
```

### 2. API Documentation (Interactive)
```
http://127.0.0.1:8000/docs
```
- Click on endpoints to expand
- Click "Try it out" to test
- Enter parameters and click "Execute"

## Test Using API Documentation

1. Open http://127.0.0.1:8000/docs
2. Find `/evidence/upload` endpoint
3. Click "Try it out"
4. Click "Choose File" and select a test file (image or PDF)
5. Click "Execute"
6. Copy the `case_id` and `stored_path` from response

## Test Using Frontend

### Option 1: Serve Frontend via FastAPI
The frontend should be available at:
```
http://127.0.0.1:8000/static/index.html
```

### Option 2: Use Python HTTP Server
```bash
cd frontend
python -m http.server 8080
```
Then open: http://localhost:8080/index.html

## Test Using curl (Command Line)

### Upload Evidence
```bash
curl -X POST "http://127.0.0.1:8000/evidence/upload" -F "file=@test.pdf"
```

### Upload Folder (multiple files or zip)
```bash
# Multiple files
curl -X POST "http://127.0.0.1:8000/evidence/upload_folder" -F "files=@file1.pdf" -F "files=@file2.jpg"

# Zip archive
curl -X POST "http://127.0.0.1:8000/evidence/upload_folder" -F "files=@evidence_folder.zip"
```

### Process Evidence
```bash
curl -X POST "http://127.0.0.1:8000/evidence/process" \
  -H "Content-Type: application/json" \
  -d '{"case_id": "your_case_id", "artifact_path": "storage/uploads/your_file.pdf"}'
```

### Process Case Folder (process all files in case data folder)
```bash
curl -X POST "http://127.0.0.1:8000/evidence/process_case" \
   -H "Content-Type: application/json" \
   -d '{"case_id":"your_case_id"}'
```

### List Case Files
```bash
curl "http://127.0.0.1:8000/evidence/cases/your_case_id/files"
```

### Get Report
```bash
curl "http://127.0.0.1:8000/reports/your_case_id"
```

## What You Should See

When you open http://127.0.0.1:8000/docs:
- A Swagger UI interface
- List of all available endpoints
- Ability to test each endpoint interactively
- Request/response examples

## Next Steps

1. **Test the API:** Open http://127.0.0.1:8000/docs
2. **Upload a test file:** Use the `/evidence/upload` endpoint
3. **Process it:** Use the `/evidence/process` endpoint
4. **Get report:** Use the `/reports/{case_id}` endpoint

The server is working! Now test the actual functionality.

