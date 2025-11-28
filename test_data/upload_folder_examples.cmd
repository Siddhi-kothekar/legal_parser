@echo off
REM Replace API_BASE with your server address, e.g., http://127.0.0.1:8000
SET API_BASE=http://127.0.0.1:8000

REM Example 1: Upload multiple files
curl -X POST "%API_BASE%/evidence/upload_folder" -F "files=@sample_witness_statement.txt" -F "files=@sample_police_memo.txt"

REM Example 2: Upload a zip archive
REM Create a zip first (PowerShell compress):
REM powershell -Command "Compress-Archive -Path sample_witness_statement.txt,sample_police_memo.txt -DestinationPath evidence.zip"
curl -X POST "%API_BASE%/evidence/upload_folder" -F "files=@evidence.zip"

REM The response will include a case_id. Use it to process the case:
REM curl -X POST "%API_BASE%/evidence/process_case" -H "Content-Type: application/json" -d "{\"case_id\": \"<case_id>\"}"

REM Poll for the report:
REM curl "%API_BASE%/reports/<case_id>"

pause
