# Test Files for AI Crime Evidence Organizer

These sample files demonstrate all features of the system:

## Files Included

### 1. `sample_witness_statement.txt`
- **Type:** Witness Statement
- **Content:** Testimony from Vikas Sharma
- **Features to Test:**
  - Entity extraction (names, locations, times)
  - Timeline building (8:12 PM, 8:15 PM, 8:30 PM, 8:35 PM, 9:00 PM)
  - Location extraction (MG Road, ABC Supermarket)
  - Event extraction (entered shop, heard argument, saw exit)

### 2. `sample_medical_report.txt`
- **Type:** Medical Report
- **Content:** Medical examination of victim
- **Features to Test:**
  - Medical entity extraction
  - Injury severity detection
  - Time extraction (10:30 PM, 10:45 PM, 8:30 PM)
  - Inconsistency detection (witness says "minor" vs medical says "deep laceration and fracture")

### 3. `sample_fir.txt`
- **Type:** FIR (First Information Report)
- **Content:** Official police complaint
- **Features to Test:**
  - Legal document classification
  - Entity extraction (multiple names, locations)
  - Timeline events
  - Description extraction (black shirt, white shirt)

### 4. `sample_police_memo.txt`
- **Type:** Police Investigation Memo
- **Content:** Investigation summary
- **Features to Test:**
  - Document classification
  - Inconsistency detection (time conflicts, injury severity)
  - Multiple timeline references
  - Evidence listing

## How to Use These Files

### Option 1: Convert to PDF (Recommended)

**Using Online Converter:**
1. Go to https://www.ilovepdf.com/txt-to-pdf or similar
2. Upload the `.txt` files
3. Convert to PDF
4. Use the PDFs in the system

**Using Python:**
```bash
pip install reportlab
python convert_to_pdf.py
```

### Option 2: Use as Text Files

The system accepts `.txt` files directly. Just upload them as-is.

### Option 3: Create Test Images

For testing image features:

1. **CCTV-like Image:**
   - Create an image with text overlay: "2024-11-15 20:12:34"
   - Add location text: "ABC Supermarket - MG Road"
   - Save as `cctv_test.jpg`

2. **Crime Scene Photo:**
   - Any image with EXIF data
   - System will extract timestamp and GPS if available

## Expected Outputs

### When Processing These Files:

1. **Entity Extraction:**
   - Names: Vikas Sharma, Rajesh Kumar, Ramesh Patel, Anil Singh, Mohan Das
   - Locations: MG Road, Bangalore, ABC Supermarket, Park Street
   - Times: Multiple timestamps extracted

2. **Timeline Construction:**
   - 8:12 PM - Suspect enters shop
   - 8:15 PM - Argument heard (witness)
   - 8:25 PM - Altercation (CCTV)
   - 8:30 PM - Suspect exits
   - 8:35 PM - Police called
   - 9:00 PM - Police arrive
   - 10:30 PM - Medical examination

3. **Inconsistencies Detected:**
   - **Time Conflict:** Witness says argument at 8:15 PM, CCTV shows 8:25 PM
   - **Injury Severity:** Witness says "minor injury", medical says "deep laceration and fracture"
   - **Clothing Change:** Suspect entered in black shirt, exited in white shirt

4. **Missing Evidence Suggestions:**
   - Forensic report
   - CCTV from neighboring shops
   - Additional witness statements
   - Phone records

## Testing Workflow

1. **Upload Files:**
   - Upload `sample_witness_statement.txt` (or PDF version)
   - Upload `sample_medical_report.txt`
   - Upload `sample_fir.txt`
   - Upload `sample_police_memo.txt`

2. **Process Each File:**
   - Use the same case_id for all files (or different ones to test separately)

3. **Get Report:**
   - View extracted text from each document
   - See entities extracted
   - Check timeline construction
   - Review inconsistencies detected
   - See missing evidence suggestions

## Creating PDFs from Text Files

I've included a Python script to convert these to PDFs. Run:

```bash
cd test_data
python create_test_pdfs.py
```

This will create PDF versions of all text files.

## Notes

- These files contain **fictional data** for testing purposes only
- All names, locations, and incidents are **fictional**
- Designed to trigger all system features:
  - Entity extraction
  - Timeline building
  - Inconsistency detection
  - Missing evidence suggestions

## What to Look For in Output

✅ **Text Extraction:** Full text from each document  
✅ **Entity Recognition:** Names, locations, dates extracted  
✅ **Timeline:** Chronological events from all documents  
✅ **Inconsistencies:** Time conflicts, injury severity conflicts  
✅ **Missing Evidence:** Suggestions for additional evidence  
✅ **Classification:** Documents correctly classified (witness statement, medical report, FIR, police memo)

### Sample Case Folder (Folder Upload)
Use the helper script to produce a sample case with documents and placeholder images and metadata.

```bash
cd test_data
python generate_sample_case.py
```

This will create `test_data/sample_case_CR_2024_001234/raw_files/` containing:
- `FIR.pdf` — converted from `sample_fir.txt`
- `witness_statement.pdf` — converted from `sample_witness_statement.txt`
- `medical_report.pdf` — converted from `sample_medical_report.txt`
- `police_memo.pdf` — converted from `sample_police_memo.txt`
- `person_photo.jpg` — a placeholder person photo
- `scene_photo.jpg` — a placeholder scene photo
- `case_metadata.json` — simple metadata file for the case

Then upload the zip or the files (multipart) using the `upload_folder` endpoint, or upload individually with `/evidence/upload` and process the case as a whole via `/evidence/process_case`.


