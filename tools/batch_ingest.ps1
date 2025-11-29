# tools\batch_ingest.ps1
Set-Location (Split-Path -Path $MyInvocation.MyCommand.Definition -Parent)
# activate venv
& .\venv\Scripts\Activate.ps1

$uploads = Get-ChildItem -Path uploads -File -Include *.pdf,*.docx,*.txt
if ($uploads.Count -eq 0) { Write-Host "No files found in uploads/"; Read-Host "Press Enter to exit"; exit }

foreach ($f in $uploads) {
    Write-Host "Ingesting $($f.Name)..."
    python ingest_pdf_and_update.py "uploads\$($f.Name)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ingest script failed for $($f.Name). Aborting." -ForegroundColor Red
        break
    }
    if (Test-Path storage\documents_updated_from_pdf.json) {
        Move-Item -Path storage\documents_updated_from_pdf.json -Destination storage\documents.json -Force
        Write-Host "Updated storage/documents.json"
    } else {
        Write-Host "Warning: No updated JSON file found." -ForegroundColor Yellow
    }
}
Write-Host "Batch ingestion complete. Run python phase4_validator.py"
Read-Host "Press Enter to exit"
