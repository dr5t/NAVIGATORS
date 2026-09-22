#!/usr/bin/env python3
"""Build publication-quality A4 PDFs from Navigators markdown documentation."""

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PDF_DIR = DOCS / "pdf"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

DOCUMENT_MAPPING = [
    ("PRODUCT_REQUIREMENTS.md", "Navigators_Product_Requirements.pdf"),
    ("SYSTEM_DESIGN.md", "Navigators_System_Design.pdf"),
    ("ARCHITECTURE.md", "Navigators_Architecture.pdf"),
    ("TECHNICAL_APPROACH.md", "Navigators_Technical_Approach.pdf"),
    ("TECH_STACK.md", "Navigators_Technology_Stack.pdf"),
    ("NAVIGATION_MATHEMATICS.md", "Navigators_Navigation_Mathematics.pdf"),
    ("AI_ML_PIPELINE.md", "Navigators_AI_ML_Pipeline.pdf"),
    ("DATASET_AND_DATA_PROVENANCE.md", "Navigators_Dataset_and_Data_Provenance.pdf"),
    ("EXPERIMENTAL_METHODOLOGY.md", "Navigators_Experimental_Methodology.pdf"),
    ("EVALUATION.md", "Navigators_Evaluation.pdf"),
    ("RESEARCH_AND_METHODOLOGY.md", "Navigators_Research_and_Methodology.pdf"),
    ("BACKEND_ARCHITECTURE.md", "Navigators_Backend_Architecture.pdf"),
    ("DATABASE_DESIGN.md", "Navigators_Database_Design.pdf"),
    ("AUTHENTICATION_AND_RBAC.md", "Navigators_Authentication_and_RBAC.pdf"),
    ("CONTRIBUTOR_SYSTEM.md", "Navigators_Contributor_System.pdf"),
    ("MODEL_LIFECYCLE.md", "Navigators_Model_Lifecycle.pdf"),
    ("OFFLINE_ARCHITECTURE.md", "Navigators_Offline_Architecture.pdf"),
    ("TESTING.md", "Navigators_Testing.pdf"),
    ("SECURITY.md", "Navigators_Security.pdf"),
    ("DEPLOYMENT.md", "Navigators_Deployment.pdf"),
    ("REPRODUCIBILITY.md", "Navigators_Reproducibility.pdf"),
    ("LIMITATIONS.md", "Navigators_Limitations.pdf"),
    ("reports/PHASE_38_E2E_TEST_REPORT.md", "Navigators_Phase_38_E2E_Test_Report.pdf"),
]

CSS_TEMPLATE = """
@page {
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
    @top-right {
        content: "Navigators Technical Documentation";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #64748b;
    }
    @bottom-center {
        content: "Developed by Navigators";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #64748b;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #0f172a;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #0f172a;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
}

h2 {
    font-size: 14pt;
    font-weight: 600;
    color: #1e293b;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 4px;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p, ul, ol {
    margin-top: 0;
    margin-bottom: 8px;
}

code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
    font-size: 8.5pt;
    background-color: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    border: 1px solid #e2e8f0;
}

pre {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 10px;
    font-size: 8pt;
    overflow-x: auto;
    page-break-inside: avoid;
    margin-bottom: 10px;
}

pre code {
    background-color: transparent;
    border: none;
    padding: 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    margin-bottom: 14px;
    font-size: 8.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #cbd5e1;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
}

th {
    background-color: #f1f5f9;
    font-weight: 600;
    color: #1e293b;
}

tr:nth-child(even) td {
    background-color: #f8fafc;
}

blockquote {
    border-left: 3px solid #3b82f6;
    margin: 8px 0;
    padding: 6px 12px;
    background-color: #f8fafc;
    color: #334155;
    font-size: 9pt;
}

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
}
"""

def generate_pdf(src_rel: str, dest_name: str):
    src_file = DOCS / src_rel
    dest_file = PDF_DIR / dest_name
    if not src_file.exists():
        print(f"Error: {src_file} does not exist.")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_html = Path(tmpdir) / "doc.html"
        tmp_css = Path(tmpdir) / "style.css"
        tmp_css.write_text(CSS_TEMPLATE)

        pandoc_cmd = [
            "pandoc",
            str(src_file),
            "--standalone",
            "--css", str(tmp_css),
            "--metadata", "title=Navigators Technical Documentation",
            "-o", str(tmp_html)
        ]
        subprocess.run(pandoc_cmd, check=True)

        chrome_cmd = [
            CHROME_BIN,
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={dest_file}",
            str(tmp_html)
        ]
        subprocess.run(chrome_cmd, check=True, capture_output=True)

    size = dest_file.stat().st_size
    print(f"✓ Generated {dest_name} ({size:,} bytes)")
    return True

def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Building Navigators Publication PDFs")
    print("=" * 60)
    success = 0
    for src, dest in DOCUMENT_MAPPING:
        if generate_pdf(src, dest):
            success += 1
    print("=" * 60)
    print(f"Completed: {success}/{len(DOCUMENT_MAPPING)} PDFs successfully generated.")
    print("=" * 60)

if __name__ == "__main__":
    main()
