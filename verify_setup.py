"""
verify_setup.py — Phase 1 Verification Script
==============================================
Run this after setting up Phase 1 to confirm everything works.

Usage: python verify_setup.py
"""

import sys
import subprocess

def check_python_version():
    """Python 3.9+ is required for modern type hints and features."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python {version.major}.{version.minor} — Need 3.9+")
        sys.exit(1)

def check_import(package: str, import_name: str = None):
    """Try importing a package and report success/failure."""
    import_name = import_name or package
    try:
        __import__(import_name)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} — Run: pip install {package}")

def check_config():
    """Verify config.yaml loads correctly."""
    try:
        import yaml
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        assert "embedding" in config, "Missing 'embedding' section"
        assert "llm" in config, "Missing 'llm' section"
        print("✅ config.yaml loads correctly")
    except FileNotFoundError:
        print("❌ config.yaml not found — create it in project root")
    except Exception as e:
        print(f"❌ config.yaml error: {e}")

def check_folder_structure():
    """Verify key folders exist."""
    from pathlib import Path
    required_folders = [
        "data/raw", "data/processed", "data/embeddings",
        "ingestion", "embeddings", "vectordb",
        "retrieval", "llm", "api", "ui", "utils", "tests"
    ]
    all_good = True
    for folder in required_folders:
        if Path(folder).exists():
            print(f"  ✅ {folder}/")
        else:
            print(f"  ❌ {folder}/ — Run: mkdir -p {folder}")
            all_good = False
    return all_good

print("=" * 50)
print("  Financial Copilot — Setup Verification")
print("=" * 50)

print("\n📌 Python Version:")
check_python_version()

print("\n📌 Key Packages:")
check_import("torch")
check_import("transformers")
check_import("sentence-transformers", "sentence_transformers")
check_import("faiss-cpu", "faiss")
check_import("fastapi")
check_import("streamlit")
check_import("loguru")
check_import("pyyaml", "yaml")
check_import("python-dotenv", "dotenv")

print("\n📌 Config File:")
check_config()

print("\n📌 Folder Structure:")
check_folder_structure()

print("\n" + "=" * 50)
print("If all ✅, you're ready for Phase 2!")
print("=" * 50)
