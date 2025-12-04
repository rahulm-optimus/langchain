"""
RAG System Diagnostic Tool
Checks all components and configurations
"""
import os
import sys
import subprocess
from dotenv import load_dotenv

print("="*70)
print("🔧 RAG SYSTEM DIAGNOSTIC TOOL")
print("="*70)

# Check 1: Python Version
print("\n1️⃣ Python Version:")
print(f"   {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ Python 3.8+ required")
else:
    print("   ✅ Python version OK")

# Check 2: Required Packages
print("\n2️⃣ Checking Required Packages:")
required_packages = {
    'langchain': 'langchain',
    'langchain-core': 'langchain_core',
    'langchain-chroma': 'langchain_chroma',
    'langchain-ollama': 'langchain_ollama',
    'langchain-text-splitters': 'langchain_text_splitters',
    'langchain-community': 'langchain_community',
    'chromadb': 'chromadb',
    'pypdf': 'pypdf',
    'python-dotenv': 'dotenv',
    'tqdm': 'tqdm'
}

missing_packages = []
for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
    except ImportError:
        print(f"   ❌ {package_name} - NOT INSTALLED")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n   📦 Install missing packages:")
    print(f"   pip install {' '.join(missing_packages)}")
else:
    print("\n   ✅ All packages installed")

# Check 3: Environment Variables
print("\n3️⃣ Environment Variables:")
load_dotenv()
model = os.getenv("model_2")
if model:
    print(f"   ✅ model_2 = {model}")
else:
    print("   ❌ model_2 not set in .env file")

# Check 4: Ollama Installation
print("\n4️⃣ Ollama Installation:")
try:
    result = subprocess.run(['ollama', '--version'], 
                          capture_output=True, 
                          text=True, 
                          timeout=5)
    print(f"   ✅ Ollama installed: {result.stdout.strip()}")
except FileNotFoundError:
    print("   ❌ Ollama not found. Install from: https://ollama.ai")
except Exception as e:
    print(f"   ⚠️ Error checking Ollama: {e}")

# Check 5: Ollama Service Status
print("\n5️⃣ Ollama Service Status:")
try:
    import requests
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        print("   ✅ Ollama service is running on http://localhost:11434")
        models = response.json().get('models', [])
        print(f"   📋 Available models: {len(models)}")
        for m in models[:5]:  # Show first 5
            print(f"      - {m['name']}")
    else:
        print(f"   ⚠️ Ollama responded with status: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ❌ Ollama service not running!")
    print("   💡 Start Ollama with: ollama serve")
except Exception as e:
    print(f"   ⚠️ Error connecting to Ollama: {e}")

# Check 6: Model Availability
print("\n6️⃣ Checking Model Availability:")
if model:
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if model in model_names or f"{model}:latest" in model_names:
                print(f"   ✅ Model '{model}' is available")
            else:
                print(f"   ❌ Model '{model}' not found")
                print(f"   💡 Pull model with: ollama pull {model}")
                print(f"   📋 Available models: {', '.join(model_names[:3])}")
    except Exception as e:
        print(f"   ⚠️ Could not verify model: {e}")

# Check 7: Test Embeddings
print("\n7️⃣ Testing Embeddings:")
if model and not missing_packages:
    try:
        from langchain_ollama import OllamaEmbeddings
        print("   🔄 Creating embedding model...")
        embeddings = OllamaEmbeddings(model=model)
        
        print("   🔄 Testing with sample text...")
        test_text = "Hello, this is a test."
        vector = embeddings.embed_query(test_text)
        
        print(f"   ✅ Embeddings working!")
        print(f"   📊 Vector dimension: {len(vector)}")
        print(f"   📊 Sample values: {vector[:3]}")
    except Exception as e:
        print(f"   ❌ Embedding test failed: {e}")
        import traceback
        print(f"   📋 Details: {traceback.format_exc()}")
else:
    print("   ⏭️ Skipped (missing requirements)")

# Check 8: Documents Folder
print("\n8️⃣ Documents Folder:")
docs_path = "./documents"
if os.path.exists(docs_path):
    files = os.listdir(docs_path)
    pdf_files = [f for f in files if f.endswith('.pdf')]
    txt_files = [f for f in files if f.endswith('.txt')]
    md_files = [f for f in files if f.endswith('.md')]
    
    print(f"   ✅ Folder exists")
    print(f"   📄 PDF files: {len(pdf_files)}")
    print(f"   📄 TXT files: {len(txt_files)}")
    print(f"   📄 MD files: {len(md_files)}")
    
    if pdf_files:
        print(f"   📋 PDF files found:")
        for f in pdf_files[:3]:
            size = os.path.getsize(os.path.join(docs_path, f)) / 1024
            print(f"      - {f} ({size:.1f} KB)")
    
    if not (pdf_files or txt_files or md_files):
        print("   ⚠️ No documents found!")
else:
    print(f"   ❌ Documents folder not found!")
    print(f"   💡 Create folder: mkdir {docs_path}")

# Check 9: Vector Store
print("\n9️⃣ Vector Store:")
chroma_path = "./chroma_db_new"
if os.path.exists(chroma_path):
    import os
    size = sum(os.path.getsize(os.path.join(chroma_path, f)) 
               for f in os.listdir(chroma_path) 
               if os.path.isfile(os.path.join(chroma_path, f))) / (1024 * 1024)
    print(f"   ✅ Vector store exists")
    print(f"   📊 Size: {size:.2f} MB")
    
    # Try to check database
    try:
        import sqlite3
        db_file = os.path.join(chroma_path, "chroma.sqlite3")
        if os.path.exists(db_file):
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            count = cursor.fetchone()[0]
            print(f"   📊 Embeddings count: {count}")
            conn.close()
    except Exception as e:
        print(f"   ⚠️ Could not read database: {e}")
else:
    print(f"   ℹ️ Vector store not created yet")
    print(f"   💡 Will be created on first run")

# Check 10: Memory & Disk Space
print("\n🔟 System Resources:")
try:
    import psutil
    
    # Memory
    mem = psutil.virtual_memory()
    print(f"   💾 RAM: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB ({mem.percent}%)")
    
    # Disk
    disk = psutil.disk_usage('.')
    print(f"   💿 Disk: {disk.free / (1024**3):.1f} GB free / {disk.total / (1024**3):.1f} GB total")
    
    if mem.percent > 90:
        print("   ⚠️ Low memory! Close other applications.")
    if disk.free / (1024**3) < 1:
        print("   ⚠️ Low disk space!")
except ImportError:
    print("   ℹ️ Install psutil for system resource info: pip install psutil")

# Summary
print("\n" + "="*70)
print("📋 DIAGNOSTIC SUMMARY")
print("="*70)

issues = []
if sys.version_info < (3, 8):
    issues.append("Python version too old")
if missing_packages:
    issues.append(f"Missing packages: {', '.join(missing_packages)}")
if not model:
    issues.append("model_2 not set in .env")
if not os.path.exists("./documents"):
    issues.append("Documents folder missing")

if issues:
    print("❌ ISSUES FOUND:")
    for issue in issues:
        print(f"   • {issue}")
    print("\n💡 Fix these issues before running the RAG system")
else:
    print("✅ ALL CHECKS PASSED!")
    print("\n🚀 System is ready to run:")
    print("   python example.py")

print("="*70)
