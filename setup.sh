#!/usr/bin/env bash
set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${BLUE}==>${NC} $1"; }
ok()    { echo -e "${GREEN} ok ${NC} $1"; }
warn()  { echo -e "${YELLOW}warn${NC} $1"; }
fail()  { echo -e "${RED}fail${NC} $1"; }

detect_os() {
  case "$(uname -s)" in
    Darwin) echo "macos" ;;
    Linux)  echo "linux" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
    *) echo "unknown" ;;
  esac
}

OS=$(detect_os)
info "Detected OS: $OS"

if [ "$OS" = "unknown" ]; then
  fail "Unsupported OS. This script needs macOS, Linux, or Git Bash on Windows."
  exit 1
fi

find_python() {
  for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
      major=${version%%.*}
      minor=${version##*.}
      if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if ! PYTHON=$(find_python); then
  fail "Python 3.11 or newer not found."
  case "$OS" in
    macos)   echo "  Install with: brew install python@3.12" ;;
    linux)   echo "  Install with: sudo apt install python3.12 python3.12-venv" ;;
    windows) echo "  Download from: https://www.python.org/downloads/" ;;
  esac
  exit 1
fi
ok "Using $($PYTHON --version)"

if [ -d ".venv" ]; then
  ok "Reusing existing .venv"
else
  info "Creating virtual environment"
  "$PYTHON" -m venv .venv
  ok "Created .venv"
fi

if [ "$OS" = "windows" ]; then
  VENV_PY=".venv/Scripts/python.exe"
  VENV_PIP=".venv/Scripts/pip.exe"
else
  VENV_PY=".venv/bin/python"
  VENV_PIP=".venv/bin/pip"
fi

info "Installing dependencies"
"$VENV_PIP" install --quiet --upgrade pip
"$VENV_PIP" install --quiet -e ".[dev]"
ok "Core dependencies installed"

INSTALL_TESSERACT=0
INSTALL_AWS=0
INSTALL_GOOGLE=0

for arg in "$@"; do
  case "$arg" in
    --tesseract) INSTALL_TESSERACT=1 ;;
    --aws)       INSTALL_AWS=1 ;;
    --google)    INSTALL_GOOGLE=1 ;;
    --all)       INSTALL_TESSERACT=1; INSTALL_AWS=1; INSTALL_GOOGLE=1 ;;
  esac
done

if [ "$INSTALL_TESSERACT" -eq 1 ]; then
  info "Setting up Tesseract backend"
  "$VENV_PIP" install --quiet -e ".[tesseract]"

  if command -v tesseract >/dev/null 2>&1; then
    ok "Tesseract binary found: $(tesseract --version 2>&1 | head -1)"
  else
    warn "Tesseract Python bindings installed, but the binary is missing."
    case "$OS" in
      macos)   echo "  Install with: brew install tesseract" ;;
      linux)   echo "  Install with: sudo apt install tesseract-ocr" ;;
      windows) echo "  Download from: https://github.com/UB-Mannheim/tesseract/wiki" ;;
    esac
  fi
fi

if [ "$INSTALL_AWS" -eq 1 ]; then
  info "Setting up AWS backends (Bedrock + Rekognition)"
  "$VENV_PIP" install --quiet -e ".[aws]"

  if ! command -v aws >/dev/null 2>&1; then
    warn "AWS CLI not found."
    case "$OS" in
      macos)   echo "  Install with: brew install awscli" ;;
      linux)   echo "  Install with: sudo apt install awscli" ;;
      windows) echo "  Download: https://awscli.amazonaws.com/AWSCLIV2.msi" ;;
    esac
  elif aws sts get-caller-identity >/dev/null 2>&1; then
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    ok "AWS credentials active (account $ACCOUNT)"

    info "Checking Bedrock model access"
    if aws bedrock list-foundation-models --region us-east-1 >/dev/null 2>&1; then
      ok "Bedrock reachable"
    else
      warn "Cannot list Bedrock models. Request model access:"
      echo "  1. Open the AWS Console -> Bedrock -> Model access"
      echo "  2. Request access to Anthropic Claude models"
      echo "  3. Wait for approval (usually instant for Claude)"
    fi
  else
    warn "AWS credentials not configured. To set them up:"
    echo "  1. IAM -> Users -> Create user -> name: captcha-crack-dev"
    echo "  2. Attach policies: AmazonBedrockReadOnly, AmazonRekognitionReadOnlyAccess"
    echo "  3. Security credentials -> Create access key -> CLI"
    echo "  4. Run: aws configure   (region us-east-1, output json)"
  fi
fi

if [ "$INSTALL_GOOGLE" -eq 1 ]; then
  info "Setting up Google backend (Vertex AI Gemini)"
  "$VENV_PIP" install --quiet -e ".[google]"

  if ! command -v gcloud >/dev/null 2>&1; then
    warn "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
  elif gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
    ok "gcloud authenticated"
    PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
    if [ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ]; then
      ok "Project: $PROJECT"
      info "Enabling Vertex AI API"
      gcloud services enable aiplatform.googleapis.com --quiet 2>/dev/null || warn "Could not enable API automatically"
    else
      warn "No project set. Run: gcloud config set project YOUR_PROJECT_ID"
    fi
  else
    warn "gcloud not authenticated. Run: gcloud auth application-default login"
  fi
fi

if [ ! -f ".env" ]; then
  info "Writing .env"
  cat > .env <<'ENVFILE'
CAPTCHA_CRACK_BACKEND=template
CAPTCHA_CRACK_FALLBACK_BACKENDS=
CAPTCHA_CRACK_HYBRID=false
CAPTCHA_CRACK_TEMPLATE_DIR=data/templates
CAPTCHA_CRACK_MAX_CANDIDATES=5
CAPTCHA_CRACK_CONFIDENCE_FLOOR=0.55
CAPTCHA_CRACK_AWS_REGION=us-east-1
CAPTCHA_CRACK_BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
CAPTCHA_CRACK_GCP_LOCATION=us-central1
CAPTCHA_CRACK_GEMINI_MODEL=gemini-1.5-flash
ENVFILE
  ok "Created .env"
else
  ok "Keeping existing .env"
fi

mkdir -p data/templates data/sample_captchas

info "Verifying installation"
if "$VENV_PY" -c "import captcha_crack, cv2, numpy" 2>/dev/null; then
  ok "Package imports cleanly"
else
  fail "Import check failed"
  exit 1
fi

info "Running test suite"
if "$VENV_PY" -m pytest -q 2>&1 | tail -3; then
  ok "Tests passed"
else
  warn "Some tests failed - the install is usable but check the output above"
fi

info "Backend availability"
"$VENV_PY" -m captcha_crack.cli backends || true

echo
ok "Setup verified"
echo
echo "Activate the environment:"
if [ "$OS" = "windows" ]; then
  echo "  source .venv/Scripts/activate"
else
  echo "  source .venv/bin/activate"
fi
echo
echo "Try it:"
echo "  captcha-crack solve data/sample_captchas/ --length 6"
echo "  captcha-crack stages path/to/captcha.png"
echo "  captcha-crack backends"
echo
echo "Optional backends (re-run this script with a flag):"
echo "  bash setup.sh --tesseract    local OCR"
echo "  bash setup.sh --aws          Bedrock Claude Vision + Rekognition"
echo "  bash setup.sh --google       Vertex AI Gemini Vision"
echo "  bash setup.sh --all          everything"
