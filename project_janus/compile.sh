#!/bin/bash
#
# JANUS Paper Compilation Script
# Compiles janus.tex with full bibliography support
#
# Usage:
#   ./compile.sh          # Full compilation with bibliography
#   ./compile.sh --quick  # Skip bibliography (faster, no citations)
#   ./compile.sh --clean  # Remove all generated files

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_status() {
    echo -e "${BLUE}[JANUS]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if file exists
if [ ! -f "janus.tex" ]; then
    print_error "janus.tex not found in current directory"
    echo "Please run this script from technical_papers/project_janus/"
    exit 1
fi

# Parse arguments
MODE="full"
if [ "$1" == "--quick" ]; then
    MODE="quick"
    print_warning "Quick mode: skipping bibliography"
elif [ "$1" == "--clean" ]; then
    print_status "Cleaning generated files..."
    rm -f janus.aux janus.bbl janus.bcf janus.blg janus.log janus.out janus.run.xml janus.toc janus.pdf
    print_success "Cleanup complete"
    exit 0
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "JANUS Paper Compilation Script"
    echo ""
    echo "Usage:"
    echo "  ./compile.sh          Compile with full bibliography (recommended)"
    echo "  ./compile.sh --quick  Fast compilation without bibliography"
    echo "  ./compile.sh --clean  Remove all generated files"
    echo "  ./compile.sh --help   Show this help message"
    exit 0
fi

# Check for required tools
print_status "Checking dependencies..."

if ! command -v pdflatex &> /dev/null; then
    print_error "pdflatex not found. Please install a LaTeX distribution:"
    echo "  - Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  - macOS: brew install --cask mactex"
    echo "  - Windows: Install MiKTeX or TeX Live"
    exit 1
fi

if [ "$MODE" == "full" ]; then
    if ! command -v biber &> /dev/null; then
        print_warning "biber not found. Falling back to quick mode (no citations)"
        echo "To enable bibliography, install biber:"
        echo "  - Ubuntu/Debian: sudo apt-get install biber"
        echo "  - macOS: included in MacTeX"
        echo "  - Windows: MiKTeX Package Manager"
        MODE="quick"
    fi
fi

print_success "All dependencies found"

# Compilation
if [ "$MODE" == "full" ]; then
    print_status "Starting full compilation with bibliography..."

    print_status "Pass 1/4: Initial pdflatex run..."
    pdflatex -interaction=nonstopmode janus.tex > /dev/null 2>&1 || {
        print_error "First pdflatex pass failed. Check janus.log for details"
        exit 1
    }

    print_status "Pass 2/4: Processing bibliography with biber..."
    biber janus > /dev/null 2>&1 || {
        print_error "Biber failed. Check janus.blg for details"
        exit 1
    }

    print_status "Pass 3/4: Resolving citations..."
    pdflatex -interaction=nonstopmode janus.tex > /dev/null 2>&1 || {
        print_error "Second pdflatex pass failed"
        exit 1
    }

    print_status "Pass 4/4: Finalizing cross-references..."
    pdflatex -interaction=nonstopmode janus.tex > /dev/null 2>&1 || {
        print_error "Third pdflatex pass failed"
        exit 1
    }

    print_success "Compilation complete with full bibliography!"

else
    print_status "Starting quick compilation (no bibliography)..."

    pdflatex -interaction=nonstopmode janus.tex > /dev/null 2>&1 || {
        print_error "pdflatex failed. Check janus.log for details"
        exit 1
    }

    pdflatex -interaction=nonstopmode janus.tex > /dev/null 2>&1 || {
        print_error "Second pdflatex pass failed"
        exit 1
    }

    print_success "Quick compilation complete!"
fi

# Check output
if [ -f "janus.pdf" ]; then
    PDF_SIZE=$(du -h janus.pdf | cut -f1)
    PDF_PAGES=$(pdfinfo janus.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')

    echo ""
    print_success "PDF generated successfully"
    echo "  File: janus.pdf"
    echo "  Size: $PDF_SIZE"
    if [ -n "$PDF_PAGES" ]; then
        echo "  Pages: $PDF_PAGES"
    fi

    # Count citations if in full mode
    if [ "$MODE" == "full" ]; then
        CITE_COUNT=$(grep -o '\\cite[tp]\?\(\[[^]]*\]\)\?{[^}]*}' janus.tex | wc -l)
        echo "  Citations: ~$CITE_COUNT references"
    fi

    echo ""
    print_status "Opening PDF..."

    # Open PDF with default viewer
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open janus.pdf
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open janus.pdf 2>/dev/null || print_warning "Could not open PDF automatically"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        start janus.pdf
    fi

else
    print_error "PDF generation failed"
    echo "Check janus.log for error details"
    exit 1
fi

# Cleanup auxiliary files (optional)
print_status "Cleaning auxiliary files..."
rm -f janus.aux janus.bbl janus.bcf janus.blg janus.log janus.out janus.run.xml janus.toc

print_success "All done! 🚀"
echo ""
echo "Next steps:"
echo "  - Review janus.pdf"
echo "  - Run visualizations: cd examples && python3 visual_*.py"
echo "  - Push to GitHub for auto-compilation"
