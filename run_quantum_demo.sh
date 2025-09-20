# =======================================================================


#!/bin/bash

echo "🧬 Quantum-Enhanced AlphaFold3 Multi-Language Integration Demo"
echo "============================================================"

# Check if IBM Quantum API token is set
if [ -z "$IBM_QUANTUM_API_TOKEN" ]; then
    echo "⚠️  Warning: IBM_QUANTUM_API_TOKEN not set in environment"
    echo "   Please add your IBM Quantum API key to Replit Secrets"
else
    echo "✅ IBM Quantum API token configured"
fi

echo ""
echo "🔬 Available Technologies:"
echo "  • Julia: Main quantum-classical hybrid computation"
echo "  • Crystal: High-performance quantum TCP server"
echo "  • Python: Async wrapper and classical fallback"
echo "  • Erlang: Distributed actor-based quantum job management"
echo "  • Chapel: High-performance parallel quantum simulations"
echo "  • Agda: Formal verification of quantum algorithms"
echo "  • Coq: Mathematical proofs of quantum correctness"
echo "  • Balsa: Asynchronous hardware circuit design"

echo ""
echo "🚀 Starting quantum protein folding pipeline..."

# Run Julia main pipeline
echo "Starting Julia quantum computation..."
julia main.jl &
JULIA_PID=$!

# Give Julia time to start
sleep 2

# Run Python async wrapper
echo "Starting Python async wrapper..."
python3 quantum_integration.py &
PYTHON_PID=$!

# Run Crystal quantum server (if IBM token is available)
if [ ! -z "$IBM_QUANTUM_API_TOKEN" ]; then
    echo "Starting Crystal quantum server on port 8080..."
    crystal run quantum_server.cr -- $IBM_QUANTUM_API_TOKEN 8080 &
    CRYSTAL_PID=$!
fi

echo ""
echo "🎯 Integration Status:"
echo "  Julia PID: $JULIA_PID"
echo "  Python PID: $PYTHON_PID"
if [ ! -z "$CRYSTAL_PID" ]; then
    echo "  Crystal PID: $CRYSTAL_PID"
fi

echo ""
echo "🌐 Access points:"
echo "  • Crystal Quantum Server: http://0.0.0.0:8080/api/status"
echo "  • Python Classical Server: http://0.0.0.0:8000"

# Wait for Julia to complete
wait $JULIA_PID
echo "✅ Julia computation completed"

# Clean up background processes
if [ ! -z "$PYTHON_PID" ]; then
    kill $PYTHON_PID 2>/dev/null
fi
if [ ! -z "$CRYSTAL_PID" ]; then
    kill $CRYSTAL_PID 2>/dev/null
fi

echo "🏁 Demo completed!"

# =======================================================================


# =======================================================================
