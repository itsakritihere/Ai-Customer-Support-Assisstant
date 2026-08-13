import { useState } from "react";
import "./App.css";

function App() {
  const [ticket, setTicket] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeTicket = async () => {
    if (!ticket.trim()) {
      setError("Please enter a customer support ticket.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ticket: ticket,
        }),
      });

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the AI service. Please make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setTicket("");
    setResult(null);
    setError("");
  };

  const useExample = () => {
    setTicket(
      "My card hasn't arrived yet and I have been waiting for two weeks."
    );
    setResult(null);
    setError("");
  };

  return (
    <div className="app">

      {/* ================= HEADER ================= */}

      <header className="navbar">

        <div className="brand">

          <div className="brand-icon">
            🤖
          </div>

          <div>
            <h1>AI Support Assistant</h1>
            <span>Intelligent Customer Support Automation</span>
          </div>

        </div>

        <div className="status">
          <span className="status-dot"></span>
          AI System Online
        </div>

      </header>


      {/* ================= MAIN ================= */}

      <main className="main-container">

        {/* Hero */}

        <section className="hero">

          <div className="hero-badge">
            NLP • Machine Learning • Generative AI
          </div>

          <h2>
            Turn customer tickets into
            <span> intelligent responses.</span>
          </h2>

          <p>
            Our AI analyzes customer support requests, identifies the
            appropriate category, and generates a context-aware response.
          </p>

        </section>


        {/* ================= INPUT CARD ================= */}

        <section className="ticket-card">

          <div className="section-header">

            <div>
              <h3>Customer Support Ticket</h3>

              <p>
                Describe the customer's problem below.
              </p>
            </div>

            <button
              className="example-button"
              onClick={useExample}
            >
              Try Example
            </button>

          </div>


          <textarea
            value={ticket}
            onChange={(e) => setTicket(e.target.value)}
            placeholder="Example: My card hasn't arrived yet and I have been waiting for two weeks..."
          />


          <div className="input-footer">

            <span>
              {ticket.length} characters
            </span>

            <div className="button-group">

              <button
                className="clear-button"
                onClick={clearAll}
              >
                Clear
              </button>

              <button
                className="analyze-button"
                onClick={analyzeTicket}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    ✨ Analyze Ticket
                  </>
                )}
              </button>

            </div>

          </div>


          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

        </section>


        {/* ================= LOADING ================= */}

        {loading && (

          <section className="loading-card">

            <div className="loading-icon">
              🤖
            </div>

            <div>

              <h3>Analyzing customer request...</h3>

              <p>
                Running NLP classification and generating an AI response.
              </p>

            </div>

          </section>

        )}


        {/* ================= RESULTS ================= */}

        {result && !result.error && (

          <section className="results-section">

            <div className="results-title">

              <div>
                <h3>Analysis Results</h3>
                <p>AI-powered ticket analysis</p>
              </div>

              <div className="success-badge">
                ✓ Analysis Complete
              </div>

            </div>


            {/* Result Cards */}

            <div className="result-grid">


              {/* Category */}

              <div className="result-card">

                <div className="result-icon category-icon">
                  🎯
                </div>

                <div>

                  <span>
                    Detected Category
                  </span>

                  <strong>
                    {result.category}
                  </strong>

                </div>

              </div>


              {/* Confidence */}

              <div className="result-card">

                <div className="result-icon confidence-icon">
                  📊
                </div>

                <div>

                  <span>
                    Model Confidence
                  </span>

                  <strong>
                    {result.confidence}%
                  </strong>

                </div>

              </div>


              {/* Label */}

              <div className="result-card">

                <div className="result-icon label-icon">
                  🏷️
                </div>

                <div>

                  <span>
                    Classification Label
                  </span>

                  <strong>
                    {result.prediction}
                  </strong>

                </div>

              </div>

            </div>


            {/* Confidence */}

            <div className="confidence-card">

              <div className="confidence-header">

                <div>
                  <strong>Classification Confidence</strong>
                  <p>
                    How confident the ML model is in its prediction.
                  </p>
                </div>

                <strong>
                  {result.confidence}%
                </strong>

              </div>


              <div className="progress-track">

                <div
                  className="progress-bar"
                  style={{
                    width: `${result.confidence}%`,
                  }}
                />

              </div>

            </div>


            {/* AI Response */}

            <div className="ai-response">

              <div className="response-header">

                <div className="response-icon">
                  🤖
                </div>

                <div>

                  <h3>
                    AI Generated Response
                  </h3>

                  <span>
                    Generated using Generative AI
                  </span>

                </div>

              </div>


              <div className="response-content">

                {result.ai_response}

              </div>

            </div>


            {/* Original Ticket */}

            <div className="original-ticket">

              <h4>
                Original Customer Ticket
              </h4>

              <p>
                "{ticket}"
              </p>

            </div>

          </section>

        )}

      </main>


      {/* ================= FOOTER ================= */}

      <footer>

        <p>
          AI Customer Support Assistant
        </p>

        <span>
          TF-IDF + Logistic Regression + Generative AI
        </span>

      </footer>

    </div>
  );
}

export default App;