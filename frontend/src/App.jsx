import { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:3000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center px-4">
      <div className="bg-white p-8 rounded-2xl shadow-xl w-full max-w-lg">
        <h1 className="text-3xl font-bold text-center text-blue-600 mb-6">
          Sentiment Analysis
        </h1>

        <textarea
          className="w-full border border-gray-300 rounded-xl p-4 focus:outline-none focus:ring-2 focus:ring-blue-400 mb-4 resize-none h-32"
          placeholder="Enter text to analyze..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          onClick={handlePredict}
          disabled={loading}
          className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Predict"}
        </button>

        {result && (
          <div className="mt-6 text-center">
            <p className="text-xl">
              <span className="font-medium text-gray-700">Label:</span>{" "}
              <span
                className={`font-bold ${
                  result.label === "positive"
                    ? "text-green-600"
                    : "text-red-600"
                }`}
              >
                {result.label}
              </span>
            </p>
            <p className="text-gray-500">Confidence: {result.score.toFixed(4)}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
