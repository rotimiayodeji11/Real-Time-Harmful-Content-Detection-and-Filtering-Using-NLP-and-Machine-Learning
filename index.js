async function detectFakeNews() {
    const url = document.getElementById("urlInput").value;
    const resultDiv = document.getElementById("result");

    // Clear previous result
    resultDiv.textContent = "Checking...";

    if (!url) {
        resultDiv.textContent = "Please enter a URL.";
        return;
    }


    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            throw new Error("Error in prediction. Please check the URL and try again.");
        }

        const data = await response.json();
        resultDiv.textContent = `Prediction: ${data.prediction}`;
    } catch (error) {
        resultDiv.textContent = error.message;
    }
}