document.getElementById('transactionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const transactionData = {
        amount: parseFloat(document.getElementById('amount').value),
        credit_limit: parseFloat(document.getElementById('credit_limit').value),
        use_chip: document.getElementById('use_chip').value,
        amount_ratio: parseFloat(document.getElementById('amount_ratio').value)
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transactionData)
        });

        const result = await response.json();
        
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('probability').textContent = 
            `${(result.probability * 100).toFixed(2)}%`;
        document.getElementById('prediction').textContent = result.prediction;
        document.getElementById('prediction').style.color = 
            result.prediction === "Fraud" ? "#e74c3c" : "#2ecc71";
        
        if(result.explanation) {
            document.getElementById('explanation').textContent = result.explanation;
        }
    } catch (error) {
        alert('Error processing request: ' + error.message);
    }
});
