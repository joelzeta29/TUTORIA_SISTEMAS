document.getElementById('submit-button').addEventListener('click', async () => {
    const userInput = document.getElementById('user-input').value;

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: userInput })
        });

        if (!response.ok) {
            throw new Error(`Error en la solicitud: ${response.statusText}`);
        }

        const data = await response.json();
        document.getElementById('response-output').innerText =
            `Emoción: ${data.emotion}\nRecomendación: ${data.recommendation}`;
    } catch (error) {
        console.error('Error:', error);
        alert('Hubo un error al procesar tu solicitud. Revisa la consola para más detalles.');
    }
});
