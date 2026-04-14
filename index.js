function sendMessage() {
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const userMessage = userInput.value;

    if (userMessage.trim() === '') return;

    // Display user message
    chatBox.innerHTML += `<div><strong>You:</strong> ${userMessage}</div>`;

    fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: userMessage
            })
        })
        .then(response => response.json())
        .then(data => {
            // Display AI response
            chatBox.innerHTML += `<div><strong>AI:</strong> ${data.response}</div>`;
        })
        .catch(error => {
            console.error('Error:', error);
        });

    userInput.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;
}