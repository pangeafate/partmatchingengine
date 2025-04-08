// DOM Elements
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const resetButton = document.getElementById('reset-chat');

// API Configuration
const API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:3000/api' 
    : window.location.origin + '/api';
let sessionId = null;

// Helper Functions
function createMessageElement(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role);
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    
    // Split the content by newlines and create paragraph elements
    const paragraphs = content.split('\n').filter(line => line.trim() !== '');
    paragraphs.forEach(paragraph => {
        const p = document.createElement('p');
        p.textContent = paragraph;
        messageContent.appendChild(p);
    });
    
    messageDiv.appendChild(messageContent);
    return messageDiv;
}

function addMessage(content, role) {
    const messageElement = createMessageElement(content, role);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createLoadingIndicator() {
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('message', 'assistant', 'loading');
    loadingDiv.id = 'loading-indicator';
    
    const loadingContent = document.createElement('div');
    loadingContent.classList.add('message-content');
    
    const loadingText = document.createElement('span');
    loadingText.textContent = 'Thinking';
    
    const loadingDots = document.createElement('div');
    loadingDots.classList.add('loading-dots');
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        loadingDots.appendChild(dot);
    }
    
    loadingContent.appendChild(loadingText);
    loadingContent.appendChild(loadingDots);
    loadingDiv.appendChild(loadingContent);
    
    return loadingDiv;
}

function showLoadingIndicator() {
    const loadingIndicator = createLoadingIndicator();
    chatMessages.appendChild(loadingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeLoadingIndicator() {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
}

// API Functions
async function sendMessage(message) {
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message,
                session_id: sessionId || 'default'
            })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        sessionId = data.session_id;
        return data.response;
    } catch (error) {
        console.error('Error sending message:', error);
        return 'Sorry, there was an error processing your request. Please try again.';
    }
}

async function resetChat() {
    try {
        await fetch(`${API_URL}/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        });
        
        // Clear the chat messages UI
        chatMessages.innerHTML = '';
        
        // Add the welcome message back
        addMessage('Welcome to the Part Matching Engine! Ask me anything about the industrial parts in our database.', 'system');
        
    } catch (error) {
        console.error('Error resetting chat:', error);
    }
}

// Event Listeners
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Clear input
    chatInput.value = '';
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Show loading indicator
    showLoadingIndicator();
    
    // Send message to API and get response
    const response = await sendMessage(message);
    
    // Remove loading indicator
    removeLoadingIndicator();
    
    // Add assistant response to chat
    addMessage(response, 'assistant');
});

resetButton.addEventListener('click', resetChat);

// Initialize health check
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('Backend is healthy');
        } else {
            console.error('Backend health check failed');
        }
    } catch (error) {
        console.error('Error checking backend health:', error);
    }
}

// Run on page load
window.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    console.log('Using API URL:', API_URL);
});