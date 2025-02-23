document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesDiv = document.getElementById('messages');
    
    // Create scroll button
    const scrollBottomBtn = document.createElement('button');
    scrollBottomBtn.className = 'scroll-bottom-btn';
    scrollBottomBtn.innerHTML = '↓';
    document.body.appendChild(scrollBottomBtn);

    // Scroll management
    let isAutoScrolling = true;
    let scrollTimeout;
    let loadingIndicator = null;

    const checkScrollPosition = () => {
        const { scrollTop, scrollHeight, clientHeight } = messagesDiv;
        const isNearBottom = scrollHeight - (scrollTop + clientHeight) < 100;
        scrollBottomBtn.classList.toggle('visible', !isNearBottom);
        isAutoScrolling = isNearBottom;
    };

    const scrollToBottom = () => {
        messagesDiv.scrollTo({
            top: messagesDiv.scrollHeight,
            behavior: 'smooth'
        });
    };

    // Message handling
    function createMessageElement(message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'user-message' : 'bot-message';
        
        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        // Message content
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = message;

        messageDiv.appendChild(content);
        messageDiv.appendChild(timestamp);
        
        return messageDiv;
    }

    function showLoadingIndicator() {
        if (loadingIndicator) return;
        
        loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'bot-message loading-dots';
        
        // Create loading animation
        const dots = document.createElement('div');
        dots.className = 'loading-dots';
        for (let i = 0; i < 3; i++) {
            dots.appendChild(document.createElement('div'));
        }
        
        loadingIndicator.appendChild(dots);
        messagesDiv.appendChild(loadingIndicator);
        scrollToBottom();
    }

    function removeLoadingIndicator() {
        if (loadingIndicator) {
            loadingIndicator.remove();
            loadingIndicator = null;
        }
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message
        messagesDiv.appendChild(createMessageElement(message, true));
        userInput.value = '';
        scrollToBottom();
        
        try {
            // Show loading indicator
            showLoadingIndicator();
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            
            if (!response.ok) throw new Error('Server error');
            
            const data = await response.json();
            removeLoadingIndicator();
            
            // Add bot response
            const botMessage = createMessageElement(data.response, false);
            
            // Add quick replies if provided
            if (data.quickReplies) {
                const quickReplies = document.createElement('div');
                quickReplies.className = 'quick-replies';
                
                data.quickReplies.forEach(reply => {
                    const button = document.createElement('button');
                    button.className = 'quick-reply';
                    button.textContent = reply;
                    button.onclick = () => {
                        userInput.value = reply;
                        sendMessage();
                    };
                    quickReplies.appendChild(button);
                });
                
                botMessage.appendChild(quickReplies);
            }
            
            messagesDiv.appendChild(botMessage);
            scrollToBottom();
            
        } catch (error) {
            removeLoadingIndicator();
            const errorMessage = createMessageElement(
                '⚠️ Sorry, there was an error processing your request. Please try again.',
                false
            );
            messagesDiv.appendChild(errorMessage);
            scrollToBottom();
            console.error('Chat error:', error);
        }
    }

    // Event listeners
    messagesDiv.addEventListener('scroll', () => {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(checkScrollPosition, 100);
    });

    scrollBottomBtn.addEventListener('click', scrollToBottom);
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => e.key === 'Enter' && sendMessage());

    // Initial setup
    checkScrollPosition();
    scrollToBottom();
});

document.addEventListener('DOMContentLoaded', () => {
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');

    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        hamburger.classList.toggle('active');
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.nav-links') && !e.target.closest('.hamburger')) {
            navLinks.classList.remove('active');
            hamburger.classList.remove('active');
        }
    });

    // Close menu after clicking a link
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('active');
            hamburger.classList.remove('active');
        });
    });
});