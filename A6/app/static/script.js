$(document).ready(function() {
    $("#chat-form").on("submit", function(e) {
        e.preventDefault();
        const userInput = $("#user-input").val();
        if (!userInput) return;
        
        appendMessage(userInput, "user");
        $("#user-input").val("");
        
        // Send JSON request
        $.ajax({
            url: "/ask",
            method: "POST",
            contentType: "application/json",  // Set the correct Content-Type
            data: JSON.stringify({ question: userInput }),  // Send data as JSON
            success: function(response) {
                appendMessage(response.answer, "bot", response.source);
            },
            error: function(xhr) {
                appendMessage("Error: " + xhr.responseText, "bot");
            }
        });
    });
});

function askQuestion(question) {
    $("#user-input").val(question);
    $("#chat-form").submit();
}

function appendMessage(text, sender, source = "") {
    const messageClass = sender === "user" ? "user-message" : "bot-message";
    const sourceHtml = source ? `<small class="text-muted">Source: ${source}</small>` : "";
    
    const html = `
        <div class="message ${messageClass}">
            <div>${text}</div>
            ${sourceHtml}
        </div>
    `;
    
    $("#chat-messages").append(html);
    $("#chat-messages").scrollTop($("#chat-messages")[0].scrollHeight);
}