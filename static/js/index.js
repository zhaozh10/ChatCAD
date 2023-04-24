const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");


chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    message = userInput.value

    const sendbtn = document.querySelector("#submit-input");
    const waitAnimation = document.querySelector("#wait-animation");
    sendbtn.style.display = "none";
    waitAnimation.style.display = "block";
    fetch("/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      }).then(function (response) {
        console.log(response);
        userInput.value = "";
        userInput.placeholder = "Invalid api key, please try again"
        sendbtn.style.display = "block";
        waitAnimation.style.display = "none";
        window.location.href = "/chat_index";
      });
});